import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np

# Add parent directory to path (assuming ultimate_ensemblev4.py is there)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ultimate_ensemblev4 import UltimateEnsembleModelV4

#############################################
# 1. Dataset Definition (Same as before)
#############################################
class MURADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.body_parts = []
        parts = [d for d in os.listdir(root_dir) if d.startswith('XR_')]
        print("\nDataset composition:")
        for body_part in parts:
            part_dir = os.path.join(root_dir, body_part)
            part_count = 0
            for root, _, files in os.walk(part_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, file)
                        self.image_paths.append(full_path)
                        label = 1 if 'positive' in root.lower() else 0
                        self.labels.append(label)
                        self.body_parts.append(body_part)
                        part_count += 1
            print(f"{body_part}: {part_count} images")
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        body_part = self.body_parts[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, body_part, img_path

#############################################
# 2. Helper Functions
#############################################
def load_misclassified(file_path):
    misclassified_set = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    misclassified_set.add(path)
        print(f"Loaded {len(misclassified_set)} misclassified sample paths from {file_path}")
    else:
        print(f"Misclassified file not found at {file_path}")
    return misclassified_set

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#############################################
# 3. Training and Validation Functions with AMP and Gradient Accumulation
#############################################
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch_num, rl_loss_weight=0.1, use_mixup=True, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    part_correct = {}
    part_total = {}
    scaler = torch.cuda.amp.GradScaler()  # for mixed precision
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch_num}")
    for i, (inputs, labels, body_parts, img_paths) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply mixup augmentation if enabled
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
        with torch.cuda.amp.autocast():
            outputs, _ = model(inputs, body_parts[0])
            if use_mixup:
                loss_sup = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss_sup = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            reward = (predicted == labels).float() * 2 - 1.0
            
            try:
                rl_loss = model.compute_total_rl_loss(reward)
            except ValueError:
                rl_loss = 0.0 * loss_sup
            loss = loss_sup + rl_loss_weight * rl_loss
            loss = loss / accumulation_steps  # scale loss for accumulation
        
        scaler.scale(loss).backward()
        
        # Accumulate gradients and update every few steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += loss.item() * accumulation_steps
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for j, part in enumerate(body_parts):
            if part not in part_correct:
                part_correct[part] = 0
                part_total[part] = 0
            part_total[part] += 1
            if predicted[j] == labels[j]:
                part_correct[part] += 1

        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            "Loss": f"{loss.item()*accumulation_steps:.4f}",
            "Acc": f"{100 * correct / total:.2f}%",
            "LR": f"{current_lr:.6f}"
        })
        del inputs, labels, outputs, loss, loss_sup, rl_loss
        torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    part_accuracy = {part: 100 * part_correct[part] / part_total[part] for part in part_total}
    return epoch_loss, epoch_acc, part_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    part_correct = {}
    part_total = {}
    with torch.no_grad():
        for inputs, labels, body_parts, _ in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs, body_parts[0])
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for j, part in enumerate(body_parts):
                if part not in part_correct:
                    part_correct[part] = 0
                    part_total[part] = 0
                part_total[part] += 1
                if predicted[j] == labels[j]:
                    part_correct[part] += 1
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    part_accuracy = {part: 100 * part_correct[part] / part_total[part] for part in part_total}
    return epoch_loss, epoch_acc, part_accuracy

#############################################
# 4. Main Training Loop with CosineAnnealingWarmRestarts and Early Stopping
#############################################
def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 32  # Reduced batch size, but effective batch size increased via accumulation
    epochs = 150
    learning_rate = 1e-4
    num_classes = 2
    rl_loss_weight = 0.1
    mixup_enabled = True
    accumulation_steps = 4  # Adjust to simulate larger batch size

    train_dir = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\data\train"
    val_dir   = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\data\valid"
    misclassified_path = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\misclassified.txt"

    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomEqualize(p=1.0)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    misclassified_set = load_misclassified(misclassified_path)
    train_dataset = MURADataset(train_dir, transform=train_transform)
    val_dataset = MURADataset(val_dir, transform=val_transform)
    
    sample_weights = []
    for path in train_dataset.image_paths:
        if path in misclassified_set:
            sample_weights.append(2.0)
        else:
            sample_weights.append(1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    model = UltimateEnsembleModelV4(num_classes=num_classes, beta=0.5).to(device)
    
    # Instead of loading pretrained weights, train the model from scratch.
    # pretrained_path = os.path.join(os.path.dirname(train_dir), "enhanced_ensemble_best.pth")
    # if os.path.exists(pretrained_path):
    #     print("Loading pretrained weights from previous enhanced ensemble...")
    #     checkpoint = torch.load(pretrained_path, map_location=device)
    #     if 'model_state_dict' in checkpoint:
    #         model.rl_ensemble.load_state_dict(checkpoint['model_state_dict'], strict=False)
    #     else:
    #         model.rl_ensemble.load_state_dict(checkpoint, strict=False)
    #     print("Pretrained weights loaded successfully.")
    # else:
    #     print("No pretrained weights found, training from scratch.")
    print("Training model from scratch without pretrained weights.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0.0
    best_epoch = 0
    early_stopping_patience = 100
    
    print("\nStarting training...")
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss, train_acc, train_part_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch + 1, rl_loss_weight=rl_loss_weight, use_mixup=mixup_enabled, accumulation_steps=accumulation_steps)
        val_loss, val_acc, val_part_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch [{epoch+1}/{epochs}] - Time: {epoch_time/60:.2f} minutes")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print("Per-body-part Train Accuracy:")
        for part, acc in train_part_acc.items():
            print(f"  {part}: {acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("Per-body-part Validation Accuracy:")
        for part, acc in val_part_acc.items():
            print(f"  {part}: {acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_acc
            }
            save_path = os.path.join(os.path.dirname(train_dir), "ultimate_ensemblev4_best.pth")
            torch.save(checkpoint, save_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        if epoch - best_epoch > early_stopping_patience:
            print("Early stopping triggered.")
            break
        
        elapsed_time = time.time() - start_time
        estimated_total = elapsed_time * (epochs / (epoch + 1))
        estimated_remaining = estimated_total - elapsed_time
        print(f"\nTime elapsed: {elapsed_time/3600:.2f} hours")
        print(f"Estimated time remaining: {estimated_remaining/3600:.2f} hours")
        print(f"Estimated completion: {(datetime.now() + timedelta(seconds=estimated_remaining)).strftime('%Y-%m-%d %H:%M:%S')}")
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()
