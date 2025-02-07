import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score  # Import scikit-learn's metric

# Add parent directory to path (assuming ultimate_ensemblev4.py is in the parent directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the model (make sure the file is named ultimate_ensemblev4.py)
from ultimate_ensemblev4 import UltimateEnsembleModelV4

#############################################
# 1. Dataset Definition for Evaluation
#############################################
class MURADatasetEval(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.body_parts = []
        # Assume each subfolder is named starting with "XR_"
        parts = [d for d in os.listdir(root_dir) if d.startswith('XR_')]
        for body_part in parts:
            part_dir = os.path.join(root_dir, body_part)
            for root, _, files in os.walk(part_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, file)
                        self.image_paths.append(full_path)
                        # Set label: assume folder name includes "positive" for abnormal cases
                        label = 1 if 'positive' in root.lower() else 0
                        self.labels.append(label)
                        self.body_parts.append(body_part)
                        
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
# 2. Evaluation Function
#############################################
def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    part_correct = defaultdict(int)
    part_total = defaultdict(int)
    
    # Lists to accumulate predictions and ground truth labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, body_parts, _ in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Use the first body part as representative in the forward pass.
            outputs, _ = model(inputs, body_parts[0])
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i, part in enumerate(body_parts):
                part_total[part] += 1
                if predicted[i] == labels[i]:
                    part_correct[part] += 1
            
            # Accumulate predictions and labels for Cohen's Kappa calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute Cohen's Kappa score
    kappa_score = cohen_kappa_score(all_labels, all_preds)
    
    avg_loss = running_loss / len(dataloader) if criterion is not None else None
    overall_acc = 100 * correct / total
    per_part_acc = {part: 100 * part_correct[part] / part_total[part] for part in part_total}
    return overall_acc, per_part_acc, avg_loss, kappa_score

#############################################
# 3. Main Evaluation Script
#############################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Set paths for validation data and model checkpoint
    val_dir = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\data\valid"
    checkpoint_path = os.path.join(os.path.dirname(val_dir), "ultimate_ensemblev4_best.pth")

    # Define transforms (same as used in training/validation)
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create evaluation dataset and DataLoader
    val_dataset = MURADatasetEval(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize the model and load checkpoint
    model = UltimateEnsembleModelV4(num_classes=2, beta=0.5).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Checkpoint not found. Exiting.")
        return

    # Optionally define criterion for loss reporting
    criterion = nn.CrossEntropyLoss()
    overall_acc, per_part_acc, avg_loss, kappa_score = evaluate(model, val_loader, device, criterion)
    print(f"Overall Validation Accuracy: {overall_acc:.2f}%")
    if avg_loss is not None:
        print(f"Average Validation Loss: {avg_loss:.4f}")
    print(f"Cohen's Kappa Score: {kappa_score:.4f}")
    print("Per-Body-Part Accuracy:")
    for part, acc in per_part_acc.items():
        print(f"  {part}: {acc:.2f}%")

if __name__ == "__main__":
    main()
