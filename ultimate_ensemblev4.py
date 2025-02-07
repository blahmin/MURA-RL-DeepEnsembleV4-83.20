import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#############################################
# 1. CBAM Module (Channel and Spatial Attention)
#############################################
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)
    
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

#############################################
# 2. BaseModel with CBAM-enhanced Feature Extractor
#############################################
class BaseModel(nn.Module):
    def __init__(self, num_classes=2):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.3)
        
        self.in_channels = 16
        self.layer1 = self._make_layer(16, 32, num_blocks=2)
        self.dropout2 = nn.Dropout(0.4)
        
        self.layer2 = self._make_layer(32, 64, num_blocks=2)
        self.dropout3 = nn.Dropout(0.4)
        
        self.layer3 = self._make_layer(64, 128, num_blocks=2)
        self.dropout4 = nn.Dropout(0.5)
        
        # Insert a CBAM block after the last convolutional layer
        self.cbam = CBAM(128)
        
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(128, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        # First block with downsampling
        layers.append(EnhancedBlock(self.in_channels, out_channels, stride=2))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(EnhancedBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.layer1(x)
        x = self.dropout2(x)
        x = self.layer2(x)
        x = self.dropout3(x)
        x = self.layer3(x)
        x = self.dropout4(x)
        
        # Apply CBAM
        x = self.cbam(x)
        
        att = self.attention(x)
        x = x * att
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.fc(x)

#############################################
# 3. EnhancedBlock (as before)
#############################################
class EnhancedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(EnhancedBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        # Residual shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

#############################################
# 4. RL-Inspired Ensemble Branch (same as before)
#############################################
class EnhancedEnsembleRL(nn.Module):
    def __init__(self, num_classes=2, beta=0.5):
        super(EnhancedEnsembleRL, self).__init__()
        self.model1 = BaseModel(num_classes)
        self.model2 = BaseModel(num_classes)
        self.model3 = BaseModel(num_classes)
        
        self.rl_weight_agent = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        self.part_attention = nn.ModuleDict({
            'SHOULDER': nn.Sequential(
                nn.Linear(num_classes * 3, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Softmax(dim=1)
            ),
            'ELBOW': nn.Sequential(
                nn.Linear(num_classes * 3, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Softmax(dim=1)
            )
        })
        
        self.default_weights = nn.Parameter(torch.ones(3) / 3)
        self.beta = beta
        
        self.rl_log_probs = None

    def forward(self, x, body_part=None):
        pred1 = self.model1(x)
        pred2 = self.model2(x)
        pred3 = self.model3(x)
        
        preds = torch.stack([pred1, pred2, pred3], dim=1)
        combined = torch.cat([pred1, pred2, pred3], dim=1)
        
        rl_weights = self.rl_weight_agent(combined)
        self.rl_log_probs = torch.log(rl_weights + 1e-8)
        
        if body_part is not None and body_part in self.part_attention:
            part_weights = self.part_attention[body_part](combined)
            weights = self.beta * rl_weights + (1.0 - self.beta) * part_weights
        else:
            weights = rl_weights
        
        weights = weights.unsqueeze(-1)
        weighted_preds = preds * weights
        ensemble_pred = torch.sum(weighted_preds, dim=1)
        
        return ensemble_pred

    def compute_rl_loss(self, reward):
        if self.rl_log_probs is None:
            raise ValueError("RL log probabilities not stored. Run a forward pass first.")
        rl_loss = -torch.mean(torch.sum(self.rl_log_probs, dim=1) * reward)
        return rl_loss

#############################################
# 5. Specialized Expert Branch (Using EfficientNet-B3)
#############################################
class SpecializedExpert(nn.Module):
    def __init__(self, num_classes=2):
        """
        Uses an EfficientNet-B3 architecture pretrained on ImageNet.
        The classifier is modified to output predictions for the given number of classes.
        """
        super(SpecializedExpert, self).__init__()
        # EfficientNet-B3 is available in torchvision (if using a recent version)
        self.model = models.efficientnet_b3(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # EfficientNet-B3 expects at least 300x300; upsample if necessary.
        if x.size(2) < 300 or x.size(3) < 300:
            x = F.interpolate(x, size=(300, 300), mode='bilinear', align_corners=False)
        return self.model(x)

#############################################
# 6. Gating Network to Fuse Predictions (Enhanced)
#############################################
class GatingNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

#############################################
# 7. Ultimate Ensemble Model V4
#############################################
class UltimateEnsembleModelV4(nn.Module):
    def __init__(self, num_classes=2, beta=0.5):
        super(UltimateEnsembleModelV4, self).__init__()
        self.num_classes = num_classes
        self.rl_ensemble = EnhancedEnsembleRL(num_classes=num_classes, beta=beta)
        self.specialist = SpecializedExpert(num_classes=num_classes)
        self.gating = GatingNetwork(num_classes=num_classes)
    
    def forward(self, x, body_part=None):
        main_pred = self.rl_ensemble(x, body_part)
        expert_pred = self.specialist(x)
        
        fusion_input = torch.cat([main_pred, expert_pred], dim=1)
        gate = self.gating(fusion_input)
        final_pred = gate * expert_pred + (1.0 - gate) * main_pred
        return final_pred, gate

    def compute_total_rl_loss(self, reward):
        return self.rl_ensemble.compute_rl_loss(reward)
