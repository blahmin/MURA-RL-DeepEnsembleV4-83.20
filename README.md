# MURA-DeepEnsembleV4-83.20

## Overview  
MURA-DeepEnsembleV4-83.20 represents the latest iteration in my reinforcement learning-enhanced musculoskeletal X-ray classification model. Unlike previous iterations, due to its radical changes from the previous versions, **this model was trained entirely from scratch**, without initializing with pretrained weights from prior versions. Despite this fundamental shift, **V4 surpassed the previous record validation accuracy within just 52 epochs**, achieving its peak score at **epoch 69** with:  
- **Validation Accuracy:** 83.20%  
- **Cohen’s Kappa Score:** 0.6608  

A really nice observation in **V4** is the **reversal of the typical training vs. validation accuracy trend**. Unlike earlier models, where training accuracy remained consistently higher than validation accuracy by a sizeble margin (often 10%+), this model demonstrated **significantly higher validation accuracy compared to training accuracy**.  
- **Epoch 69 Example:** **Train Accuracy: 69.94% vs. Validation Accuracy: 83.20%**  
- This reversal suggests that the **new generalization techniques** introduced in V4, such as **CBAM attention, stronger augmentation, and refined gating networks**, have significantly reduced overfitting.  

## Model Performance  
### **Overall Results**
- **Validation Accuracy:** 83.20%  
- **Cohen’s Kappa Score:** 0.6608  
- **Average Validation Loss:** 0.4296  

### **Per-Body-Part Validation Accuracy**  
| Body Part  | Accuracy |
|------------|----------|
| **XR_ELBOW**  | 86.88% |
| **XR_FINGER** | 81.78% |
| **XR_FOREARM** | 84.67% |
| **XR_HAND** | 78.91% |
| **XR_HUMERUS** | 87.15% |
| **XR_SHOULDER** | 78.51% |
| **XR_WRIST** | 86.19% |

---

## **Key Advancements from V3 to V4**
MURA-DeepEnsembleV4 incorporates **major architectural and training improvements** aimed at **enhancing accuracy, generalization, and interpretability**.

### **1. Attention Mechanism – CBAM Integration**
- **What Changed:** **Convolutional Block Attention Module (CBAM)** was incorporated into the **BaseModel**.  
- **Technical Details:**  
  - CBAM consists of **channel and spatial attention** modules.  
  - It refines feature selection by enhancing **spatially and channel-wise informative features**.  
- **Benefit:** **Improves feature representation**, especially in **fine-grained medical image details**.  

### **2. Specialized Expert Branch – EfficientNet-B3**
- **What Changed:**  
  - **Replaced DenseNet121** with **EfficientNet-B3** as the expert model.  
  - Integrated **upsampling mechanisms** to adjust input resolution.  
- **Benefit:** EfficientNet-B3 provides **more efficient and robust feature extraction**.  

### **3. Gating Network Refinements**
- **What Changed:**  
  - Added an **extra hidden layer** to the **gating network** responsible for fusing ensemble and expert predictions.  
- **Technical Details:**  
  - The gating network now has a **two-layer fully connected structure** with **ReLU activations**.  
- **Benefit:** Enables a **more nuanced fusion strategy**, optimizing decision-making.  

### **4. Training Enhancements**
- **Advanced Data Augmentation**:  
  - Implemented **mixup augmentation** to improve feature variance.  
- **Mixed Precision Training (AMP)**:  
  - Used **Automatic Mixed Precision (AMP)** for faster training and reduced memory usage.  
- **Gradient Accumulation**:  
  - Simulated **larger batch sizes** for stability.  
- **New Learning Rate Scheduler**:  
  - **Switched to CosineAnnealingWarmRestarts** for dynamic learning rate adaptation.  
- **Early Stopping Implementation**:  
  - Improved training efficiency and prevented unnecessary overfitting.  

---

## **Training Pipeline**
- **Dataset:** MURA (Musculoskeletal Radiographs)  
- **Augmentations:** Mixup, Random Erasing, Color Jitter, Affine Transforms  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** AdamW  
- **Scheduler:** CosineAnnealingWarmRestarts  
- **Batch Size:** 32  

---

## **Installation**
Install dependencies:  
pip install torch torchvision albumentations pillow

python
Copy
Edit

Download the RL Model
The RL-trained model ultimate_ensemble_v4.pth is too large for GitHub version control and is available in GitHub Releases. Download manually or use:

python
Copy
Edit

## Future Improvements
The next iteration, DeepEnsembleV5, will refine multi-scale feature extraction, enhance specialized expert diversity, and introduce transformer-based gating for improved decision fusion.

Multi-Scale Feature Extraction:

A MultiScaleBlock aggregates features across different kernel sizes, improving the model's ability to capture both fine-grained and large-scale patterns in X-ray images.
CBAM Retained:

The CBAM attention module remains, ensuring that channel and spatial attention continue to refine feature representations.
Heterogeneous Specialized Experts:

Instead of a single expert model, V5 will incorporate two diverse expert branches (EfficientNet-B3 and DenseNet121), whose outputs will be averaged for better ensemble diversity.
Transformer-Based Gating:

The gating network will integrate a transformer encoder layer, processing concatenated predictions to generate more nuanced fusion weights for the final decision.
Overall Ensemble Structure Enhancements:

DeepEnsembleV5 will fuse the RL-inspired ensemble branch with specialized expert predictions through an optimized gating mechanism.

## Conclusion
MURA-DeepEnsembleV4 achieves the highest validation accuracy and kappa score thus far, demonstrating the effectiveness of training from scratch, CBAM attention, EfficientNet-B3 specialization, and refined training strategies. However, the inversion of training vs. validation accuracy dynamics highlights new challenges in underfitting, which future iterations will address.
