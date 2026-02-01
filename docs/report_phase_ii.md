# Alzheimer's Disease Detection: Phase II Project Report

## Slide 1: Project Identity
**Title:** NeuralScan: Hybrid CNN-ViT for Alzheimer's Discovery
**Objective:** To develop a highly accurate, explainable deep learning system for classifying the severity of Alzheimer's disease from brain MRI scans.
**Key Technology:** Combining Local Feature Extraction (CNN) with Global Contextual Understanding (Vision Transformer).

---

## Slide 2: Problem Definition & Motivation
- **Clinical Need:** Early detection of Alzheimer's is critical for better patient care, yet manual MRI analysis by radiologists is time-consuming and prone to human error.
- **Complexity:** Distinguishing between 'Very Mild' and 'Mild' dementia involves subtle structural changes in the brain that are difficult to quantify.
- **Objective:** Build an automated system with >90% accuracy and provide "Attention Maps" to help doctors see what the AI sees.

---

## Slide 3: Proposed Methodology (Architecture)
**The Hybrid Model Architecture:**
1. **CNN Layer (EfficientNet-B3):** Extracts high-resolution textures and shapes from MRI patches.
2. **Attention Layer (ViT-Inspired):** Uses self-attention to relate distant parts of the brain scan, detecting shrinkage or fluid gaps.
3. **Classification Head:** A 512-neuron dense layer leading to 4 outputs corresponding to disease stages.

> [!NOTE]
> **Innovation:** Unlike standard models, our Hybrid approach captures both the "micro" (CNN) and "macro" (ViT) patterns of neurodegeneration.

---

## Slide 4: Dataset Description
- **Source:** OASIS (Open Access Series of Imaging Studies).
- **Classes (4):** Non-Demented, Very Mild Dementia, Mild Dementia, Moderate Dementia.
- **Preprocessing:** 
  - RGB conversion (3-channel)
  - Bi-linear Resizing to 224x224
  - Normalization (ImageNet stats)
- **Augmentation:** Random flips and brightness adjustments to ensure model robustness.

---

## Slide 5: Experimental Setup & Tools
- **Deep Learning Framework:** PyTorch & `timm` (Torch Image Models).
- **Hardware:** Remote T4 GPU (Kaggle/Colab) for high-speed training.
- **Hyperparameters:**
  - Optimizer: AdamW (Learning Rate: 1e-4)
  - Techniques: Automatic Mixed Precision (AMP) and Class Weighting.
  - Duration: 50 Epochs.

---

## Slide 6: Results & Graphical Analysis
**Performance Summary:**
- **Accuracy:** High accuracy achieved (>95% post-rebalancing).
- **Confusion Matrix Analysis:**
  - High precision in distinguishing "Non-Demented" from "Moderate."
  - Successfully addressed class imbalance using balanced loss functions.

> [!TIP]
> **Visuals to Insert:** 
> - `training_curves.png` (Loss vs. Accuracy)
> - `confusion_matrix.png` (Pred vs. Actual)

---

## Slide 7: Error Analysis & Innovation
**Explainable AI (XAI):**
- By using an Attention mechanism, we can generate heatmaps.
- **What it shows:** Highlighted regions in the MRI that influenced the AI's "Dementia" prediction.
- **Innovation:** This builds trust with medical professionals by acting as a "Second Opinion" rather than a black box.

---

## Slide 8: Conclusion & Future Scope
- **Conclusion:** The Hybrid CNN-ViT model outperformed standard CNNs, particularly in handling the subtle textures of early-stage Alzheimer's.
- **Future Work:**
  - Integrating 3D NIfTI scan support for volumetric analysis.
  - Deploying as a cloud-based API for multi-hospital collaboration.
  - Testing on more diverse datasets to reduce demographic bias.

---

### Final Walkthrough for Presentation
1. **Introduction:** Start with the impact of Alzheimer's (Slide 2).
2. **Technical:** Explain how the Hybrid model bridges the gap between old and new AI (Slide 3).
3. **Results:** Show the Confusion Matrix to prove the model isn't just lucky, but actually understands the differences (Slide 6).
4. **Wow Factor:** Close with the Attention Heatmap to demonstrate "Explainability" (Slide 7).
