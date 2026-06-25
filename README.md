# SPECT Myocardial Perfusion Imaging — Deep Learning Classification

A deep learning implementation for automated classification of SPECT myocardial perfusion imaging (MPI) to detect perfusion abnormalities (myocardial ischemia and/or infarction).

This project is based on the methodology proposed in:

> Kaplan Berkaya, S., Ak Sivrikoz, I., & Gunal, S. (2020). Classification models for SPECT myocardial perfusion imaging. *Computers in Biology and Medicine*, 123, 103893. https://doi.org/10.1016/j.compbiomed.2020.103893

---

## Overview

SPECT MPI is the most widely used non-invasive imaging method for detecting coronary artery disease (CAD). Manual interpretation is time-consuming, experience-dependent, and subject to significant inter-observer variability. This project implements a computer-aided diagnosis (CAD) system to automate SPECT image classification into Normal or Abnormal, combining classical medical knowledge with modern deep learning.

Two classification approaches were implemented and evaluated:

- **DL-based model** — transfer learning with VGG16 pre-trained on ImageNet, fine-tuned for binary classification of SPECT images
- **Knowledge-based model** — a rule-based pipeline using RGB color thresholding, anatomical segmentation, and shape feature extraction (F5), designed to encode expert clinical knowledge

Both models were implemented independently and evaluated against the paper's reported benchmarks.
---

## Dataset

The dataset is publicly available on Kaggle, released by the original authors:

🔗 https://www.kaggle.com/selcankaplan/spect-mpi

| Attribute | Value |
|---|---|
| Total patients | 192 |
| Normal | 42 |
| Abnormal (Ischemia/Infarction) | 150 |
| Age range | 26–96 (avg 61.5) |
| Sex | 73M / 119F |
| Image views | Short Axis (SA), Horizontal Long Axis (HLA), Vertical Long Axis (VLA) |
| Split | 66% train / 17% validation / 17% test |

Each image is a summed stress+rest perfusion image in 897×976 resolution. Labels (Normal / Abnormal) were assigned by two expert nuclear medicine readers with >10 years of clinical experience.

---

## Methods

### DL-Based Model

Transfer learning with **VGG16** pre-trained on ImageNet. The final classification layers are replaced with a task-specific head for binary SPECT classification.

**Architecture:**
- Base: VGG16 (frozen feature extractor)
- Head: Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
- Optimizer: Adam (lr = 0.0001)
- Loss: Binary cross-entropy
- Class imbalance handled via computed class weights
- Early stopping on validation loss (patience = 5)

### Knowledge-Based Model

A fully rule-based pipeline that encodes expert reader logic directly into image processing steps:

1. **RGB-based color thresholding** — extracts regions of tracer uptake using two-stage thresholding at pixel values 110 and 185, targeting orange, pink, and purple regions that indicate blood flow boundaries
2. **Section extraction** — divides the SPECT image into SA (Short Axis), HLA (Horizontal Long Axis), and VLA (Vertical Long Axis) subsections
3. **LVC centroid localization** — identifies the left ventricular cavity and computes its centroid for spatial reference
4. **Extracardiac activity elimination** — detects and masks inferior wall artifacts that would otherwise produce false positives
5. **Segmentation into 6 regions** — each image element is divided into segments A–F at 60° intervals around the computed centroid
6. **Feature extraction (F5)** — the ratio of pixel counts in consecutive regions between stress and rest images is extracted and compared against classification thresholds

> **Note:** The original paper extracts five features (F1–F5). This implementation uses F5 only — the ratio of consecutive areas — which directly captures the blood flow reduction pattern most indicative of ischemia and infarction.

Classification rules applied to F5:

| F5 Range | Classification |
|---|---|
| 0.80 < F5 < 2.00 | Normal |
| 2.00 < F5 < 8.00 | Ischemia (Abnormal) |
| F5 ≤ 0.80 or F5 ≥ 8.00 | Infarction (Abnormal) |

---

## Results

Performance is reported as balanced accuracy to account for class imbalance (42 Normal vs 150 Abnormal patients).

| Model | Paper Accuracy | Our Accuracy |
|---|---|---|
| DL-based (VGG16) | 86% | 76% |
| Knowledge-based (F5 only) | 93% | **94.3%** |

The knowledge-based model **exceeded the paper's reported accuracy** despite using only one of the five features (F5) described in the original work — indicating that the ratio of consecutive perfusion areas is the most discriminative single feature for detecting abnormalities.

The DL model underperformed relative to the paper (76% vs 86%), which is consistent with the small training set (~127 images after the 66/17/17 split). The paper itself noted limited training data as the primary constraint on DL performance.
---

## Repository Structure
