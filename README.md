# SPECT Myocardial Perfusion Imaging — Deep Learning Classification

A deep learning implementation for automated classification of SPECT myocardial perfusion imaging (MPI) to detect perfusion abnormalities (myocardial ischemia and/or infarction).

This project is based on the methodology proposed in:

> Kaplan Berkaya, S., Ak Sivrikoz, I., & Gunal, S. (2020). Classification models for SPECT myocardial perfusion imaging. *Computers in Biology and Medicine*, 123, 103893. https://doi.org/10.1016/j.compbiomed.2020.103893

---

## Overview

SPECT MPI is the most widely used non-invasive imaging method for detecting coronary artery disease (CAD). Manual interpretation is time-consuming, experience-dependent, and subject to significant inter-observer variability. This project implements a computer-aided diagnosis (CAD) system using deep learning to automate SPECT image classification.

Two classification approaches are explored:
- **DL-based model** — transfer learning with pre-trained CNNs, and an SVM classifier using deep/shallow feature extraction (own implementation)
- **Knowledge-based model** — rule-based approach using color thresholding, segmentation, and heuristics (as described in the paper)

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

Transfer learning is applied by replacing the final classification layers of pre-trained CNNs with task-specific layers adapted to 2-class SPECT classification.

**Pre-trained networks explored:**

| Network | Input Size | Depth |

| VGG-16 | 224×224 | 16 |

---

## Results

Performance is reported as balanced accuracy (arithmetic mean across classes, to handle class imbalance).

| Model | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Transfer Learning (VGG-16) | 0.86 | 1.00 | 0.71 |
| SVM + Deep Features (VGG-19) | 0.79 | 1.00 | 0.57 |
| SVM + Shallow Features (VGG-19) | **0.94** | 0.88 | **1.00** |
| Knowledge-Based (paper) | 0.93 | **1.00** | 0.86 |

The SVM with shallow VGG-19 features achieves the highest overall accuracy (94%). Using shallower features outperforms deeper ones on this small dataset because they are more general and less overfit to the ImageNet domain.

---

## Repository Structure
