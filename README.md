# federated-cardiomegaly-fairness

# Federated learning for cardiomegaly detection across unequal hospitals

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aylinkadkhodamanesh-dotcom/federated-cardiomegaly-fairness/blob/main/federated_cardiomegaly.ipynb)

Full code and outputs: [federated_cardiomegaly.ipynb](federated_cardiomegaly.ipynb)

Can hospitals train a shared chest X-ray model without sharing patient data, and who benefits when they do?

This project simulates four hospitals of very different size, case mix and image quality, and compares three ways of training a deep learning model to detect **cardiomegaly** (an enlarged heart, a common sign of heart failure):

1. **Local only**: each hospital trains its own model on its own data.
2. **Centralized**: all data pooled in one place. Best case, but not allowed in practice for privacy reasons.
3. **Federated (FedAvg)**: each hospital trains locally and only model weights are sent to a server and averaged. No image leaves its hospital.

A fourth run tests a simple fairness idea: giving every hospital an **equal vote** in the average instead of weighting by hospital size.

Every model is evaluated on each hospital's own test set, so performance is reported site by site, not only as one global number.

## Setup

**Data.** ChestMNIST (NIH ChestX-ray14 via MedMNIST, 28×28 pixels), label used: cardiomegaly. Positive cases are rare (about 2.5% of images), so the loss gives positive cases more weight and results are reported as **AUC** rather than accuracy.

**Simulated hospitals.** Each hospital differs in size, cardiomegaly rate and "scanner" appearance, to mimic real multi-centre data:

| Hospital | Training images | Cardiomegaly cases | Image appearance |
|---|---|---|---|
| H1_large | 30,000 | 1,200 (4.0%) | original |
| H2_medium | 20,000 | 400 (2.0%) | brighter |
| H3_small | 8,000 | 240 (3.0%) | lower contrast |
| H4_tiny | 3,000 | 45 (1.5%) | darker and noisy |

Each hospital also has its own test set (about 5,600 images, around 140 to 160 positive cases) with the same appearance.

![One cardiomegaly case per hospital](hospitals.png)

**Model.** A small CNN (three convolution blocks, 16/32/64 channels) built in PyTorch. It uses **GroupNorm instead of BatchNorm** on purpose: BatchNorm keeps running statistics that get mixed badly when models from very different hospitals are averaged.

**Training budget.** The same for every method: 8 passes over the data (8 epochs, or 8 federated rounds of one local epoch each), Adam optimizer, learning rate 0.001, fixed random seed.

## Results

Test AUC per hospital:

| Method | H1_large | H2_medium | H3_small | H4_tiny | Mean | Worst site | Gap (best − worst) |
|---|---|---|---|---|---|---|---|
| Local only | 0.836 | 0.753 | 0.732 | 0.628 | 0.737 | 0.628 | 0.208 |
| Centralized | 0.867 | 0.867 | 0.864 | 0.808 | 0.851 | 0.808 | 0.059 |
| FedAvg | 0.831 | 0.854 | 0.849 | 0.798 | 0.833 | 0.798 | 0.056 |
| FedAvg equal weights | 0.806 | 0.830 | 0.834 | 0.782 | 0.813 | 0.782 | 0.053 |

![AUC per hospital](auc_per_hospital.png)

![Learning curves](learning_curves.png)

## Findings

**1. Federated learning gets close to pooling the data.** FedAvg reached a mean AUC of 0.833, about 98% of the centralized model (0.851), without any image leaving its hospital.

**2. Small hospitals gain the most from collaboration.** Training alone, performance fell with hospital size, and the tiny hospital reached only 0.628. With FedAvg it reached 0.798 (+0.17), the medium and small hospitals gained about 0.10 and 0.12, and the large hospital stayed about the same (0.836 vs 0.831). The gap between the best and worst site shrank from 0.21 to 0.06.

**3. Equal weighting did not make the model fairer.** Giving each hospital an equal vote lowered AUC at every site, including the tiny hospital it was meant to help (0.782 vs 0.798 with standard FedAvg). The likely reason is that the tiny hospital's update comes from only 45 positive cases on noisy images, so weighting it equally adds noise to the shared model rather than useful signal. Fairness across sites seems to need smarter aggregation than simple reweighting, for example methods that adapt weights to each site's loss or keep some site-specific layers.

## Limitations

- One random seed only. Small differences (around 0.01 to 0.02 AUC) may not be meaningful; repeating with several seeds would give confidence intervals.
- 8 rounds only, and the learning curves are still rising, so all methods are probably undertrained.
- Hospitals are simulated by splitting one public dataset and applying artificial image changes. Real hospitals differ in more complex ways (scanners, patient populations, labeling practices).
- Low resolution (28×28) was used so the whole study runs on a CPU in about one hour.
- Fairness is measured across sites only. The dataset version used has no patient level attributes such as sex or age, so fairness across patient groups was not assessed.

## Tools

Python, PyTorch, NumPy, pandas, scikit-learn, Matplotlib, MedMNIST.

## Data source

Yang, J. et al. MedMNIST v2: A large scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data* (2023). ChestMNIST is derived from the NIH ChestX-ray14 dataset (Wang, X. et al., CVPR 2017).
