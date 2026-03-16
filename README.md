
# README

## NTIRE Image Denoising (σ = 50) – Team 13 Restormer Implementation

### Overview

This repository contains the implementation used by **Team 13** for the **NTIRE Image Denoising Challenge (σ = 50)**. The model is based on the **Restormer architecture**, which is a transformer-based network designed for high-resolution image restoration tasks.

Due to limited access to local GPU resources, the training and experimentation were conducted using **Kaggle GPU environments**. The final model weights were obtained from the best-performing checkpoint during training.

The primary notebook used for development and experimentation is:

```
team13_restormer.ipynb
```

This notebook contains the complete pipeline including dataset preparation, training, and model saving.

---

# Repository Structure

```
NTIRE2025_Dn50_challenge
│
├── models
│   └── team13_Restormer.py          # Restormer architecture used for inference
│
├── model_zoo
│   ├── team13_initialweights.pth    # Initial Restormer σ=50 pretrained weights
│   └── 13_Restormer.pth             # Best fine-tuned model after training
│   └── team13_Restormer.pth             # Best Model that was obtained during training
│
├── team13_restormer.ipynb           # Kaggle notebook used for training
│
├── add_noise.py                     # Script to generate noisy DIV2K dataset
│
└── README.md
```

---

# Dataset Requirements

The model was trained using the **DIV2K dataset**.

You must specify the dataset paths inside the training script before running training.

Example directory structure:

```
datasets
│
├── DIV2K_train_HR
│   └── (clean images)
│
└── DIV2K_train_noise50
    └── (noisy images generated using add_noise.py)
```

Inside the training script (`train_restormer.py`) the following paths must be updated:

```python
clean_dir = "./datasets/DIV2K_train_HR"
noisy_dir = "./datasets/DIV2K_train_noise50"
```

These paths should point to the locations where the **clean DIV2K dataset** and the **generated noisy dataset** are stored.

---

# Dataset Preparation

Noise is added to the DIV2K dataset using the script:

```
add_noise.py
```

This script generates synthetic Gaussian noise with:

```
σ = 50
```

to produce the noisy training dataset.

Example command:

```
python add_noise.py
```

This will generate the folder:

```
DIV2K_train_noise50
```

which is used during training.

---

# Training Pipeline

The training process follows these steps:

1. Load the **DIV2K clean dataset**
2. Generate **noisy images with σ = 50**
3. Initialize the **Restormer architecture**
4. Load **initial Restormer pretrained weights**
5. Train the model using noisy-clean image pairs
6. Monitor loss and PSNR during training
7. Save the **best-performing model checkpoint**

The best model is saved as:

```
model_zoo/13_Restormer.pth
```

---

# Model Initialization

The training begins using pretrained Restormer weights for Gaussian denoising (σ = 50):

```
model_zoo/team13_initialweights.pth
```

These weights serve as the starting point for fine-tuning on the DIV2K dataset.

---

# Inference / Evaluation

During evaluation, the script `test_demo.py`:

1. Creates the Restormer architecture
2. Loads the trained weights from:

```
model_zoo/13_Restormer.pth
```

3. Runs inference on the provided noisy dataset.

This ensures reproducible results consistent with the trained model.

---

# Training Environment

Training was performed using **Kaggle GPU environments** due to limited availability of local GPU resources.

The notebook used for this process is:

```
team13_restormer.ipynb
```

This notebook contains the complete training workflow including dataset generation, model training, and checkpoint saving.

---

# Notes

* The model architecture used is **Restormer**.
* The model was fine-tuned using **DIV2K with synthetic Gaussian noise (σ = 50)**.
* The best checkpoint obtained during training is included in `model_zoo`.

---
