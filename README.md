# BHI DATA GENERATION REPO
## This repository includes the project's test files and model artifacts, featuring evaluations and comparisons across multiple models. It stands as a milestone marking my first recognition in an international competition.

## CTGAN Model Overview

**CTGAN (Conditional Tabular GAN)** was proposed by the SDV (Synthetic Data Vault) team in 2019. It is a Generative Adversarial Network (GAN) model designed to generate **structured tabular data**. The model addresses two core challenges that traditional GANs face when handling tabular data:

- Difficulty in modeling categorical variables
- Handling highly imbalanced discrete distributions

---

## Background and Motivation

Tabular data has the following characteristics:

- A mixture of continuous and categorical variables
- Distributions of categorical variables may be highly imbalanced 
- Complex interdependencies between features (e.g., nonlinear relations between age and income)

While traditional GANs have achieved success in image and text domains, directly applying them to tabular data introduces the following issues:

- One-hot encoding of categorical variables creates high-dimensional inputs, making training difficult
- Joint modeling of continuous and categorical variables is challenging
- Data sparsity makes training prone to collapse

CTGAN provides effective solutions to these problems.

---

## Model Principles and Key Design

### 1. Conditional Generator

CTGAN introduces a **conditional vector** during training to specify the value of a categorical variable in the current training sample.

This forces the GAN generator to learn data distributions **under specific category conditions**, helping to address class imbalance issues.

---

### 2. Mode-specific Normalization

For each continuous feature, instead of applying global mean-variance normalization, CTGAN normalizes **based on the distribution under the specific category condition**, avoiding distribution distortion.

This is designed to tackle the difficulty of modeling multimodal continuous distributions in tabular data.

---

### 3. Training by Sampling

Each training iteration **randomly samples a categorical feature and value as a condition**, constructing batch data so that the network learns balanced distributions across all categories.

---

### 4. Generator and Discriminator Architecture

- **The Generator takes as input:**
  - A noise vector `z` (typically from a standard Gaussian distribution)
  - A categorical condition vector `c` (indicating the class of the sample to generate)

  It outputs synthetic samples with the same dimensionality as real data.

- **The Discriminator determines whether a sample is real or synthetic**, while also considering the categorical condition vector `c`, forming a **conditional discrimination mechanism**.
