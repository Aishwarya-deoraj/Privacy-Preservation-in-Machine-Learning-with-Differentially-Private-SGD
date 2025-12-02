# Privacy Preservation in Machine Learning with Differentially Private SGD

This repository contains the source code and accompanying materials for my term project:

> **“Privacy Preservation in Machine Learning with Differentially Private SGD:  
> A Survey and Experimental Analysis”**  
> **Course:** CSCI-B 544 – Security for Networked Systems  

The project investigates how **Differentially Private Stochastic Gradient Descent (DP-SGD)** mitigates training-data leakage in machine learning models, with a small experimental study on **MNIST** using **Opacus** (PyTorch).

---

## Repository Structure

- `SourceCode_adeoraj.ipynb`  
  Jupyter notebook containing all experiments:
  - DP-SGD training with Opacus
  - Non-DP baseline training
  - Noise multiplier sweep (`σ`)
  - Clipping norm (`C`) ablation
  - Training loss/accuracy curves
  - Simple membership-inference signal (confidence gap)
---

## Key Ideas

1. **Threat models**
   - **Membership Inference Attacks (MIA):** Decide whether a given sample was in the training set.
   - **Data Extraction:** Large language models memorizing and regurgitating rare training sequences.

2. **Defense**
   - **Differential Privacy (DP):** Limits how much model outputs can change when a single record is added/removed.
   - **DP-SGD:** Implements DP by combining **per-sample gradient clipping** with **Gaussian noise**.

3. **Experiments**
   - Accuracy and privacy (ε) as a function of **noise multiplier σ**.
   - Effect of **clipping norm C** on utility at fixed σ.
   - **Training dynamics** (loss, accuracy per epoch) for non-DP vs DP-SGD.
   - A simple **membership inference signal** via max softmax confidence on train vs test.

---

## Environment Setup

```bash
conda create -n dp-ml python=3.11 -y
conda activate dp-ml

# Core libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu version if needed
pip install opacus
pip install jupyter matplotlib numpy
