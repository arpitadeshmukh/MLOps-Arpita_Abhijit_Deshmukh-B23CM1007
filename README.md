# Assignment 5 — ViT LoRA Fine-tuning & Adversarial Attacks

**Name:** Arpita Abhijit Deshmukh &nbsp;|&nbsp; **Roll No:** B23CM1007

---

## Links

| Resource | URL |
|---|---|
| WandB — Q1 | [MLOPS Assignment-5](https://wandb.ai/b23cm1007-indian-institute-of-technology-jodhpur/MLOPS%20Assignment-5/workspace?nw=nwuserb23cm1007) |
| WandB — Q2 | [MLOPS-Assignment-5-Adversarial](https://wandb.ai/b23cm1007-indian-institute-of-technology-jodhpur/MLOPS-Assignment-5-Adversarial/runs/04wc8aso?nw=nwuserb23cm1007) |
| HuggingFace — Best LoRA Model | [arpita2desh/Vit-Lora-CIFAR100-MLOPS](https://huggingface.co/arpita2desh/Vit-Lora-CIFAR100-MLOPS) |
| HuggingFace — Adversarial Weights | [arpita2desh/Adversarial-cifar10-MLOPS](https://huggingface.co/arpita2desh/Adversarial-cifar10-MLOPS) |

---

## Repository Structure

```
Assignment5/
├── LoRA_Finetuning.py                  # Q1: ViT-S LoRA fine-tuning on CIFAR-100
├── Adversarial_Attack.py            # Q2: Adversarial attacks on CIFAR-10
├── requirements.txt
├── res.csv                   # Q1 experiment results
├── q2_results.json           # Q2 experiment results
└── B23CM1007_Arpita_Abhijit_Deshmukh_Ass5.pdf
```

---

## Setup

> All experiments must be run inside a Docker container.

### 1. Pull and run the Docker container

```bash
docker pull pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime bash
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt** contents:

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.0
peft>=0.7.0
optuna>=3.4.0
wandb>=0.16.0
adversarial-robustness-toolbox>=1.17.0
huggingface_hub>=0.19.0
tqdm
pandas
matplotlib
```

### 3. Login to WandB and HuggingFace

```bash
wandb login
huggingface-cli login
```

---

## Running the Code

### Q1 — ViT-S LoRA Fine-tuning on CIFAR-100

```bash
python LoRA_Finetuning.py
```

This will:
- Fine-tune the ViT-S classification head without LoRA (baseline)
- Run Optuna grid search over all 9 rank × alpha combinations (r ∈ {2,4,8}, α ∈ {2,4,8}, dropout = 0.1)
- Save the best model to `best_vit_lora_model/`
- Push the best model to HuggingFace
- Log all metrics to WandB
- Save experiment results to `res.csv`

### Q2 — Adversarial Attacks on CIFAR-10

```bash
python Adversarial_Attack.py
```

This will:
- Train ResNet-18 from scratch on CIFAR-10 (target ≥ 72% accuracy)
- Run FGSM attack from scratch and via IBM ART at ε ∈ {0.01, 0.05, 0.10}
- Train ResNet-34 adversarial detectors for PGD and BIM attacks
- Save all weights to `q2_weights/`
- Log metrics, visualizations, and 10-sample panels to WandB
- Save results to `q2_results.json`

---

## Results

### Q1 — Test Accuracy Summary

| LoRA Layers | Rank | Alpha | Dropout | Test Acc (%) | Trainable Params |
|---|---|---|---|---|---|
| Without LoRA (head only) | — | — | — | 79.54 | 38,500 |
| Q, K, V | 2 | 2 | 0.1 | 89.78 | 75,364 |
| Q, K, V | 2 | 4 | 0.1 | 89.76 | 75,364 |
| Q, K, V | 2 | 8 | 0.1 | 89.70 | 75,364 |
| **Q, K, V** | **4** | **2** | **0.1** | **90.27** | **112,228** |
| Q, K, V | 4 | 4 | 0.1 | 90.00 | 112,228 |
| Q, K, V | 4 | 8 | 0.1 | 90.11 | 112,228 |
| Q, K, V | 8 | 2 | 0.1 | 89.84 | 185,956 |
| Q, K, V | 8 | 4 | 0.1 | 89.98 | 185,956 |
| Q, K, V | 8 | 8 | 0.1 | 90.25 | 185,956 |

**Best configuration (Optuna):** Rank = 4, Alpha = 2, Dropout = 0.1 → **90.27%**

---

### Q2 — FGSM Attack Results

| ε | Scratch Acc (%) | ART Acc (%) |
|---|---|---|
| 0.01 | 58.86 | 67.31 |
| 0.05 | 22.16 | 35.77 |
| 0.10 | 8.55 | 21.94 |

### Q2 — Adversarial Detection Results

| Attack | Detection Acc (%) |
|---|---|
| PGD (IBM ART) | 100.00 |
| BIM (IBM ART) | 100.00 |
