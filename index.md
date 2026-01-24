# ML–DLOps Assignment 1

**Name:** Arpita Abhijit Deshmukh  
**Roll No.:** B23CM1007  
**Email ID:** b23cm1007@iitj.ac.in

---

## Report & Code Links
- **Question 1 (Colab):** [Link](https://colab.research.google.com/drive/15jMj5_K2rrH81WaOkINfthO4JYMyoqOM?usp=sharing)
- **Question 2 (Colab):** [link](https://colab.research.google.com/drive/1Ov4oErNKU0z2_K5bxhefOYxpy49O3IIr)
- **GitHub Repository:** [Link](https://github.com/arpitadeshmukh/MLOps-Arpita_Abhijit_Deshmukh-B23CM1007/tree/Assignment1)

---

## Question 1(a): CNN Experiments

### Hyperparameters
| Parameter | Values |
|---------|--------|
| Batch Size | 16, 32 |
| Optimizer | SGD, Adam |
| Learning Rate | 0.001, 0.0001 |
| Model | ResNet-18, ResNet-50 |
| Pin Memory | True, False |
| Epochs | 5, 10 |

---

### MNIST Results
| Epochs | Pin | Batch | Optim | LR | ResNet-18 | ResNet-50 |
|------|-----|-------|-------|----|-----------|-----------|
| 5 | True | 16 | SGD | 0.001 | 99.34% | 99.13% |
| 5 | True | 16 | SGD | 0.0001 | 98.27% | 97.63% |
| 5 | True | 16 | Adam | 0.001 | 98.92% | 98.93% |
| 5 | True | 16 | Adam | 0.0001 | 99.11% | 99.05% |
| 5 | False | 32 | SGD | 0.001 | 99.03% | 98.93% |
| 5 | False | 32 | SGD | 0.0001 | 96.70% | 94.67% |
| 5 | False | 32 | Adam | 0.001 | 99.18% | 98.51% |
| 5 | False | 32 | Adam | 0.0001 | 99.09% | 99.15% |

---

### FashionMNIST Results
| Epochs | Pin | Batch | Optim | LR | ResNet-18 | ResNet-50 |
|------|-----|-------|-------|----|-----------|-----------|
| 10 | True | 16 | SGD | 0.001 | 91.50% | 89.89% |
| 10 | True | 16 | SGD | 0.0001 | 89.71% | 84.32% |
| 10 | True | 16 | Adam | 0.001 | 91.89% | 89.84% |
| 10 | True | 16 | Adam | 0.0001 | 91.62% | 91.52% |
| 5 | False | 32 | SGD | 0.001 | 90.39% | 88.18% |
| 5 | False | 32 | SGD | 0.0001 | 84.40% | 79.97% |
| 5 | False | 32 | Adam | 0.001 | 92.22% | 92.03% |
| 5 | False | 32 | Adam | 0.0001 | 90.35% | 89.98% |

---

### Key Observations (CNN)
- **MNIST:** Easy dataset; very high accuracy with little hyperparameter sensitivity.
- **FashionMNIST:** More complex; strongly affected by model and optimizer.
- **Model:** ResNet-18 matches or outperforms ResNet-50 with lower training cost.
- **Optimizer:** Adam consistently outperforms SGD.
- **Learning Rate:** 0.001 performs better than 0.0001.
- **Batch Size:** Smaller batch improves generalization.
- **Pin Memory:** No effect on accuracy.
- **Epochs:** FashionMNIST benefits from more epochs.

---

## Question 1(b): SVM Experiments

### MNIST Results
| Kernel | C | Degree | Accuracy | Time (ms) |
|------|---|--------|----------|-----------|
| RBF | 1 | – | 94.45% | 10285.51 |
| RBF | 10 | – | 95.55% | 8977.22 |
| Poly | 1 | 2 | 94.15% | 8220.28 |
| Poly | 1 | 3 | 93.40% | 9914.45 |
| Poly | 10 | 2 | 95.20% | 6378.44 |
| Poly | 10 | 3 | 94.10% | 8375.44 |

---

### FashionMNIST Results
| Kernel | C | Degree | Accuracy | Time (ms) |
|------|---|--------|----------|-----------|
| RBF | 1 | – | 86.00% | 9880.03 |
| RBF | 10 | – | 87.35% | 10376.90 |
| Poly | 1 | 2 | 84.75% | 8799.31 |
| Poly | 1 | 3 | 82.80% | 19842.46 |
| Poly | 10 | 2 | 86.80% | 8276.69 |
| Poly | 10 | 3 | 85.75% | 8479.50 |

---

### Key Observations (SVM)
- SVM performs well on MNIST but struggles on FashionMNIST.
- Increasing **C** improves accuracy for both datasets.
- **RBF kernel** performs best overall.
- Higher-degree polynomial kernels increase computation time without accuracy gain.

---

## Question 2: Device Experiments

| Device | Optim | ResNet-18 Acc | ResNet-50 Acc | Time R18 (ms) | Time R50 (ms) | FLOPs R18 | FLOPs R50 |
|------|-------|---------------|---------------|---------------|---------------|-----------|-----------|
| CPU | SGD | 82.10% | 81.01% | 8593745.54 | 26735498.56 | 1.82e9 | 4.13e9 |
| CPU | Adam | 87.20% | 88.10% | 9342563.67 | 27645175.89 | 1.82e9 | 4.13e9 |
| GPU | SGD | 84.58% | 79.55% | 332970.71 | 1005069.34 | 1.82e9 | 4.13e9 |
| GPU | Adam | 90.50% | 90.60% | 346891.90 | 1038062.73 | 1.82e9 | 4.13e9 |

---

### Key Observations
- GPU is significantly faster than CPU while FLOPs remain unchanged.
- Adam achieves higher accuracy than SGD across devices.
- ResNet-50 increases computation without consistent accuracy gains.

---

## Best Model
**ResNet-18 with Adam optimizer on GPU** provides the best trade-off between accuracy, training time, and computational efficiency.

---

## Learning Outcomes
- Learned how dataset complexity affects model choice and performance.
- Understood that deeper models do not always improve accuracy.
- Gained insight into optimizer, learning rate, and hardware effects in ML-DLOps pipelines.
