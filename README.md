# Optimizers for Hematological Image Classification (IJISTA)

Companion repository for the manuscript **“Comparative Analysis of Optimizers in CNNs for Hematological Image Classification”** (IJISTA).

This repository contains the complete five-optimizer benchmark: **SGD with momentum, RMSprop, Adam, AdamW and RAdam** on **ResNet18, EfficientNetV2-S and MobileNetV3-Large**, using **PBC, Naturalize 2K-PBC and Raabin-WBC**, with **10 seeds per configuration** and **450 training runs** in total.

The repository is intentionally limited to the five optimizers reported in the manuscript. No other optimizer is part of this benchmark.

---

## 1. Contents

```
.
├── README.md
├── Optimizers_IJISTA_Training.ipynb      # single training notebook (5 optimizers)
├── 1. ResNet18 - PBC (64x64)/
│   └── resultados_por_semilla.csv
├── 2. ResNet18 - Naturalize2K (96x96)/
│   └── resultados_por_semilla.csv
├── 3. ResNet18 - Raabin-WBC (128x128)/
│   └── resultados_por_semilla.csv
├── 4. EfficientNetV2 - PBC (64x64)/
│   └── resultados_por_semilla.csv
├── 5. EfficientNetV2 - Naturalize2K (96x96)/
│   └── resultados_por_semilla.csv
├── 6. EfficientNetV2 - Raabin-WBC (128x128)/
│   └── resultados_por_semilla.csv
├── 7. MobileNetV3 - PBC (64x64)/
│   └── resultados_por_semilla.csv
├── 8. MobileNetV3 - Naturalize2K (96x96)/
│   └── resultados_por_semilla.csv
├── 9. MobileNetV3 - Raabin-WBC (128x128)/
│   └── resultados_por_semilla.csv
└── Tablas_y_CSVs/
    ├── resumen_5_optimizadores.csv
    ├── semillas_5_optimizadores.csv
    └── wilcoxon_pareado_holm_5_optimizadores.csv
```

`Tablas_y_CSVs/semillas_5_optimizadores.csv` contains the 450 seed-level rows. The nine per-scenario files contain the same rows grouped by architecture and dataset.

---

## 2. Datasets

| Dataset | Classes | Resolution | Images | Train / Validation / Test | Split |
|---|---:|---:|---:|---|---|
| PBC (Peripheral Blood Cell) | 8 | 64 × 64 | 17,092 | 11,964 / 2,563 / 2,565 | Seeded random image-level 70/15/15 split |
| Naturalize 2K-PBC | 11 | 96 × 96 | 26,534 | 21,234 / 2,650 / 2,650 | Official `Train` / `Validate` / `Test` folders |
| Raabin-WBC | 5 | 128 × 128 | 16,633 | 11,643 / 2,494 / 2,496 | Seeded random image-level 70/15/15 split (official partition not used) |

### Class labels

- **PBC (8):** Basophil, Eosinophil, Erythroblast, Immature Granulocyte (IG), Lymphocyte, Monocyte, Neutrophil, Platelet. The original folder name `erythrocyte` is mapped to `erythroblast`.
- **Naturalize 2K-PBC (11):** Basophil, Banded Neutrophil, Eosinophil, Erythroblast, Lymphocyte, Meta-myelocyte, Monocyte, Myelocyte, Platelet, Pro-myelocyte, Segmented Neutrophil.
- **Raabin-WBC (5):** Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil.

### Dataset sources and citations

- **PBC:** A. Acevedo, A. Merino, S. Alférez, et al., “A dataset of microscopic peripheral blood cell images for development of automatic recognition systems,” *Data in Brief*, vol. 30, p. 105474, 2020. Kaggle: `unclesamulus/blood-cells-image-dataset`.
- **Naturalize 2K-PBC:** M. Abou Ali, F. Dornaika, and I. Arganda-Carreras, “Blood cell revolution: Unveiling 11 distinct types with ‘Naturalize’ augmentation,” *Algorithms*, vol. 16, no. 12, p. 562, 2023. Kaggle: `mohamadabouali1/blood-cells-dataset-11-classes-26534-images`.
- **Raabin-WBC:** Z. Mousavi Kouzehkanan et al., “A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm,” *Scientific Reports*, vol. 12, 2022. Kaggle: `raabindata/raabin-wbc`.

The datasets are public. Please cite the original dataset papers if you use them.

---

## 3. Experimental protocol

- **Epochs:** 80 per run.
- **Mini-batch size:** 64.
- **Loss:** cross-entropy.
- **Learning-rate scheduler:** none.
- **Early stopping:** none.
- **Reported metrics:** final-epoch test metrics.
- **Seeds (10):** `7, 42, 123, 456, 784, 999, 1337, 2026, 3407, 8888`.
- **Initialization:** all backbones are trained from scratch with random weights (no ImageNet pre-training).
- **Classifier head:** dropout (`p = 0.5`) followed by a linear layer matching the number of classes.
- **Gradient clipping:** `max_norm = 1.0`.
- **Normalization:** ImageNet channel statistics, mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`.
- **Class imbalance:** `WeightedRandomSampler` with inverse-frequency weights. The number of samples per epoch is `min_class_count × num_classes × factor`, with factor 2 for PBC and 3 for Naturalize 2K-PBC and Raabin-WBC.
- **Test metric:** accuracy, cross-entropy loss, macro F1, macro recall and macro one-vs-rest AUC. Accuracy standard deviations are sample standard deviations (`ddof = 1`).

### Data augmentation

| Dataset | Training augmentation |
|---|---|
| PBC | Resize; horizontal flip (p = 0.5); rotation (±30°); affine translation (0.1, 0.1) and scale (0.8–1.2); color jitter (brightness 0.3, contrast 0.3, saturation 0.2, hue 0.05); random resized crop (scale 0.7–1.0) |
| Naturalize 2K-PBC | Resize; horizontal flip (p = 0.5); rotation (±25°); affine translation (0.1, 0.1) and scale (0.8–1.2); random resized crop (scale 0.8–1.0); color jitter (brightness 0.3, contrast 0.3, saturation 0.3, hue 0.1); random grayscale (p = 0.1) |
| Raabin-WBC | Resize; horizontal flip; vertical flip; rotation (±15°); color jitter (brightness 0.2, contrast 0.2, saturation 0.2, hue 0.05) |

Validation and test images are only resized and normalized.

---

## 4. Optimizers and hyperparameters

| Optimizer | Learning rate | Other hyperparameters | Weight decay |
|---|---:|---|---:|
| SGD with momentum | 0.01 | momentum = 0.9 | 0.0 |
| RMSprop | 0.001 | alpha = 0.99, epsilon = 1e-8 | 0.0 |
| Adam | 0.001 | beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 | 0.0 |
| AdamW | 0.001 | beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 | 0.01 |
| RAdam | 0.001 | beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 | 0.0 |

These rates were fixed before the comparisons and were not tuned per optimizer, architecture or dataset. The AdamW–Adam contrast therefore combines the presence of weight decay with its decoupled implementation.

### Architectures

| Backbone | Parameters | Notes |
|---|---:|---|
| ResNet18 | 11.18 M | Identity skip connections |
| EfficientNetV2-S | 20.19 M | Fused MBConv stages |
| MobileNetV3-Large | 4.21 M | Depthwise-separable convolutions and squeeze-and-excitation attention |

---

## 5. Software

The original runs were executed in Google Colab on NVIDIA T4 and L4 GPUs. The training notebooks logged:

- **Python:** 3.13.9
- **PyTorch:** 2.14.0
- **torchvision:** 0.29.0 (bundled with PyTorch 2.14.0 in the runtime used here)
- **CUDA:** GPU runtime provided by Colab (T4 16 GB or L4 24 GB)
- **Other libraries:** NumPy, pandas, scikit-learn and Matplotlib as provided by the Colab runtime; exact versions were not logged.

The notebook also uses the Kaggle CLI only to download the public datasets. If the datasets are already available locally, no download is needed.

---

## 6. How to use the notebook

1. Open `Optimizers_IJISTA_Training.ipynb` in Google Colab or a local Jupyter environment with PyTorch installed.
2. If you want the notebook to download the datasets, set `PREPARE_DATA = True` and provide a valid `~/.kaggle/kaggle.json`. Otherwise, place the datasets under `DATA_ROOT` using the layout described in the notebook.
3. Set `RUN_ALL = True` to execute all nine scenarios, or edit `SELECTED_SCENARIOS` to run a subset.
4. The notebook writes one `resultados_por_semilla.csv` per scenario and the aggregate files `resumen_5_optimizadores.csv` and `wilcoxon_pareado_holm_5_optimizadores.csv`.

A full run is expensive: 450 trainings of 80 epochs. The notebook is designed so that a scenario or a single seed can be run independently.

### Expected dataset layout

```
data/
├── pbc/
│   └── full_dataset/
│       ├── basophil/
│       ├── eosinophil/
│       ├── erythroblast/
│       ├── ig/
│       ├── lymphocyte/
│       ├── monocyte/
│       ├── neutrophil/
│       └── platelet/
├── naturalize/
│   └── Naturalize 2K-PBC Dataset/
│       ├── Train/
│       ├── Validate/
│       └── Test/
└── raabin/
    └── full_dataset/
        ├── Basophil/
        ├── Eosinophil/
        ├── Lymphocyte/
        ├── Monocyte/
        └── Neutrophil/
```

---

## 7. Reported results

Values below are computed directly from `Tablas_y_CSVs/resumen_5_optimizadores.csv`. Accuracy is mean ± sample standard deviation across 10 seeds; loss and macro F1 are means. Bold marks the highest mean accuracy per architecture within each dataset. The complete per-seed values, including AUC and recall, are in the CSV files.

### Table 1. PBC (8 classes, 64 × 64)

| Architecture | Optimizer | Accuracy (%) | Loss | Macro F1 |
|---|---|---:|---:|---:|
| ResNet18 | Adam | 94.15 ± 0.31 | 0.187 | 0.935 |
| ResNet18 | AdamW | **94.53 ± 0.27** | 0.158 | 0.939 |
| ResNet18 | RAdam | 94.30 ± 0.34 | 0.169 | 0.936 |
| ResNet18 | SGD | 91.68 ± 0.26 | 0.269 | 0.898 |
| ResNet18 | RMSprop | 91.00 ± 0.26 | 0.294 | 0.892 |
| EfficientNetV2-S | Adam | 90.23 ± 0.24 | 0.302 | 0.887 |
| EfficientNetV2-S | AdamW | 90.31 ± 0.35 | 0.293 | 0.889 |
| EfficientNetV2-S | RAdam | 89.97 ± 0.31 | 0.305 | 0.885 |
| EfficientNetV2-S | SGD | 91.65 ± 0.25 | 0.260 | 0.892 |
| EfficientNetV2-S | RMSprop | **92.19 ± 0.29** | 0.242 | 0.908 |
| MobileNetV3-Large | Adam | 88.18 ± 0.18 | 0.374 | 0.869 |
| MobileNetV3-Large | AdamW | 88.58 ± 0.19 | 0.361 | 0.874 |
| MobileNetV3-Large | RAdam | 88.35 ± 0.22 | 0.368 | 0.871 |
| MobileNetV3-Large | SGD | 90.39 ± 0.20 | 0.311 | 0.889 |
| MobileNetV3-Large | RMSprop | **91.64 ± 0.18** | 0.275 | 0.905 |

### Table 2. Naturalize 2K-PBC (11 classes, 96 × 96)

| Architecture | Optimizer | Accuracy (%) | Loss | Macro F1 |
|---|---|---:|---:|---:|
| ResNet18 | Adam | 91.48 ± 0.17 | 0.252 | 0.921 |
| ResNet18 | AdamW | 91.92 ± 0.14 | 0.244 | 0.923 |
| ResNet18 | RAdam | 91.69 ± 0.12 | 0.245 | 0.922 |
| ResNet18 | SGD | **92.12 ± 0.23** | 0.257 | 0.928 |
| ResNet18 | RMSprop | 89.56 ± 0.25 | 0.378 | 0.901 |
| EfficientNetV2-S | Adam | 92.71 ± 0.18 | 0.214 | 0.928 |
| EfficientNetV2-S | AdamW | **93.03 ± 0.18** | 0.200 | 0.931 |
| EfficientNetV2-S | RAdam | 92.86 ± 0.11 | 0.209 | 0.929 |
| EfficientNetV2-S | SGD | 92.88 ± 0.22 | 0.213 | 0.933 |
| EfficientNetV2-S | RMSprop | 91.49 ± 0.23 | 0.276 | 0.917 |
| MobileNetV3-Large | Adam | 90.67 ± 0.18 | 0.289 | 0.908 |
| MobileNetV3-Large | AdamW | **91.01 ± 0.13** | 0.283 | 0.910 |
| MobileNetV3-Large | RAdam | 90.86 ± 0.13 | 0.286 | 0.909 |
| MobileNetV3-Large | SGD | 89.10 ± 0.18 | 0.358 | 0.893 |
| MobileNetV3-Large | RMSprop | 85.84 ± 0.20 | 0.418 | 0.869 |

### Table 3. Raabin-WBC (5 classes, 128 × 128)

| Architecture | Optimizer | Accuracy (%) | Loss | Macro F1 |
|---|---|---:|---:|---:|
| ResNet18 | Adam | 96.05 ± 0.20 | 0.119 | 0.927 |
| ResNet18 | AdamW | 95.95 ± 0.07 | 0.108 | 0.921 |
| ResNet18 | RAdam | 95.81 ± 0.08 | 0.112 | 0.918 |
| ResNet18 | SGD | 96.45 ± 0.23 | 0.111 | 0.932 |
| ResNet18 | RMSprop | **96.75 ± 0.14** | 0.156 | 0.934 |
| EfficientNetV2-S | Adam | 96.21 ± 0.14 | 0.125 | 0.936 |
| EfficientNetV2-S | AdamW | **96.85 ± 0.16** | 0.125 | 0.944 |
| EfficientNetV2-S | RAdam | 96.68 ± 0.13 | 0.124 | 0.940 |
| EfficientNetV2-S | SGD | 96.13 ± 0.12 | 0.127 | 0.929 |
| EfficientNetV2-S | RMSprop | 93.66 ± 0.14 | 0.219 | 0.889 |
| MobileNetV3-Large | Adam | 91.72 ± 0.15 | 0.250 | 0.874 |
| MobileNetV3-Large | AdamW | **91.93 ± 0.02** | 0.237 | 0.871 |
| MobileNetV3-Large | RAdam | 91.78 ± 0.02 | 0.238 | 0.868 |
| MobileNetV3-Large | SGD | 91.04 ± 0.21 | 0.367 | 0.872 |
| MobileNetV3-Large | RMSprop | 91.55 ± 0.17 | 0.239 | 0.881 |

### Statistical analysis

`Tablas_y_CSVs/wilcoxon_pareado_holm_5_optimizadores.csv` contains the 90 paired comparisons (10 optimizer pairs × 9 scenarios). Each comparison uses a two-sided paired Wilcoxon signed-rank test on the 10 seed-level accuracy values, with Holm correction within each scenario (`alpha = 0.05`). The file reports the mean difference in percentage points, win/tie/loss counts, the Wilcoxon statistics and the exact and Holm-adjusted p-values.

---

## 8. Scope and limitations

- The results characterize this fixed-rate, from-scratch protocol. They do not estimate each optimizer’s best achievable performance after tuning.
- PBC and Raabin-WBC use seeded random image-level splits; no patient-level grouping was available or used.
- Each dataset was evaluated at a single resolution (64, 96 or 128 pixels), so dataset and resolution effects are not separated.
- Adam used zero weight decay while AdamW used 0.01; the AdamW–Adam contrast is not a pure test of decoupled weight decay.
- Macro one-vs-rest AUC is reported per seed in the CSV files. AUC and accuracy need not rank models identically.
- No clinical superiority or generalization to other tasks, populations, scanners or laboratories is claimed.

---

## 9. Data and code availability

The datasets are public and are cited above. The experiment configurations and the results of every run are available in this repository: <https://github.com/marcelayanes/optimizers>.

## 10. Citation

If you use this repository, please cite the manuscript:

> M. de los Ángeles Yanes-Pérez et al., “Comparative Analysis of Optimizers in CNNs for Hematological Image Classification,” *IJISTA* (manuscript).

Please also cite the three dataset papers listed in Section 2.
