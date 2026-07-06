# Hybrid HHO-GWO: A Lightweight Optimization Framework for IoT Intrusion Detection

## Overview

This repository contains the official implementation of the paper:

> **Hybrid HHO-GWO: A Lightweight Optimization Framework for Simultaneous Feature Selection and Classifier Optimization in IoT Intrusion Detection**

The proposed framework introduces a hybrid metaheuristic optimization strategy that combines **Harris Hawks Optimization (HHO)** and **Grey Wolf Optimization (GWO)** to simultaneously optimize:

- informative feature selection,
- classifier hyperparameters,
- lightweight model construction,

while preserving high intrusion-detection performance for deployment in resource-constrained Internet of Things (IoT) environments.

Unlike conventional intrusion detection systems that optimize feature selection or classifier parameters independently, the proposed framework jointly optimizes both components, producing compact, computationally efficient, and highly accurate intrusion-detection models suitable for edge computing.

---

# Highlights

- Hybrid Harris Hawks Optimization + Grey Wolf Optimization
- Joint feature selection and hyperparameter optimization
- Lightweight Random Forest / Extra Trees optimization
- Packet-based and Flow-based traffic evaluation
- Edge deployment suitability analysis
- Robustness evaluation across unseen attack families
- Fully reproducible experimental pipeline
- Publication-quality tables, figures, reports, and Excel summaries

---

# Repository Structure

```
Hybrid-HHO-GWO/
│
├── Codes/
│   ├── Experiment_1_CICIoT_Dataset_Audit_Attack_Selection.py
│   ├── Experiment_2_Baseline_DDoS_vs_Benign_Classification.py
│   ├── Experiment_3_Feature_Selection_HHO_GWO_Hybrid.py
│   ├── Experiment_4_Proposed_Hybrid_HHO_GWO_Lightweight_Model.py
│   ├── Experiment_5_Lightweight_Edge_Suitability.py
│   └── Experiment_6_Robustness_Across_Attack_Categories.py
│
├── Data/
│
├── Outputs/
│   ├── Experiment1/
│   ├── Experiment2/
│   ├── Experiment3/
│   ├── Experiment4/
│   ├── Experiment5/
│   └── Experiment6/
│
├── README.md
├── LICENSE
└── requirements.txt
```

---

# Dataset

The experiments use the **CICIoT2023** benchmark dataset.

The dataset contains

- Packet-Based traffic
- Flow-Based traffic
- 33 attack subtypes
- 7 attack families
- Benign traffic

Attack families include

- DDoS
- DoS
- Mirai
- Reconnaissance
- Spoofing
- Web-Based
- Brute Force

Both packet-based and flow-based branches are processed independently throughout all experiments because they exhibit distinct structural characteristics and predictive behavior. :contentReference[oaicite:3]{index=3}

---

# Methodology

The proposed framework consists of six sequential experiments.

## Experiment 1 — Dataset Audit and Attack Selection

Purpose

- audit CICIoT2023
- inspect attack distributions
- compare packet vs flow branches
- analyze feature schemas
- evaluate missing values
- determine suitable attack families

Outputs

- dataset summaries
- feature inventory
- attack distributions
- packet-flow comparison
- missing-value analysis

---

## Experiment 2 — Baseline DDoS Detection

Baseline classifiers

- Random Forest
- Extra Trees
- Logistic Regression

Evaluation metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Purpose

Establish baseline performance before feature optimization.

---

## Experiment 3 — Feature Selection Benchmark

Feature-selection methods

- All Features
- Mutual Information
- Random Forest Importance
- Harris Hawks Optimization (HHO)
- Grey Wolf Optimization (GWO)
- Hybrid HHO-GWO

Metrics

- Accuracy
- Macro-F1
- ROC-AUC
- Feature reduction
- Optimization time

Purpose

Evaluate whether Hybrid HHO-GWO provides better compactness while preserving predictive performance.

---

## Experiment 4 — Proposed Hybrid HHO-GWO Framework

Core contribution of the paper.

Simultaneously optimizes

- feature subset
- classifier selection
- classifier hyperparameters

Optimization variables

- feature subset
- number of estimators
- maximum tree depth
- minimum split size
- minimum leaf size
- maximum feature ratio
- classifier type

Evaluation

- Accuracy
- Macro-F1
- Recall
- False Positive Rate
- ROC-AUC
- Feature reduction
- Training time
- Inference time
- Optimization history

---

## Experiment 5 — Lightweight Edge Suitability

Evaluates deployment feasibility on edge devices.

Compared models

- Full-feature model
- Hybrid HHO-GWO optimized model

Measured

- Peak memory
- Model size
- CPU inference latency
- Training time
- Inference time
- Latency per sample
- Accuracy retention
- Macro-F1 retention

---

## Experiment 6 — Robustness Across Attack Categories

Training

- DDoS
- Benign

Testing

- DoS
- Mirai
- Recon
- Spoofing
- Web-Based
- Brute Force

Purpose

Evaluate cross-family generalization capability.

---

# Hybrid HHO-GWO Optimization

The proposed optimizer combines

### Harris Hawks Optimization

Provides

- adaptive exploration
- local exploitation
- escape from local minima

### Grey Wolf Optimization

Provides

- leadership-guided search
- global convergence
- exploitation of promising regions

The hybrid algorithm combines both search strategies to simultaneously optimize feature subsets and classifier configurations, producing lightweight yet highly accurate intrusion-detection models. :contentReference[oaicite:4]{index=4}

---

# Experimental Pipeline

```
Dataset Audit
        │
        ▼
Baseline Classification
        │
        ▼
Feature Selection Benchmark
        │
        ▼
Hybrid HHO-GWO Optimization
        │
        ▼
Lightweight Edge Evaluation
        │
        ▼
Cross-Attack Robustness Analysis
```

---

# Main Contributions

- Structural analysis of CICIoT2023
- Independent packet-based and flow-based modeling
- Hybrid HHO-GWO optimization
- Joint feature selection and hyperparameter optimization
- Lightweight IDS construction
- Edge deployment evaluation
- Cross-family robustness validation
- Fully reproducible experimental pipeline

These contributions collectively address predictive performance, computational efficiency, deployment feasibility, and robustness within a unified cyber-defense framework. :contentReference[oaicite:5]{index=5}

---

# Requirements

Python 3.10+

Main libraries

```
numpy
pandas
scikit-learn
matplotlib
scipy
pathlib
pickle
json
```

Install

```bash
pip install -r requirements.txt
```

---

# Running the Experiments

Execute sequentially.

```bash
python Experiment_1_CICIoT_Dataset_Audit_Attack_Selection.py
```

```bash
python Experiment_2_Baseline_DDoS_vs_Benign_Classification.py
```

```bash
python Experiment_3_Feature_Selection_HHO_GWO_Hybrid.py
```

```bash
python Experiment_4_Proposed_Hybrid_HHO_GWO_Lightweight_Model.py
```

```bash
python Experiment_5_Lightweight_Edge_Suitability.py
```

```bash
python Experiment_6_Robustness_Across_Attack_Categories.py
```

---

# Generated Outputs

Each experiment automatically creates

```
tables/
figures/
reports/
Excel workbooks/
CSV summaries/
Markdown reports/
```

Outputs include

- feature-selection results
- optimization history
- confusion matrices
- ROC statistics
- classification reports
- robustness analysis
- edge deployment metrics
- publication-ready figures

---

# Reproducibility

All experiments use

- fixed random seed
- identical preprocessing
- consistent train/test splitting
- branch-independent evaluation
- automated result generation

Each experiment is self-contained and can be executed independently.

---

# Citation

If you use this repository, please cite:

```
Mahmoud Rokaya,
Hybrid HHO-GWO: A Lightweight Optimization Framework for Simultaneous Feature Selection and Classifier Optimization in IoT Intrusion Detection,
2026.
```

---

# License

This repository is released under the MIT License.

---

# Contact

**Dr. Mahmoud Rokaya**

Associate Professor

College of Computers and Information Technology

Taif University

Saudi Arabia

GitHub:
https://github.com/mahmoudrokaya

Email:
baselmah@yahoo.com
