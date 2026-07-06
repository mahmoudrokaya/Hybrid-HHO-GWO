# Documentation

# Hybrid HHO-GWO: A Lightweight Optimization Framework for IoT Intrusion Detection

---

# 1. Introduction

This repository contains the complete implementation of the proposed **Hybrid Harris Hawks Optimization–Grey Wolf Optimization (Hybrid HHO-GWO)** framework for lightweight intrusion detection in Internet of Things (IoT) environments.

The framework combines feature selection and classifier hyperparameter optimization within a unified optimization process to construct computationally efficient intrusion-detection models suitable for deployment on edge devices.

The implementation follows a six-stage experimental pipeline that progresses from dataset auditing to robustness evaluation across unseen cyberattack categories.

---

# 2. Objectives

The framework was designed to:

- analyze the structural characteristics of the CIC IoT-DIAD 2024 dataset;
- establish baseline intrusion-detection performance;
- compare multiple feature-selection strategies;
- jointly optimize feature subsets and classifier hyperparameters;
- construct lightweight machine learning models;
- evaluate deployment suitability for edge-IoT devices;
- assess robustness against previously unseen attack families.

---

# 3. Software Architecture

The project is organized into six independent experiments.

```
Dataset
   │
   ▼
Experiment 1
Dataset Audit
   │
   ▼
Experiment 2
Baseline Classification
   │
   ▼
Experiment 3
Feature Selection Comparison
   │
   ▼
Experiment 4
Hybrid HHO-GWO Optimization
   │
   ▼
Experiment 5
Edge Suitability Evaluation
   │
   ▼
Experiment 6
Cross-Attack Robustness
```

Each experiment is implemented as a standalone Python script.

---

# 4. Repository Structure

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
│
├── README.md
├── DOCUMENTATION.md
├── requirements.txt
└── LICENSE
```

---

# 5. Dataset

The experiments utilize the **CIC IoT-DIAD 2024** dataset.

Characteristics include:

- 33 attack types
- 7 attack families
- benign traffic
- packet-based traffic representation
- flow-based traffic representation
- traffic collected from 105 real IoT devices
- 84 flow-based features extracted using CICFlowMeter

---

# 6. Experimental Pipeline

## Experiment 1 — Dataset Audit

Purpose

- inspect dataset organization;
- identify attack categories;
- compare packet and flow branches;
- analyze feature distributions;
- evaluate missing values;
- summarize dataset statistics.

Outputs

- dataset summaries;
- feature inventories;
- attack distributions;
- packet-versus-flow comparisons;
- publication-quality figures.

---

## Experiment 2 — Baseline Classification

Purpose

Construct baseline DDoS-versus-Benign classifiers using all available features.

Algorithms

- Random Forest
- Extra Trees
- Logistic Regression

Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Experiment 3 — Feature Selection Benchmark

Purpose

Compare conventional and optimization-based feature-selection methods.

Methods

- All Features
- Mutual Information
- Random Forest Importance
- Harris Hawks Optimization
- Grey Wolf Optimization
- Hybrid HHO-GWO

Evaluation

- Accuracy
- Macro-F1
- ROC-AUC
- Number of selected features
- Feature reduction percentage
- Optimization time

---

## Experiment 4 — Proposed Hybrid HHO-GWO

Purpose

Simultaneously optimize

- feature subset;
- classifier type;
- classifier hyperparameters.

Optimized variables include

- selected features;
- number of estimators;
- maximum tree depth;
- minimum split size;
- minimum leaf size;
- maximum feature ratio.

Evaluation metrics

- Accuracy
- Macro-F1
- Recall
- False Positive Rate
- ROC-AUC
- Training time
- Inference time
- Optimization history

---

## Experiment 5 — Edge Suitability

Purpose

Evaluate deployment feasibility of lightweight models.

Compared models

- Full-feature model
- Hybrid HHO-GWO optimized model

Measured metrics

- Peak memory usage
- Model size
- Training time
- CPU inference latency
- Latency per sample
- Latency per 1000 samples
- Accuracy retention
- Macro-F1 retention

---

## Experiment 6 — Robustness Evaluation

Purpose

Assess whether models trained only on DDoS traffic generalize to unseen attacks.

Training classes

- DDoS
- Benign

Testing categories

- DoS
- Mirai
- Recon
- Spoofing
- Web-Based
- Brute Force

Evaluation

- Accuracy
- Macro-F1
- Attack Recall
- False Positive Rate
- Family-level detection rates

---

# 7. Hybrid HHO-GWO Optimization

The proposed optimization framework combines two complementary swarm intelligence algorithms.

## Harris Hawks Optimization

Provides

- adaptive exploitation;
- local refinement;
- rapid convergence.

## Grey Wolf Optimization

Provides

- global exploration;
- leadership-guided search;
- population diversity.

The hybrid strategy alternates between both behaviors to balance exploration and exploitation during optimization.

Each candidate solution simultaneously represents

- selected feature subset;
- classifier type;
- classifier hyperparameters.

---

# 8. Execution

Experiments should be executed sequentially.

```bash
python Codes/Experiment_1_CICIoT_Dataset_Audit_Attack_Selection.py

python Codes/Experiment_2_Baseline_DDoS_vs_Benign_Classification.py

python Codes/Experiment_3_Feature_Selection_HHO_GWO_Hybrid.py

python Codes/Experiment_4_Proposed_Hybrid_HHO_GWO_Lightweight_Model.py

python Codes/Experiment_5_Lightweight_Edge_Suitability.py

python Codes/Experiment_6_Robustness_Across_Attack_Categories.py
```

---

# 9. Output Files

Each experiment automatically creates dedicated output folders containing

- CSV tables;
- Excel summaries;
- publication-quality figures;
- confusion matrices;
- classification reports;
- optimization histories;
- selected feature lists;
- edge-deployment metrics;
- robustness summaries;
- Markdown reports.

---

# 10. Reproducibility

The implementation supports reproducible experimentation through

- fixed random seed;
- deterministic preprocessing;
- identical train-test splitting strategy;
- branch-separated evaluation;
- automated report generation;
- standardized evaluation metrics.

---

# 11. Software Requirements

- Python 3.10 or later

Required packages

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy
- OpenPyXL

Install using

```bash
pip install -r requirements.txt
```

---

# 12. Citation

If this repository contributes to your research, please cite the associated publication.

```
Mahmoud Rokaya.

Hybrid HHO-GWO:
A Lightweight Optimization Framework for Simultaneous
Feature Selection and Classifier Optimization
in IoT Intrusion Detection.

2026.
```

---

# 13. Contact

**Dr. Mahmoud Rokaya**

Associate Professor

College of Computers and Information Technology

Taif University

Saudi Arabia

GitHub:
https://github.com/mahmoudrokaya

Email:
baselmah@yahoo.com
