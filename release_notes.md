# Hybrid HHO-GWO v1.0.0

**Initial public release**

This release accompanies the research project:

> **Hybrid HHO-GWO: A Lightweight Optimization Framework for Simultaneous Feature Selection and Classifier Optimization in IoT Intrusion Detection**

---

## Overview

This repository presents a complete, reproducible implementation of the proposed **Hybrid Harris Hawks Optimization–Grey Wolf Optimization (Hybrid HHO-GWO)** framework for lightweight intrusion detection in Internet of Things (IoT) environments.

The framework jointly optimizes feature selection and classifier hyperparameters to construct accurate, computationally efficient, and edge-deployable intrusion detection models.

---

## What's Included

### Complete Experimental Pipeline

This release includes the full implementation of six sequential experiments:

- **Experiment 1** – CIC IoT-DIAD 2024 dataset auditing and attack-space analysis
- **Experiment 2** – Baseline DDoS versus Benign classification
- **Experiment 3** – Comparative evaluation of feature-selection methods
- **Experiment 4** – Proposed Hybrid HHO-GWO optimization framework
- **Experiment 5** – Lightweight edge deployment suitability analysis
- **Experiment 6** – Cross-attack robustness and generalization evaluation

---

## Implemented Methods

Feature-selection methods

- All Features
- Mutual Information
- Random Forest Feature Importance
- Harris Hawks Optimization (HHO)
- Grey Wolf Optimization (GWO)
- Proposed Hybrid HHO-GWO

Classification models

- Random Forest
- Extra Trees
- Logistic Regression

Optimization

- Simultaneous feature selection
- Hyperparameter optimization
- Lightweight model construction

---

## Generated Outputs

Each experiment automatically produces

- CSV result tables
- Publication-quality figures
- Confusion matrices
- Classification reports
- Selected feature lists
- Optimization histories
- Edge deployment metrics
- Robustness analysis
- Markdown reports

---

## Repository Contents

- Complete Python source code
- Documentation
- README
- Requirements file
- Reproducibility instructions
- MIT License

---

## Dataset

Experiments are based on the **CIC IoT-DIAD 2024** dataset, which contains:

- 33 attack types
- 7 attack families
- Packet-based traffic
- Flow-based traffic
- Traffic collected from 105 real IoT devices

---

## Reproducibility

The implementation supports reproducible experimentation through:

- Fixed random seeds
- Standardized preprocessing
- Consistent train/test splitting
- Automated output generation
- Publication-ready experimental pipeline

---

## Citation

If you use this software in your research, please cite the associated publication.

---

## License

Released under the **MIT License**.

---

## Authors

**Dr. Mahmoud Rokaya**

Associate Professor

College of Computers and Information Technology

Taif University

Saudi Arabia

GitHub: https://github.com/mahmoudrokaya

---

Thank you for your interest in the Hybrid HHO-GWO framework. Contributions, issues, and feedback are welcome.
