Source code description for the repository:

# Source Code

This repository contains the complete source code for the Hybrid HHO-GWO lightweight IoT intrusion-detection framework.

The implementation includes six reproducible Python experiments:

1. Dataset audit and attack selection
2. Baseline DDoS vs Benign classification
3. Feature-selection comparison using MI, RF importance, HHO, GWO, and Hybrid HHO-GWO
4. Proposed Hybrid HHO-GWO lightweight model
5. Edge suitability evaluation
6. Robustness testing across unseen attack categories

The code supports both packet-based and flow-based CIC IoT-DIAD 2024 / CICIoT-style traffic representations and generates all experimental outputs, including CSV tables, figures, classification reports, optimization histories, confusion matrices, model files, and reproducibility reports.

## Code Structure

```text
Codes/
├── Experiment_1_CICIoT_Dataset_Audit_Attack_Selection.py
├── Experiment_2_Baseline_DDoS_vs_Benign_Classification.py
├── Experiment_3_Feature_Selection_HHO_GWO_Hybrid.py
├── Experiment_4_Proposed_Hybrid_HHO_GWO_Lightweight_Model.py
├── Experiment_5_Lightweight_Edge_Suitability.py
└── Experiment_6_Robustness_Across_Attack_Categories.py
Execution Order
python Codes/Experiment_1_CICIoT_Dataset_Audit_Attack_Selection.py
python Codes/Experiment_2_Baseline_DDoS_vs_Benign_Classification.py
python Codes/Experiment_3_Feature_Selection_HHO_GWO_Hybrid.py
python Codes/Experiment_4_Proposed_Hybrid_HHO_GWO_Lightweight_Model.py
python Codes/Experiment_5_Lightweight_Edge_Suitability.py
python Codes/Experiment_6_Robustness_Across_Attack_Categories.py
Main Dependencies
pip install -r requirements.txt
Outputs

Each script automatically creates experiment-specific folders containing:

tables
figures
reports
classification results
selected features
optimization history
robustness summaries
edge-efficiency metrics
Reproducibility

All experiments use a fixed random seed and consistent preprocessing procedures to support reproducibility. The full pipeline evaluates structural dataset properties, baseline detection, feature-selection performance, optimized lightweight modeling, edge suitability, and cross-attack robustness.
