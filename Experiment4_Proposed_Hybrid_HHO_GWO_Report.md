
# Experiment 4 — Proposed Hybrid HHO-GWO Lightweight Model

## Objective

This experiment evaluates the proposed Hybrid HHO-GWO framework for:

1. Feature selection.
2. Hyperparameter optimization.
3. Lightweight model construction.

## Dataset Task

Binary DDoS-vs-Benign classification using CIC IoT packet-based and flow-based branches.

## Evaluation Metrics

- Accuracy
- Macro-F1
- Recall
- False Positive Rate
- ROC-AUC
- Selected feature count
- Feature reduction percentage
- Training time
- Inference time
- Optimization time

## Results

      Branch   Model_Type  Selected_Features  Accuracy  Macro_F1   Recall  False_Positive_Rate  ROC_AUC  Training_Time_Seconds  Inference_Time_Seconds  Inference_Time_Per_Sample  n_estimators  max_depth  min_samples_split  min_samples_leaf max_features  Total_Features  Reduction_Percent  Optimization_Time_Seconds                                                                                                                Best_Hyperparameters_JSON
Packet-Based RandomForest                  8  0.990156  0.990156 0.981771             0.001458 0.999391               0.962493                0.031347                   0.000002           119         20                  4                 2          0.5             103          92.233010                  93.935358 {"model_type": "RandomForest", "n_estimators": 119, "max_depth": 20, "min_samples_split": 4, "min_samples_leaf": 2, "max_features": 0.5}
  Flow-Based RandomForest                  9  0.938438  0.938421 0.921875             0.045000 0.982490               0.393325                0.031144                   0.000002            49         12                  7                 1          0.5              70          87.142857                  55.000080  {"model_type": "RandomForest", "n_estimators": 49, "max_depth": 12, "min_samples_split": 7, "min_samples_leaf": 1, "max_features": 0.5}

## Output Files

- experiment4_proposed_hybrid_hho_gwo_metrics.csv
- experiment4_selected_features.csv
- experiment4_optimization_history.csv
- Experiment4_Proposed_Hybrid_HHO_GWO_Results.xlsx
