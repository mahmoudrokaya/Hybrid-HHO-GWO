
# Experiment 2 — Branch-Separated Baseline DDoS vs Benign Classification

## Objective

This experiment establishes baseline DDoS-vs-Benign classification performance before applying feature selection.

## Reason for Branch Separation

Packet-based and flow-based files were processed separately because Experiment 1 showed that their feature spaces have no shared features.

## Files

- Packet DDoS/Benign files: 106
- Flow DDoS/Benign files: 66

## Models

- Random Forest
- Extra Trees
- Logistic Regression

## Main Results

      Branch              Model  Samples  Original_Common_Features  Usable_Features_After_Cleaning  Accuracy  Precision   Recall       F1  ROC_AUC
Packet-Based       RandomForest   120000                       104                             104  0.993667   0.999494 0.987833 0.993630 0.999775
Packet-Based         ExtraTrees   120000                       104                             104  0.993458   0.999073 0.987833 0.993421 0.999727
Packet-Based LogisticRegression   120000                       104                             104  0.978167   0.991688 0.964417 0.977862 0.995765
  Flow-Based       RandomForest   120000                        70                              70  0.953667   0.962846 0.943750 0.953203 0.988401
  Flow-Based         ExtraTrees   120000                        70                              70  0.953417   0.961102 0.945083 0.953025 0.986371
  Flow-Based LogisticRegression   120000                        70                              70  0.838833   0.866571 0.801000 0.832496 0.920647

## Output Files

- experiment2_branch_separated_baseline_results.csv
- experiment2_branch_common_features.csv
- experiment2_usable_feature_list.csv
- Experiment2_Branch_Separated_Baseline_Results.xlsx

## Next Step

Experiment 3 should apply feature-selection methods to each branch separately:

1. Baseline all features.
2. Mutual Information.
3. Random Forest feature importance.
4. HHO.
5. GWO.
6. Hybrid HHO-GWO.
