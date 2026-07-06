
# Experiment 5 — Lightweight Edge Suitability

## Objective

This experiment tests whether the reduced feature set produced by the proposed Hybrid HHO-GWO framework improves suitability for edge-IoT deployment.

## Compared Settings

1. Full-feature model using the optimized hyperparameters from Experiment 4.
2. Hybrid HHO-GWO reduced-feature model using the selected features from Experiment 4.

## Evaluation Metrics

- Memory usage
- Model size
- CPU inference time
- Latency per sample
- Training time
- Accuracy
- Macro-F1
- Recall
- False positive rate

## Edge Suitability Metrics

 Accuracy  Macro_F1   Recall  False_Positive_Rate  Training_Time_Seconds  Mean_Inference_Time_Seconds  Std_Inference_Time_Seconds  Latency_Per_Sample_Seconds  Latency_Per_1000_Samples_Seconds  Peak_Memory_MB  Model_Size_MB       Branch          Model_Setting  Feature_Count  Feature_Reduction_Percent
 0.992344  0.992343 0.985208             0.000521               9.966148                     0.030996                    0.001342                1.614358e-06                          0.001614      118.713345       5.599295 Packet-Based          Full_Features            103                   0.000000
 0.990104  0.990103 0.981250             0.001042               0.866408                     0.032850                    0.010661                1.710960e-06                          0.001711       93.978096      10.396232 Packet-Based Hybrid_HHO_GWO_Reduced              8                  92.233010
 0.945885  0.945868 0.927813             0.036042               2.097056                     0.026704                    0.010017                1.390812e-06                          0.001391      103.855950       4.682616   Flow-Based          Full_Features             70                   0.000000
 0.938281  0.938265 0.921875             0.045312               0.383425                     0.015632                    0.000314                8.141714e-07                          0.000814       92.081409       5.734557   Flow-Based Hybrid_HHO_GWO_Reduced              9                  87.142857

## Lightweight Improvement Summary

      Branch  Feature_Count_Reduction  Feature_Reduction_Percent  Model_Size_Reduction_Percent  Memory_Reduction_Percent  Inference_Time_Reduction_Percent  Latency_Reduction_Percent  Macro_F1_Difference  Accuracy_Difference  Recall_Difference  FPR_Difference
Packet-Based                       95                  92.233010                    -85.670380                 20.836115                         -5.983898                  -5.983898            -0.002240            -0.002240          -0.003958        0.000521
  Flow-Based                       61                  87.142857                    -22.464812                 11.337378                         41.460739                  41.460739            -0.007603            -0.007604          -0.005938        0.009271

## Output Files

- experiment5_edge_suitability_metrics.csv
- experiment5_lightweight_improvement_summary.csv
- Experiment5_Lightweight_Edge_Suitability_Results.xlsx
