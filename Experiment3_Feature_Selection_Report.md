
# Experiment 3 — Feature Selection Comparison

## Objective

This experiment evaluates whether feature selection can reduce feature dimensionality while preserving DDoS-vs-Benign detection performance.

## Branches

- Packet-Based
- Flow-Based

## Feature Selection Methods

1. All features baseline
2. Mutual Information
3. Random Forest feature importance
4. Harris Hawks Optimization
5. Grey Wolf Optimization
6. Hybrid HHO-GWO

## Classifiers

- Random Forest
- Extra Trees

## Outputs

- experiment3_feature_selection_results.csv
- experiment3_selected_features.csv
- Experiment3_Feature_Selection_HHO_GWO_Hybrid_Results.xlsx
- Performance figures

## Results Preview

       Model  Accuracy  Precision   Recall       F1  ROC_AUC  Training_Time_Seconds  Inference_Time_Seconds       Branch        Selection_Method  Total_Features  Selected_Features  Reduction_Percent  Feature_Selection_Time_Seconds
RandomForest  0.992135   0.999260 0.985000 0.992079 0.999778               2.885947                0.047019 Packet-Based            All_Features             103                103           0.000000                        0.000000
  ExtraTrees  0.992083   0.998417 0.985729 0.992033 0.999717               0.624732                0.036591 Packet-Based            All_Features             103                103           0.000000                        0.000000
RandomForest  0.989062   0.998408 0.979688 0.988959 0.999358               1.668419                0.031137 Packet-Based      Mutual_Information             103                 30          70.873786                       18.900751
  ExtraTrees  0.989115   0.998725 0.979479 0.989009 0.999290               0.384873                0.031042 Packet-Based      Mutual_Information             103                 30          70.873786                       18.900751
RandomForest  0.992083   0.998628 0.985521 0.992031 0.999727               1.450325                0.031526 Packet-Based RandomForest_Importance             103                 30          70.873786                        3.204178
  ExtraTrees  0.992135   0.997892 0.986354 0.992090 0.999618               0.304864                0.031808 Packet-Based RandomForest_Importance             103                 30          70.873786                        3.204178
RandomForest  0.989427   0.998726 0.980104 0.989328 0.999353               0.884854                0.031762 Packet-Based                     HHO             103                 11          89.320388                       17.260987
  ExtraTrees  0.987187   0.998189 0.976146 0.987044 0.999230               0.217809                0.031435 Packet-Based                     HHO             103                 11          89.320388                       17.260987
RandomForest  0.993490   0.999578 0.987396 0.993450 0.999732               1.276393                0.050984 Packet-Based                     GWO             103                 35          66.019417                       23.322221
  ExtraTrees  0.991927   0.998101 0.985729 0.991877 0.999771               0.342511                0.051564 Packet-Based                     GWO             103                 35          66.019417                       23.322221
RandomForest  0.991302   0.999259 0.983333 0.991232 0.999530               1.715245                0.031337 Packet-Based          Hybrid_HHO_GWO             103                 40          61.165049                       44.730881
  ExtraTrees  0.990469   0.998096 0.982812 0.990395 0.999524               0.343259                0.032000 Packet-Based          Hybrid_HHO_GWO             103                 40          61.165049                       44.730881
RandomForest  0.955000   0.966268 0.942917 0.954450 0.988265               1.823091                0.047815   Flow-Based            All_Features              70                 70           0.000000                        0.000000
  ExtraTrees  0.954688   0.965550 0.943021 0.954153 0.986144               0.661544                0.046877   Flow-Based            All_Features              70                 70           0.000000                        0.000000
RandomForest  0.952500   0.962423 0.941771 0.951985 0.987858               1.290076                0.031122   Flow-Based      Mutual_Information              70                 21          70.000000                       10.356875
  ExtraTrees  0.952760   0.961657 0.943125 0.952301 0.986461               0.339960                0.047358   Flow-Based      Mutual_Information              70                 21          70.000000                       10.356875
RandomForest  0.946667   0.954526 0.938021 0.946202 0.984549               1.363428                0.031238   Flow-Based RandomForest_Importance              70                 21          70.000000                        1.880715
  ExtraTrees  0.947031   0.953024 0.940417 0.946679 0.982368               0.375974                0.047341   Flow-Based RandomForest_Importance              70                 21          70.000000                        1.880715
RandomForest  0.951406   0.962240 0.939688 0.950830 0.987135               1.283408                0.047682   Flow-Based                     HHO              70                 30          57.142857                       19.134153
  ExtraTrees  0.951146   0.961137 0.940312 0.950611 0.985314               0.405438                0.047763   Flow-Based                     HHO              70                 30          57.142857                       19.134153
RandomForest  0.953333   0.962094 0.943854 0.952887 0.987183               1.113322                0.047486   Flow-Based                     GWO              70                 28          60.000000                       18.394575
  ExtraTrees  0.952760   0.962837 0.941875 0.952241 0.984461               0.441571                0.046964   Flow-Based                     GWO              70                 28          60.000000                       18.394575
RandomForest  0.946927   0.952632 0.940625 0.946590 0.984203               1.433953                0.078562   Flow-Based          Hybrid_HHO_GWO              70                 16          77.142857                       42.368707
  ExtraTrees  0.945729   0.951753 0.939063 0.945365 0.982469               0.397927                0.048769   Flow-Based          Hybrid_HHO_GWO              70                 16          77.142857                       42.368707
