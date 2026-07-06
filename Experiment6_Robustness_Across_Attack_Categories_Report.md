
# Experiment 6 — Robustness Across Attack Categories

## Objective

This experiment trains the Hybrid HHO-GWO reduced-feature detector on DDoS-vs-Benign traffic and evaluates whether the learned detector generalizes to unseen attack families.

## Training

- Positive class: DDoS
- Negative class: Benign

## Testing

- Benign
- DoS
- Mirai
- Recon
- Spoofing
- Web-Based
- Brute Force

## Branches

- Packet-Based
- Flow-Based

## Main Outputs

- experiment6_overall_robustness_results.csv
- experiment6_family_level_generalization_results.csv
- Experiment6_Robustness_Across_Attack_Categories_Results.xlsx

## Overall Results

      Branch     Train_Task                                           Test_Task  Shared_Features  Selected_Features_Used  Accuracy  Macro_F1  Attack_Recall_Overall  False_Positive_Rate  True_Positive_Rate
Packet-Based DDoS_vs_Benign Benign_plus_DoS_Mirai_Recon_Spoofing_Web_BruteForce              103                       8  0.888147  0.707841               0.882105             0.000063            0.882105
  Flow-Based DDoS_vs_Benign Benign_plus_DoS_Mirai_Recon_Spoofing_Web_BruteForce               50                       7  0.764863  0.628054               0.747213             0.038479            0.747213

## Family-Level Results

      Branch Test_Family  Samples Expected_Label  Detection_Rate  Miss_or_False_Alarm_Rate
Packet-Based      Benign    48000         Benign        0.999938                  0.000063
Packet-Based Brute Force    12000         Attack        0.311833                  0.688167
Packet-Based         DoS   456000         Attack        0.987309                  0.012691
Packet-Based       Mirai   264000         Attack        0.981182                  0.018818
Packet-Based       Recon    60000         Attack        0.591367                  0.408633
Packet-Based    Spoofing    36000         Attack        0.682361                  0.317639
Packet-Based   Web-Based    60000         Attack        0.171250                  0.828750
  Flow-Based      Benign    48000         Benign        0.961521                  0.038479
  Flow-Based Brute Force     3619         Attack        0.951644                  0.048356
  Flow-Based         DoS   300000         Attack        0.665857                  0.334143
  Flow-Based       Mirai   174509         Attack        0.919586                  0.080414
  Flow-Based       Recon    12000         Attack        0.862917                  0.137083
  Flow-Based    Spoofing    33349         Attack        0.512849                  0.487151
  Flow-Based   Web-Based    11328         Attack        0.748411                  0.251589
