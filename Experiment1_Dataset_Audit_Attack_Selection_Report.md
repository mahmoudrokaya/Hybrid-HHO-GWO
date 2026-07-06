
# Experiment 1 — CICIoT Dataset Audit and Attack Selection

## Objective

This experiment performs the first dataset-level audit for the CIC IoT research direction.
The goal is to determine the most suitable experimental scenario for the optimization-centered study.

The adopted experimental direction is:

**Hybrid HHO-GWO Feature Selection for Lightweight DDoS and Multi-Attack Detection on CICIoT Packet/Flow Features**

## Dataset Root

`D:\47\472\New-Papers\Atlam_4_2026\Data`

## Output Folder

`D:\47\472\New-Papers\Atlam_4_2026\CIC IoT\Experiment1_Dataset_Audit_Attack_Selection`

## Main Dataset Findings

- Total CSV files discovered: 314
- Successfully read files: 314
- Packet-based files: 181
- Flow-based files: 133

## Feature Structure

- Packet features: 135
- Flow features: 288
- Shared packet-flow features: 0
- Packet-flow Jaccard overlap: 0.0000

## Attack Families

attack_family
DDoS           164
DoS             68
Mirai           51
Benign           8
Web-Based        8
Spoofing         6
Recon            6
Brute Force      2
Unknown          1

## Recommended Experiment Sequence

1. Binary DDoS vs Benign detection.
2. DDoS subtype classification.
3. Multi-class classification across major attack families.
4. HHO, GWO, and Hybrid HHO-GWO feature selection comparison.
5. Lightweight edge suitability evaluation using selected feature subsets.

## Generated Candidate Files

- `experiment1_candidate_files_binary_ddos_vs_benign.csv`
- `experiment1_candidate_files_ddos_subtype_classification.csv`
- `experiment1_candidate_files_multiclass_attack_families.csv`

## Interpretation

The dataset is suitable for the optimization-centered CIC IoT study.
However, packet-based and flow-based files should not be mixed directly because their feature spaces are structurally different.
The recommended strategy is to run the experiments separately for packet-based and flow-based representations, then compare whether the proposed hybrid HHO-GWO optimizer behaves consistently across both representations.
