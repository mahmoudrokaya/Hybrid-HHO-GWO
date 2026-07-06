import os
import re
import json
import math
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# EXPERIMENT 1 — DATASET AUDIT AND ATTACK SELECTION
# ============================================================

DATA_ROOT = Path(r"D:\47\472\New-Papers\Atlam_4_2026\Data")
RESULTS_ROOT = Path(r"D:\47\472\New-Papers\Atlam_4_2026\CIC IoT")
EXPERIMENT_DIR = RESULTS_ROOT / "Experiment1_Dataset_Audit_Attack_Selection"

TABLE_DIR = EXPERIMENT_DIR / "tables"
FIGURE_DIR = EXPERIMENT_DIR / "figures"
REPORT_DIR = EXPERIMENT_DIR / "reports"

for d in [EXPERIMENT_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_ROWS = 30000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_name(x):
    x = str(x)
    x = os.path.basename(x)
    x = re.sub(r"\.pcap_Flow\.csv$", "", x, flags=re.I)
    x = re.sub(r"\.csv$", "", x, flags=re.I)
    x = x.replace("_", "-").replace(" ", "-")
    x = re.sub(r"-+", "-", x)
    return x.strip("-")


def infer_branch(path):
    text = str(path).lower()
    if "packet" in text:
        return "Packet-Based"
    if "flow" in text:
        return "Flow-Based"
    return "Unknown"


def infer_attack_family(path_or_name):
    text = normalize_name(path_or_name).lower()

    if "benign" in text:
        return "Benign"
    if "ddos" in text:
        return "DDoS"
    if re.search(r"(^|-)dos($|-)", text):
        return "DoS"
    if "mirai" in text:
        return "Mirai"
    if "recon" in text or "scan" in text or "hostdiscovery" in text:
        return "Recon"
    if "spoof" in text or "mitm" in text:
        return "Spoofing"
    if "brute" in text or "dictionary" in text or "password" in text:
        return "Brute Force"
    if any(k in text for k in ["web", "xss", "sql", "injection", "upload", "backdoor"]):
        return "Web-Based"

    return "Unknown"


def safe_read_csv_sample(file_path, nrows=SAMPLE_ROWS):
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(file_path, nrows=nrows, low_memory=False, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Could not read file: {file_path}")


def detect_label_column(columns):
    candidates = [
        "label", "Label", "LABEL",
        "class", "Class", "CLASS",
        "attack", "Attack",
        "category", "Category",
        "type", "Type"
    ]

    for c in candidates:
        if c in columns:
            return c

    for c in columns:
        lc = str(c).lower()
        if any(k in lc for k in ["label", "class", "attack", "category"]):
            return c

    return None


def human_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size)
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.2f} {units[i]}"


def entropy(counts):
    counts = np.array(list(counts), dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def normalized_entropy(counts):
    counts = list(counts)
    if len(counts) <= 1:
        return 0.0
    return entropy(counts) / math.log2(len(counts))


def plot_bar(df, x, y, title, output_path, top_n=40):
    if df.empty:
        return

    data = df.sort_values(y, ascending=False).head(top_n)

    plt.figure(figsize=(13, 6))
    plt.bar(data[x].astype(str), data[y])
    plt.xticks(rotation=75, ha="right")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# DISCOVER FILES
# ============================================================

csv_files = sorted(DATA_ROOT.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {DATA_ROOT}")

print(f"Detected CSV files: {len(csv_files)}")

# ============================================================
# FILE-LEVEL AUDIT
# ============================================================

audit_records = []
feature_records = []
label_records = []

for i, file_path in enumerate(csv_files, 1):
    print(f"[{i}/{len(csv_files)}] {file_path.name}")

    branch = infer_branch(file_path)
    subtype = normalize_name(file_path.name)
    family = infer_attack_family(file_path)

    record = {
        "file_name": file_path.name,
        "relative_path": str(file_path.relative_to(DATA_ROOT)),
        "full_path": str(file_path),
        "branch": branch,
        "attack_family": family,
        "attack_subtype": subtype,
        "size_bytes": file_path.stat().st_size,
        "size_readable": human_size(file_path.stat().st_size),
        "sample_rows_read": 0,
        "columns": 0,
        "numeric_columns": 0,
        "categorical_columns": 0,
        "label_column_detected": "",
        "missing_values_sample": 0,
        "missing_ratio_sample": 0,
        "duplicate_rows_sample": 0,
        "read_status": "failed",
        "error": ""
    }

    try:
        df = safe_read_csv_sample(file_path)
        record["read_status"] = "success"
        record["sample_rows_read"] = len(df)
        record["columns"] = df.shape[1]
        record["numeric_columns"] = len(df.select_dtypes(include=[np.number]).columns)
        record["categorical_columns"] = df.shape[1] - record["numeric_columns"]
        record["missing_values_sample"] = int(df.isna().sum().sum())
        record["missing_ratio_sample"] = float(df.isna().sum().sum() / max(df.size, 1))
        record["duplicate_rows_sample"] = int(df.duplicated().sum())

        label_col = detect_label_column(df.columns)
        record["label_column_detected"] = label_col if label_col else ""

        for col in df.columns:
            feature_records.append({
                "feature": col,
                "file_name": file_path.name,
                "branch": branch,
                "attack_family": family,
                "attack_subtype": subtype,
                "dtype": str(df[col].dtype),
                "missing_count_sample": int(df[col].isna().sum()),
                "missing_ratio_sample": float(df[col].isna().sum() / max(len(df), 1)),
                "unique_values_sample": int(df[col].nunique(dropna=True))
            })

        if label_col:
            counts = df[label_col].astype(str).value_counts()
            for label, count in counts.items():
                label_records.append({
                    "file_name": file_path.name,
                    "branch": branch,
                    "attack_family_from_path": family,
                    "attack_subtype_from_path": subtype,
                    "label_column": label_col,
                    "label_value": label,
                    "sample_count": int(count)
                })
        else:
            label_records.append({
                "file_name": file_path.name,
                "branch": branch,
                "attack_family_from_path": family,
                "attack_subtype_from_path": subtype,
                "label_column": "No explicit label column",
                "label_value": subtype,
                "sample_count": len(df)
            })

    except Exception as e:
        record["error"] = str(e)

    audit_records.append(record)

audit_df = pd.DataFrame(audit_records)
features_df = pd.DataFrame(feature_records)
labels_df = pd.DataFrame(label_records)

audit_df.to_csv(TABLE_DIR / "experiment1_file_schema_audit.csv", index=False, encoding="utf-8-sig")
features_df.to_csv(TABLE_DIR / "experiment1_feature_inventory_long.csv", index=False, encoding="utf-8-sig")
labels_df.to_csv(TABLE_DIR / "experiment1_label_values_from_samples.csv", index=False, encoding="utf-8-sig")

# ============================================================
# BRANCH, ATTACK, AND SIZE SUMMARIES
# ============================================================

branch_summary = audit_df.groupby("branch").agg(
    files=("file_name", "count"),
    total_size_bytes=("size_bytes", "sum"),
    avg_columns=("columns", "mean"),
    avg_missing_ratio=("missing_ratio_sample", "mean"),
    avg_duplicate_rows=("duplicate_rows_sample", "mean")
).reset_index()

branch_summary["total_size_readable"] = branch_summary["total_size_bytes"].apply(human_size)

attack_summary = audit_df.groupby(["attack_family", "branch"]).agg(
    files=("file_name", "count"),
    total_size_bytes=("size_bytes", "sum"),
    avg_columns=("columns", "mean")
).reset_index()

attack_summary["total_size_readable"] = attack_summary["total_size_bytes"].apply(human_size)

subtype_summary = audit_df.groupby(["attack_family", "attack_subtype", "branch"]).agg(
    files=("file_name", "count"),
    total_size_bytes=("size_bytes", "sum")
).reset_index()

subtype_summary["total_size_readable"] = subtype_summary["total_size_bytes"].apply(human_size)

branch_summary.to_csv(TABLE_DIR / "experiment1_branch_summary.csv", index=False, encoding="utf-8-sig")
attack_summary.to_csv(TABLE_DIR / "experiment1_attack_family_summary.csv", index=False, encoding="utf-8-sig")
subtype_summary.to_csv(TABLE_DIR / "experiment1_attack_subtype_summary.csv", index=False, encoding="utf-8-sig")

plot_bar(
    branch_summary,
    "branch",
    "files",
    "Experiment 1: Packet vs Flow File Counts",
    FIGURE_DIR / "experiment1_packet_vs_flow_file_counts.png"
)

plot_bar(
    attack_summary.groupby("attack_family", as_index=False)["files"].sum(),
    "attack_family",
    "files",
    "Experiment 1: Attack Family File Distribution",
    FIGURE_DIR / "experiment1_attack_family_file_distribution.png"
)

# ============================================================
# FEATURE OVERLAP ANALYSIS
# ============================================================

feature_presence = features_df.groupby(["feature", "branch"]).size().reset_index(name="count")

packet_features = set(feature_presence[feature_presence["branch"] == "Packet-Based"]["feature"])
flow_features = set(feature_presence[feature_presence["branch"] == "Flow-Based"]["feature"])

shared_features = packet_features & flow_features
packet_unique = packet_features - flow_features
flow_unique = flow_features - packet_features

feature_overlap_summary = pd.DataFrame([
    {"item": "packet_features", "count": len(packet_features)},
    {"item": "flow_features", "count": len(flow_features)},
    {"item": "shared_features", "count": len(shared_features)},
    {"item": "packet_unique_features", "count": len(packet_unique)},
    {"item": "flow_unique_features", "count": len(flow_unique)},
    {
        "item": "packet_flow_jaccard_overlap",
        "count": len(shared_features) / max(len(packet_features | flow_features), 1)
    }
])

feature_sets_df = pd.DataFrame({
    "shared_features": pd.Series(sorted(shared_features)),
    "packet_unique_features": pd.Series(sorted(packet_unique)),
    "flow_unique_features": pd.Series(sorted(flow_unique))
})

feature_overlap_summary.to_csv(TABLE_DIR / "experiment1_packet_flow_feature_overlap_summary.csv", index=False, encoding="utf-8-sig")
feature_sets_df.to_csv(TABLE_DIR / "experiment1_packet_flow_feature_sets.csv", index=False, encoding="utf-8-sig")

# ============================================================
# MISSINGNESS AND DUPLICATES
# ============================================================

missing_by_file = audit_df[[
    "file_name", "branch", "attack_family", "attack_subtype",
    "missing_values_sample", "missing_ratio_sample", "duplicate_rows_sample"
]].copy()

missing_by_feature = features_df.groupby(["feature", "branch"]).agg(
    files=("file_name", "nunique"),
    mean_missing_ratio=("missing_ratio_sample", "mean"),
    max_missing_ratio=("missing_ratio_sample", "max")
).reset_index().sort_values("mean_missing_ratio", ascending=False)

missing_by_file.to_csv(TABLE_DIR / "experiment1_missingness_by_file.csv", index=False, encoding="utf-8-sig")
missing_by_feature.to_csv(TABLE_DIR / "experiment1_missingness_by_feature.csv", index=False, encoding="utf-8-sig")

plot_bar(
    missing_by_feature.head(40),
    "feature",
    "mean_missing_ratio",
    "Experiment 1: Top Missing Features",
    FIGURE_DIR / "experiment1_top_missing_features.png"
)

# ============================================================
# DDoS AND BENIGN SELECTION FOR FUTURE EXPERIMENTS
# ============================================================

ddos_files = audit_df[audit_df["attack_family"] == "DDoS"].copy()
benign_files = audit_df[audit_df["attack_family"] == "Benign"].copy()

ddos_binary_candidates = pd.concat([ddos_files, benign_files], ignore_index=True)

ddos_subtype_candidates = ddos_files.copy()

multiclass_7_category_candidates = audit_df[
    audit_df["attack_family"].isin([
        "Benign", "DDoS", "DoS", "Mirai", "Recon",
        "Spoofing", "Web-Based", "Brute Force"
    ])
].copy()

ddos_binary_candidates.to_csv(TABLE_DIR / "experiment1_candidate_files_binary_ddos_vs_benign.csv", index=False, encoding="utf-8-sig")
ddos_subtype_candidates.to_csv(TABLE_DIR / "experiment1_candidate_files_ddos_subtype_classification.csv", index=False, encoding="utf-8-sig")
multiclass_7_category_candidates.to_csv(TABLE_DIR / "experiment1_candidate_files_multiclass_attack_families.csv", index=False, encoding="utf-8-sig")

# ============================================================
# IMBALANCE ANALYSIS
# ============================================================

family_counts = attack_summary.groupby("attack_family")["files"].sum().to_dict()
subtype_counts = subtype_summary.groupby("attack_subtype")["files"].sum().to_dict()

imbalance_df = pd.DataFrame([
    {
        "level": "attack_family",
        "classes": len(family_counts),
        "max_count": max(family_counts.values()) if family_counts else 0,
        "min_count": min(family_counts.values()) if family_counts else 0,
        "imbalance_ratio": max(family_counts.values()) / max(min(family_counts.values()), 1) if family_counts else 0,
        "normalized_entropy": normalized_entropy(family_counts.values())
    },
    {
        "level": "attack_subtype",
        "classes": len(subtype_counts),
        "max_count": max(subtype_counts.values()) if subtype_counts else 0,
        "min_count": min(subtype_counts.values()) if subtype_counts else 0,
        "imbalance_ratio": max(subtype_counts.values()) / max(min(subtype_counts.values()), 1) if subtype_counts else 0,
        "normalized_entropy": normalized_entropy(subtype_counts.values())
    }
])

imbalance_df.to_csv(TABLE_DIR / "experiment1_imbalance_summary.csv", index=False, encoding="utf-8-sig")

# ============================================================
# CONSOLIDATED EXCEL WORKBOOK
# ============================================================

excel_path = EXPERIMENT_DIR / "Experiment1_Dataset_Audit_Attack_Selection_All_Tables.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    audit_df.to_excel(writer, index=False, sheet_name="File_Audit")
    branch_summary.to_excel(writer, index=False, sheet_name="Branch_Summary")
    attack_summary.to_excel(writer, index=False, sheet_name="Attack_Family")
    subtype_summary.to_excel(writer, index=False, sheet_name="Attack_Subtype")
    feature_overlap_summary.to_excel(writer, index=False, sheet_name="Feature_Overlap")
    feature_sets_df.to_excel(writer, index=False, sheet_name="Feature_Sets")
    missing_by_file.to_excel(writer, index=False, sheet_name="Missing_By_File")
    missing_by_feature.to_excel(writer, index=False, sheet_name="Missing_By_Feature")
    imbalance_df.to_excel(writer, index=False, sheet_name="Imbalance")
    ddos_binary_candidates.to_excel(writer, index=False, sheet_name="DDoS_vs_Benign")
    ddos_subtype_candidates.to_excel(writer, index=False, sheet_name="DDoS_Subtypes")
    multiclass_7_category_candidates.to_excel(writer, index=False, sheet_name="Multiclass_Families")

# ============================================================
# SCIENTIFIC REPORT
# ============================================================

report = f"""
# Experiment 1 — CICIoT Dataset Audit and Attack Selection

## Objective

This experiment performs the first dataset-level audit for the CIC IoT research direction.
The goal is to determine the most suitable experimental scenario for the optimization-centered study.

The adopted experimental direction is:

**Hybrid HHO-GWO Feature Selection for Lightweight DDoS and Multi-Attack Detection on CICIoT Packet/Flow Features**

## Dataset Root

`{DATA_ROOT}`

## Output Folder

`{EXPERIMENT_DIR}`

## Main Dataset Findings

- Total CSV files discovered: {len(audit_df)}
- Successfully read files: {(audit_df["read_status"] == "success").sum()}
- Packet-based files: {(audit_df["branch"] == "Packet-Based").sum()}
- Flow-based files: {(audit_df["branch"] == "Flow-Based").sum()}

## Feature Structure

- Packet features: {len(packet_features)}
- Flow features: {len(flow_features)}
- Shared packet-flow features: {len(shared_features)}
- Packet-flow Jaccard overlap: {len(shared_features) / max(len(packet_features | flow_features), 1):.4f}

## Attack Families

{attack_summary.groupby("attack_family")["files"].sum().sort_values(ascending=False).to_string()}

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
"""

report_path = REPORT_DIR / "Experiment1_Dataset_Audit_Attack_Selection_Report.md"
report_path.write_text(report, encoding="utf-8")

# ============================================================
# FINAL PRINT
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1 COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Dataset root: {DATA_ROOT}")
print(f"Experiment folder: {EXPERIMENT_DIR}")
print(f"Main workbook: {excel_path}")
print(f"Report: {report_path}")
print("=" * 80)