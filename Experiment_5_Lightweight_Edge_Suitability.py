import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import json
import time
import pickle
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path(r"D:\47\472\New-Papers\Atlam_4_2026\Data")

EXP4_RESULTS = Path(
    r"D:\47\472\New-Papers\Atlam_4_2026\CIC IoT\Experiment4_Proposed_Hybrid_HHO_GWO"
)

EXP4_METRICS = EXP4_RESULTS / "tables" / "experiment4_proposed_hybrid_hho_gwo_metrics.csv"
EXP4_FEATURES = EXP4_RESULTS / "tables" / "experiment4_selected_features.csv"

RESULT_ROOT = Path(
    r"D:\47\472\New-Papers\Atlam_4_2026\CIC IoT\Experiment5_Lightweight_Edge_Suitability"
)

TABLE_DIR = RESULT_ROOT / "tables"
FIGURE_DIR = RESULT_ROOT / "figures"
MODEL_DIR = RESULT_ROOT / "models"
REPORT_DIR = RESULT_ROOT / "reports"

for d in [RESULT_ROOT, TABLE_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_ROWS_PER_FILE = 12000
RANDOM_STATE = 42
REPEAT_INFERENCE = 10

np.random.seed(RANDOM_STATE)

# ============================================================
# HELPERS
# ============================================================

def infer_branch(path):
    text = str(path).lower()
    if "packet" in text:
        return "Packet-Based"
    if "flow" in text:
        return "Flow-Based"
    return "Unknown"


def infer_attack_family(path):
    name = path.name.lower()
    if "benign" in name:
        return "Benign"
    if "ddos" in name:
        return "DDoS"
    return None


def safe_read(file_path, nrows=None):
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(file_path, nrows=nrows, encoding=enc, low_memory=False)
        except Exception:
            pass
    return None


def clean_numeric_df(df):
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")

    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    numeric = numeric.dropna(axis=1, how="all")
    nunique = numeric.nunique(dropna=True)
    numeric = numeric.loc[:, nunique > 1]
    return numeric


def get_common_numeric_columns(frames):
    common = None
    for df in frames:
        cols = set(clean_numeric_df(df).columns)
        common = cols if common is None else common & cols
    return sorted(list(common)) if common else []


def safe_clean_feature_matrix(X):
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")

    too_large = []
    for col in X.columns:
        max_abs = X[col].abs().max(skipna=True)
        if pd.notna(max_abs) and max_abs > 1e12:
            too_large.append(col)

    if too_large:
        X = X.drop(columns=too_large)

    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]

    return X


def load_branch_data(branch_name, branch_files):
    loaded = []

    for file_path in branch_files:
        family = infer_attack_family(file_path)
        if family not in ["DDoS", "Benign"]:
            continue

        df = safe_read(file_path, nrows=MAX_ROWS_PER_FILE)
        if df is None or df.empty:
            continue

        df["target"] = 1 if family == "DDoS" else 0
        loaded.append(df)

    common_cols = get_common_numeric_columns(loaded)

    aligned = []
    for df in loaded:
        temp = df[common_cols].copy()
        temp["target"] = df["target"].values
        aligned.append(temp)

    data = pd.concat(aligned, ignore_index=True)

    ddos = data[data["target"] == 1]
    benign = data[data["target"] == 0]
    min_n = min(len(ddos), len(benign))

    ddos = ddos.sample(min_n, random_state=RANDOM_STATE)
    benign = benign.sample(min_n, random_state=RANDOM_STATE)

    data = pd.concat([ddos, benign], ignore_index=True)
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = safe_clean_feature_matrix(data.drop(columns=["target"]))
    y = data["target"].values

    imputer = SimpleImputer(strategy="median")
    X_np = imputer.fit_transform(X)
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    return X_np, y, X.columns.tolist()


def build_model(params):
    model_type = params.get("model_type", "RandomForest")

    common_params = {
        "n_estimators": int(params.get("n_estimators", 100)),
        "max_depth": int(params.get("max_depth", 12)),
        "min_samples_split": int(params.get("min_samples_split", 2)),
        "min_samples_leaf": int(params.get("min_samples_leaf", 1)),
        "max_features": params.get("max_features", "sqrt"),
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

    if model_type == "ExtraTrees":
        return ExtraTreesClassifier(**common_params)

    return RandomForestClassifier(**common_params)


def model_size_mb(model):
    b = pickle.dumps(model)
    return len(b) / (1024 * 1024)


def benchmark_model(model, X_train, y_train, X_test, y_test):
    tracemalloc.start()
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    inference_times = []
    preds = None

    for _ in range(REPEAT_INFERENCE):
        start = time.perf_counter()
        preds = model.predict(X_test)
        inference_times.append(time.perf_counter() - start)

    inference_time = float(np.mean(inference_times))
    inference_std = float(np.std(inference_times))
    per_sample_latency = inference_time / len(X_test)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Macro_F1": f1_score(y_test, preds, average="macro", zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "False_Positive_Rate": fpr,
        "Training_Time_Seconds": train_time,
        "Mean_Inference_Time_Seconds": inference_time,
        "Std_Inference_Time_Seconds": inference_std,
        "Latency_Per_Sample_Seconds": per_sample_latency,
        "Latency_Per_1000_Samples_Seconds": per_sample_latency * 1000,
        "Peak_Memory_MB": peak / (1024 * 1024),
        "Model_Size_MB": model_size_mb(model)
    }

# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("Experiment 5 — Lightweight Edge Suitability")
print("=" * 80)

exp4_metrics = pd.read_csv(EXP4_METRICS)
exp4_features = pd.read_csv(EXP4_FEATURES)

all_csv = sorted(DATA_ROOT.rglob("*.csv"))
target_files = [f for f in all_csv if infer_attack_family(f) in ["DDoS", "Benign"]]

packet_files = [f for f in target_files if infer_branch(f) == "Packet-Based"]
flow_files = [f for f in target_files if infer_branch(f) == "Flow-Based"]

branches = {
    "Packet-Based": packet_files,
    "Flow-Based": flow_files
}

results = []

for branch_name, files in branches.items():
    print("\n" + "=" * 80)
    print(f"Branch: {branch_name}")
    print("=" * 80)

    X, y, feature_names = load_branch_data(branch_name, files)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    branch_metric = exp4_metrics[exp4_metrics["Branch"] == branch_name].iloc[0]
    params = json.loads(branch_metric["Best_Hyperparameters_JSON"])

    selected_names = exp4_features[exp4_features["Branch"] == branch_name]["Feature"].tolist()
    selected_idx = [feature_names.index(f) for f in selected_names if f in feature_names]

    print(f"All features: {len(feature_names)}")
    print(f"Selected features: {len(selected_idx)}")

    # Full-feature model
    full_model = build_model(params)
    full_metrics = benchmark_model(full_model, X_train, y_train, X_test, y_test)
    full_metrics.update({
        "Branch": branch_name,
        "Model_Setting": "Full_Features",
        "Feature_Count": len(feature_names),
        "Feature_Reduction_Percent": 0.0
    })
    results.append(full_metrics)

    # Reduced-feature model
    reduced_model = build_model(params)
    reduced_metrics = benchmark_model(
        reduced_model,
        X_train[:, selected_idx],
        y_train,
        X_test[:, selected_idx],
        y_test
    )
    reduced_metrics.update({
        "Branch": branch_name,
        "Model_Setting": "Hybrid_HHO_GWO_Reduced",
        "Feature_Count": len(selected_idx),
        "Feature_Reduction_Percent": 100 * (1 - len(selected_idx) / len(feature_names))
    })
    results.append(reduced_metrics)

    # Save models
    with open(MODEL_DIR / f"experiment5_{branch_name.replace('-', '_')}_full_model.pkl", "wb") as f:
        pickle.dump(full_model, f)

    with open(MODEL_DIR / f"experiment5_{branch_name.replace('-', '_')}_reduced_model.pkl", "wb") as f:
        pickle.dump(reduced_model, f)

results_df = pd.DataFrame(results)

# ============================================================
# IMPROVEMENT SUMMARY
# ============================================================

improvement_records = []

for branch in results_df["Branch"].unique():
    full = results_df[
        (results_df["Branch"] == branch) &
        (results_df["Model_Setting"] == "Full_Features")
    ].iloc[0]

    reduced = results_df[
        (results_df["Branch"] == branch) &
        (results_df["Model_Setting"] == "Hybrid_HHO_GWO_Reduced")
    ].iloc[0]

    improvement_records.append({
        "Branch": branch,
        "Feature_Count_Reduction": full["Feature_Count"] - reduced["Feature_Count"],
        "Feature_Reduction_Percent": reduced["Feature_Reduction_Percent"],
        "Model_Size_Reduction_Percent": 100 * (1 - reduced["Model_Size_MB"] / full["Model_Size_MB"]),
        "Memory_Reduction_Percent": 100 * (1 - reduced["Peak_Memory_MB"] / full["Peak_Memory_MB"]),
        "Inference_Time_Reduction_Percent": 100 * (1 - reduced["Mean_Inference_Time_Seconds"] / full["Mean_Inference_Time_Seconds"]),
        "Latency_Reduction_Percent": 100 * (1 - reduced["Latency_Per_Sample_Seconds"] / full["Latency_Per_Sample_Seconds"]),
        "Macro_F1_Difference": reduced["Macro_F1"] - full["Macro_F1"],
        "Accuracy_Difference": reduced["Accuracy"] - full["Accuracy"],
        "Recall_Difference": reduced["Recall"] - full["Recall"],
        "FPR_Difference": reduced["False_Positive_Rate"] - full["False_Positive_Rate"]
    })

improvement_df = pd.DataFrame(improvement_records)

# ============================================================
# SAVE TABLES
# ============================================================

metrics_csv = TABLE_DIR / "experiment5_edge_suitability_metrics.csv"
improvement_csv = TABLE_DIR / "experiment5_lightweight_improvement_summary.csv"

results_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
improvement_df.to_csv(improvement_csv, index=False, encoding="utf-8-sig")

excel_path = RESULT_ROOT / "Experiment5_Lightweight_Edge_Suitability_Results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    results_df.to_excel(writer, index=False, sheet_name="Edge_Metrics")
    improvement_df.to_excel(writer, index=False, sheet_name="Improvement_Summary")

# ============================================================
# FIGURES
# ============================================================

plot_metrics = [
    "Model_Size_MB",
    "Peak_Memory_MB",
    "Mean_Inference_Time_Seconds",
    "Latency_Per_1000_Samples_Seconds",
    "Training_Time_Seconds",
    "Macro_F1"
]

for metric in plot_metrics:
    plt.figure(figsize=(9, 5))
    labels = results_df["Branch"] + " — " + results_df["Model_Setting"]
    plt.bar(labels, results_df[metric])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(metric)
    plt.title(f"Experiment 5 — {metric}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"experiment5_{metric}.png", dpi=300)
    plt.close()

# ============================================================
# REPORT
# ============================================================

report = f"""
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

{results_df.to_string(index=False)}

## Lightweight Improvement Summary

{improvement_df.to_string(index=False)}

## Output Files

- experiment5_edge_suitability_metrics.csv
- experiment5_lightweight_improvement_summary.csv
- Experiment5_Lightweight_Edge_Suitability_Results.xlsx
"""

report_path = REPORT_DIR / "Experiment5_Lightweight_Edge_Suitability_Report.md"
report_path.write_text(report, encoding="utf-8")

print("\n" + "=" * 80)
print("EXPERIMENT 5 COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Results folder: {RESULT_ROOT}")
print(f"Metrics CSV: {metrics_csv}")
print(f"Excel: {excel_path}")
print(f"Report: {report_path}")
print("=" * 80)