# Experiment_2_Baseline_DDoS_vs_Benign_Classification.py

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"D:\47\472\New-Papers\Atlam_4_2026\Data")

RESULT_ROOT = Path(
    r"D:\47\472\New-Papers\Atlam_4_2026\CIC IoT\Experiment2_Baseline_DDoS_vs_Benign"
)

TABLE_DIR = RESULT_ROOT / "tables"
FIGURE_DIR = RESULT_ROOT / "figures"
REPORT_DIR = RESULT_ROOT / "reports"

for folder in [RESULT_ROOT, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

MAX_ROWS_PER_FILE = 15000
RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
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
            return pd.read_csv(
                file_path,
                nrows=nrows,
                encoding=enc,
                low_memory=False
            )
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
        numeric_cols = set(clean_numeric_df(df).columns)

        if common is None:
            common = numeric_cols
        else:
            common &= numeric_cols

    return sorted(list(common)) if common else []


def safe_clean_feature_matrix(X, branch_name):
    X = X.copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")

    too_large_cols = []

    for col in X.columns:
        max_abs = X[col].abs().max(skipna=True)

        if pd.notna(max_abs) and max_abs > 1e12:
            too_large_cols.append(col)

    if too_large_cols:
        print(f"Dropping extremely large columns in {branch_name}: {too_large_cols}")
        X = X.drop(columns=too_large_cols)

    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()

    if constant_cols:
        print(f"Dropping constant columns in {branch_name}: {constant_cols}")
        X = X.drop(columns=constant_cols)

    return X


def plot_confusion_matrix(cm, labels, title, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metric_comparison(results_df, metric, output_path):
    plt.figure(figsize=(10, 5))

    labels = results_df["Branch"] + " — " + results_df["Model"]
    plt.bar(labels, results_df[metric])

    plt.ylim(0, 1.05)
    plt.xticks(rotation=60, ha="right")
    plt.title(f"Experiment 2 — {metric}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_branch_experiment(branch_name, branch_files):
    print("\n" + "=" * 80)
    print(f"Running branch: {branch_name}")
    print("=" * 80)

    loaded = []

    for file_path in branch_files:
        family = infer_attack_family(file_path)

        if family not in ["DDoS", "Benign"]:
            continue

        df = safe_read(file_path, nrows=MAX_ROWS_PER_FILE)

        if df is None or df.empty:
            continue

        df["target"] = 1 if family == "DDoS" else 0
        df["source_file"] = file_path.name

        loaded.append(df)

    print(f"Loaded files for {branch_name}: {len(loaded)}")

    if len(loaded) < 2:
        print(f"Skipping {branch_name}: not enough files.")
        return None, None, None

    common_cols = get_common_numeric_columns(loaded)

    print(f"Common numeric features for {branch_name}: {len(common_cols)}")

    if len(common_cols) == 0:
        print(f"Skipping {branch_name}: no common numeric features.")
        return None, None, None

    aligned = []

    for df in loaded:
        temp = df[common_cols].copy()
        temp["target"] = df["target"].values
        aligned.append(temp)

    data = pd.concat(aligned, ignore_index=True)

    ddos = data[data["target"] == 1]
    benign = data[data["target"] == 0]

    min_n = min(len(ddos), len(benign))

    if min_n == 0:
        print(f"Skipping {branch_name}: one class is missing.")
        return None, None, None

    ddos = ddos.sample(min_n, random_state=RANDOM_STATE)
    benign = benign.sample(min_n, random_state=RANDOM_STATE)

    data = pd.concat([ddos, benign], ignore_index=True)
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Balanced dataset shape for {branch_name}: {data.shape}")
    print(data["target"].value_counts())

    X_raw = data.drop(columns=["target"])
    y = data["target"].values

    X_clean = safe_clean_feature_matrix(X_raw, branch_name)

    if X_clean.shape[1] == 0:
        print(f"Skipping {branch_name}: no usable features after cleaning.")
        return None, None, None

    usable_features = X_clean.columns.tolist()

    print(f"Usable features after cleaning for {branch_name}: {len(usable_features)}")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_clean)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    branch_results = []

    for model_name, model in models.items():
        print(f"Training {branch_name} — {model_name}")

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        else:
            auc = np.nan

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        cm = confusion_matrix(y_test, preds)

        safe_branch = branch_name.replace(" ", "_").replace("-", "_")

        plot_confusion_matrix(
            cm,
            labels=["Benign", "DDoS"],
            title=f"{branch_name} — {model_name}",
            output_path=FIGURE_DIR / f"experiment2_{safe_branch}_{model_name}_confusion_matrix.png"
        )

        report = classification_report(
            y_test,
            preds,
            target_names=["Benign", "DDoS"],
            output_dict=True,
            zero_division=0
        )

        pd.DataFrame(report).transpose().to_csv(
            TABLE_DIR / f"experiment2_{safe_branch}_{model_name}_classification_report.csv",
            encoding="utf-8-sig"
        )

        branch_results.append({
            "Branch": branch_name,
            "Model": model_name,
            "Samples": len(data),
            "Original_Common_Features": len(common_cols),
            "Usable_Features_After_Cleaning": len(usable_features),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": auc
        })

    feature_info = pd.DataFrame({
        "Branch": branch_name,
        "Feature": usable_features
    })

    return pd.DataFrame(branch_results), usable_features, feature_info


# ============================================================
# MAIN EXECUTION
# ============================================================

print("=" * 80)
print("Experiment 2 — Branch-Separated Baseline DDoS vs Benign Classification")
print("=" * 80)

all_csv = sorted(DATA_ROOT.rglob("*.csv"))

target_files = [
    f for f in all_csv
    if infer_attack_family(f) in ["DDoS", "Benign"]
]

print(f"Total DDoS/Benign files found: {len(target_files)}")

packet_files = [
    f for f in target_files
    if infer_branch(f) == "Packet-Based"
]

flow_files = [
    f for f in target_files
    if infer_branch(f) == "Flow-Based"
]

print(f"Packet DDoS/Benign files: {len(packet_files)}")
print(f"Flow DDoS/Benign files: {len(flow_files)}")

all_results = []
feature_summary = []
feature_info_tables = []

for branch_name, files in [
    ("Packet-Based", packet_files),
    ("Flow-Based", flow_files)
]:
    results, features, feature_info = run_branch_experiment(branch_name, files)

    if results is not None:
        all_results.append(results)

        feature_summary.append({
            "Branch": branch_name,
            "Usable_Features": len(features),
            "Feature_List": "; ".join(features)
        })

        feature_info_tables.append(feature_info)

if not all_results:
    raise RuntimeError("No valid branch experiment was completed.")

results_df = pd.concat(all_results, ignore_index=True)
feature_summary_df = pd.DataFrame(feature_summary)
feature_info_df = pd.concat(feature_info_tables, ignore_index=True)

results_csv = TABLE_DIR / "experiment2_branch_separated_baseline_results.csv"
feature_summary_csv = TABLE_DIR / "experiment2_branch_common_features.csv"
feature_info_csv = TABLE_DIR / "experiment2_usable_feature_list.csv"

results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
feature_summary_df.to_csv(feature_summary_csv, index=False, encoding="utf-8-sig")
feature_info_df.to_csv(feature_info_csv, index=False, encoding="utf-8-sig")

for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
    plot_metric_comparison(
        results_df,
        metric,
        FIGURE_DIR / f"experiment2_{metric}_branch_comparison.png"
    )

excel_path = RESULT_ROOT / "Experiment2_Branch_Separated_Baseline_Results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    results_df.to_excel(writer, index=False, sheet_name="Baseline_Results")
    feature_summary_df.to_excel(writer, index=False, sheet_name="Feature_Summary")
    feature_info_df.to_excel(writer, index=False, sheet_name="Usable_Features")

report = f"""
# Experiment 2 — Branch-Separated Baseline DDoS vs Benign Classification

## Objective

This experiment establishes baseline DDoS-vs-Benign classification performance before applying feature selection.

## Reason for Branch Separation

Packet-based and flow-based files were processed separately because Experiment 1 showed that their feature spaces have no shared features.

## Files

- Packet DDoS/Benign files: {len(packet_files)}
- Flow DDoS/Benign files: {len(flow_files)}

## Models

- Random Forest
- Extra Trees
- Logistic Regression

## Main Results

{results_df.to_string(index=False)}

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
"""

report_path = REPORT_DIR / "Experiment2_Branch_Separated_Baseline_Report.md"
report_path.write_text(report, encoding="utf-8")

print("\n" + "=" * 80)
print("EXPERIMENT 2 COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Results folder: {RESULT_ROOT}")
print(f"Results CSV: {results_csv}")
print(f"Main Excel: {excel_path}")
print(f"Report: {report_path}")
print("=" * 80)