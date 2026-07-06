# Experiment 6 — Robustness Across Attack Categories

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_ROOT = Path(r"D:\47\472\New-Papers\Atlam_4_2026\Data")
EXP4_RESULTS = Path(r"D:\47\472\New-Papers\Atlam_4_2026\CIC IoT\Experiment4_Proposed_Hybrid_HHO_GWO")

EXP4_METRICS = EXP4_RESULTS / "tables" / "experiment4_proposed_hybrid_hho_gwo_metrics.csv"
EXP4_FEATURES = EXP4_RESULTS / "tables" / "experiment4_selected_features.csv"

RESULT_ROOT = Path(r"D:\47\472\New-Papers\Atlam_4_2026\CIC IoT\Experiment6_Robustness_Across_Attack_Categories")
TABLE_DIR = RESULT_ROOT / "tables"
FIGURE_DIR = RESULT_ROOT / "figures"
REPORT_DIR = RESULT_ROOT / "reports"

for d in [RESULT_ROOT, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_ROWS_PER_FILE = 12000
RANDOM_STATE = 42

TARGET_ATTACKS = ["DoS", "Mirai", "Recon", "Spoofing", "Web-Based", "Brute Force"]

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
    if "mirai" in name:
        return "Mirai"
    if "recon" in name or "scan" in name or "hostdiscovery" in name:
        return "Recon"
    if "spoof" in name or "mitm" in name:
        return "Spoofing"
    if "brute" in name or "dictionary" in name or "password" in name:
        return "Brute Force"
    if any(k in name for k in ["web", "xss", "sql", "injection", "upload", "backdoor"]):
        return "Web-Based"
    if "dos" in name:
        return "DoS"

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

def load_files_for_branch(branch_name, files, allowed_families, binary_mode=True):
    loaded = []

    for file_path in files:
        family = infer_attack_family(file_path)
        if family not in allowed_families:
            continue

        df = safe_read(file_path, nrows=MAX_ROWS_PER_FILE)
        if df is None or df.empty:
            continue

        if binary_mode:
            df["target"] = 0 if family == "Benign" else 1
        else:
            df["target"] = family

        df["attack_family"] = family
        loaded.append(df)

    if not loaded:
        return None, None, None, None

    common_cols = get_common_numeric_columns(loaded)
    if len(common_cols) == 0:
        return None, None, None, None

    aligned = []
    families = []

    for df in loaded:
        temp = df[common_cols].copy()
        temp["target"] = df["target"].values
        temp["attack_family"] = df["attack_family"].values
        aligned.append(temp)

    data = pd.concat(aligned, ignore_index=True)

    X = safe_clean_feature_matrix(data.drop(columns=["target", "attack_family"]))
    y = data["target"].values
    attack_family = data["attack_family"].values

    return X, y, attack_family, X.columns.tolist()

def build_model(params):
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", 12)),
        min_samples_split=int(params.get("min_samples_split", 2)),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        max_features=params.get("max_features", "sqrt"),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

def preprocess_fit_transform(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_np = imputer.fit_transform(X_train)
    X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)

    X_test_np = imputer.transform(X_test)
    X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)

    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test_np)

    X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train_np, X_test_np

def plot_bar(df, x_col, y_col, title, out):
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.xticks(rotation=60, ha="right")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

print("=" * 80)
print("Experiment 6 — Robustness Across Attack Categories")
print("=" * 80)

exp4_metrics = pd.read_csv(EXP4_METRICS)
exp4_features = pd.read_csv(EXP4_FEATURES)

all_csv = sorted(DATA_ROOT.rglob("*.csv"))

all_results = []
family_results = []

for branch_name in ["Packet-Based", "Flow-Based"]:
    print("\n" + "=" * 80)
    print(f"Branch: {branch_name}")
    print("=" * 80)

    branch_files = [f for f in all_csv if infer_branch(f) == branch_name]

    train_X, train_y, _, train_features = load_files_for_branch(
        branch_name,
        branch_files,
        allowed_families=["Benign", "DDoS"],
        binary_mode=True
    )

    if train_X is None:
        print(f"Skipping {branch_name}: no valid training data.")
        continue

    test_X, test_y, test_families, test_features = load_files_for_branch(
        branch_name,
        branch_files,
        allowed_families=["Benign"] + TARGET_ATTACKS,
        binary_mode=True
    )

    if test_X is None:
        print(f"Skipping {branch_name}: no valid robustness test data.")
        continue

    shared_features = [f for f in train_features if f in test_features]

    if len(shared_features) == 0:
        print(f"Skipping {branch_name}: no shared train/test features.")
        continue

    selected_names = exp4_features[exp4_features["Branch"] == branch_name]["Feature"].tolist()
    selected_names = [f for f in selected_names if f in shared_features]

    if len(selected_names) == 0:
        selected_names = shared_features

    train_X = train_X[shared_features]
    test_X = test_X[shared_features]

    train_selected_idx = [shared_features.index(f) for f in selected_names]

    ddos = pd.DataFrame(train_X)
    ddos["target"] = train_y

    ddos_pos = ddos[ddos["target"] == 1]
    benign = ddos[ddos["target"] == 0]
    min_n = min(len(ddos_pos), len(benign))

    ddos_pos = ddos_pos.sample(min_n, random_state=RANDOM_STATE)
    benign = benign.sample(min_n, random_state=RANDOM_STATE)

    train_balanced = pd.concat([ddos_pos, benign], ignore_index=True)
    train_balanced = train_balanced.sample(frac=1, random_state=RANDOM_STATE)

    X_train_df = train_balanced.drop(columns=["target"])
    y_train = train_balanced["target"].values

    X_train_np, X_test_np = preprocess_fit_transform(X_train_df, test_X)

    X_train_sel = X_train_np[:, train_selected_idx]
    X_test_sel = X_test_np[:, train_selected_idx]

    branch_metric = exp4_metrics[exp4_metrics["Branch"] == branch_name].iloc[0]
    params = json.loads(branch_metric["Best_Hyperparameters_JSON"])

    model = build_model(params)
    model.fit(X_train_sel, y_train)

    pred = model.predict(X_test_sel)

    overall = {
        "Branch": branch_name,
        "Train_Task": "DDoS_vs_Benign",
        "Test_Task": "Benign_plus_DoS_Mirai_Recon_Spoofing_Web_BruteForce",
        "Shared_Features": len(shared_features),
        "Selected_Features_Used": len(selected_names),
        "Accuracy": accuracy_score(test_y, pred),
        "Macro_F1": f1_score(test_y, pred, average="macro", zero_division=0),
        "Attack_Recall_Overall": recall_score(test_y, pred, zero_division=0)
    }

    cm = confusion_matrix(test_y, pred)
    tn, fp, fn, tp = cm.ravel()
    overall["False_Positive_Rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    overall["True_Positive_Rate"] = tp / (tp + fn) if (tp + fn) > 0 else 0

    all_results.append(overall)

    report = classification_report(
        test_y,
        pred,
        target_names=["Benign", "Attack"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        TABLE_DIR / f"experiment6_{branch_name.replace('-', '_')}_overall_classification_report.csv",
        encoding="utf-8-sig"
    )

    for fam in sorted(set(test_families)):
        idx = test_families == fam
        if idx.sum() == 0:
            continue

        y_fam = test_y[idx]
        pred_fam = pred[idx]

        if fam == "Benign":
            detection_rate = (pred_fam == 0).mean()
            false_alarm_or_miss = (pred_fam == 1).mean()
        else:
            detection_rate = (pred_fam == 1).mean()
            false_alarm_or_miss = (pred_fam == 0).mean()

        family_results.append({
            "Branch": branch_name,
            "Test_Family": fam,
            "Samples": int(idx.sum()),
            "Expected_Label": "Benign" if fam == "Benign" else "Attack",
            "Detection_Rate": detection_rate,
            "Miss_or_False_Alarm_Rate": false_alarm_or_miss
        })

    selected_df = pd.DataFrame({
        "Branch": branch_name,
        "Selected_Feature": selected_names
    })
    selected_df.to_csv(
        TABLE_DIR / f"experiment6_{branch_name.replace('-', '_')}_selected_features_used.csv",
        index=False,
        encoding="utf-8-sig"
    )

overall_df = pd.DataFrame(all_results)
family_df = pd.DataFrame(family_results)

overall_df.to_csv(TABLE_DIR / "experiment6_overall_robustness_results.csv", index=False, encoding="utf-8-sig")
family_df.to_csv(TABLE_DIR / "experiment6_family_level_generalization_results.csv", index=False, encoding="utf-8-sig")

excel_path = RESULT_ROOT / "Experiment6_Robustness_Across_Attack_Categories_Results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    overall_df.to_excel(writer, index=False, sheet_name="Overall_Robustness")
    family_df.to_excel(writer, index=False, sheet_name="Family_Level")

for branch in family_df["Branch"].unique():
    sub = family_df[family_df["Branch"] == branch]
    plot_bar(
        sub,
        "Test_Family",
        "Detection_Rate",
        f"Experiment 6 — Generalization Detection Rate — {branch}",
        FIGURE_DIR / f"experiment6_{branch.replace('-', '_')}_family_detection_rate.png"
    )

report = f"""
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

{overall_df.to_string(index=False)}

## Family-Level Results

{family_df.to_string(index=False)}
"""

(REPORT_DIR / "Experiment6_Robustness_Across_Attack_Categories_Report.md").write_text(report, encoding="utf-8")

print("\n" + "=" * 80)
print("EXPERIMENT 6 COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Results folder: {RESULT_ROOT}")
print(f"Excel: {excel_path}")
print("=" * 80)