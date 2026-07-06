import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"D:\47\472\New-Papers\Atlam_4_2026\Data")

RESULT_ROOT = Path(
    r"D:\47\472\New-Papers\Atlam_4_2026\CIC IoT\Experiment4_Proposed_Hybrid_HHO_GWO"
)

TABLE_DIR = RESULT_ROOT / "tables"
FIGURE_DIR = RESULT_ROOT / "figures"
REPORT_DIR = RESULT_ROOT / "reports"

for folder in [RESULT_ROOT, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

MAX_ROWS_PER_FILE = 12000
RANDOM_STATE = 42

POPULATION_SIZE = 12
ITERATIONS = 15
MIN_FEATURES = 8
FITNESS_SAMPLE_SIZE = 30000

np.random.seed(RANDOM_STATE)

# ============================================================
# DATA HELPERS
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

    if len(loaded) < 2:
        raise RuntimeError(f"Not enough files for {branch_name}")

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

# ============================================================
# HYBRID HHO-GWO ENCODING
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_solution(position, n_features):
    feature_part = position[:n_features]
    hp_part = position[n_features:]

    probs = sigmoid(feature_part)
    mask = (probs > 0.5).astype(int)

    if mask.sum() < MIN_FEATURES:
        top_idx = np.argsort(probs)[-MIN_FEATURES:]
        mask[:] = 0
        mask[top_idx] = 1

    # Hyperparameters
    n_estimators = int(40 + sigmoid(hp_part[0]) * 160)       # 40–200
    max_depth = int(4 + sigmoid(hp_part[1]) * 20)            # 4–24
    min_samples_split = int(2 + sigmoid(hp_part[2]) * 10)    # 2–12
    min_samples_leaf = int(1 + sigmoid(hp_part[3]) * 5)      # 1–6

    max_features_options = ["sqrt", "log2", 0.5, 0.75]
    idx = int(sigmoid(hp_part[4]) * len(max_features_options))
    idx = min(idx, len(max_features_options) - 1)
    max_features = max_features_options[idx]

    model_options = ["RandomForest", "ExtraTrees"]
    model_idx = int(sigmoid(hp_part[5]) * len(model_options))
    model_idx = min(model_idx, len(model_options) - 1)
    model_type = model_options[model_idx]

    params = {
        "model_type": model_type,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features
    }

    return mask, params


def build_model(params):
    if params["model_type"] == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )

    return ExtraTreesClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )


def fitness(position, X_train, y_train, X_val, y_val, n_features):
    mask, params = decode_solution(position, n_features)
    selected_idx = np.where(mask == 1)[0]

    if len(selected_idx) < MIN_FEATURES:
        return 10.0

    model = build_model(params)

    model.fit(X_train[:, selected_idx], y_train)
    pred = model.predict(X_val[:, selected_idx])

    macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)

    feature_ratio = len(selected_idx) / n_features
    complexity_ratio = params["n_estimators"] / 200

    # Lower is better
    score = (1.0 - macro_f1) + 0.08 * feature_ratio + 0.02 * complexity_ratio

    return score


def hybrid_hho_gwo_optimize(X_train, y_train, X_val, y_val):
    n_features = X_train.shape[1]
    dim = n_features + 6

    population = np.random.uniform(-1, 1, (POPULATION_SIZE, dim))

    alpha = None
    beta = None
    delta = None

    alpha_score = np.inf
    beta_score = np.inf
    delta_score = np.inf

    history = []

    for t in range(ITERATIONS):
        print(f"Hybrid HHO-GWO iteration {t + 1}/{ITERATIONS}")

        for i in range(POPULATION_SIZE):
            score = fitness(population[i], X_train, y_train, X_val, y_val, n_features)

            if score < alpha_score:
                delta_score, delta = beta_score, beta
                beta_score, beta = alpha_score, alpha
                alpha_score, alpha = score, population[i].copy()
            elif score < beta_score:
                delta_score, delta = beta_score, beta
                beta_score, beta = score, population[i].copy()
            elif score < delta_score:
                delta_score, delta = score, population[i].copy()

        a = 2 - t * (2 / ITERATIONS)
        E1 = 2 * (1 - t / ITERATIONS)

        for i in range(POPULATION_SIZE):
            if np.random.rand() < 0.5:
                # GWO phase
                new_positions = []

                for leader in [alpha, beta, delta]:
                    if leader is None:
                        leader = alpha

                    r1 = np.random.rand(dim)
                    r2 = np.random.rand(dim)
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = np.abs(C * leader - population[i])
                    new_positions.append(leader - A * D)

                population[i] = np.mean(new_positions, axis=0)

            else:
                # HHO phase
                E0 = 2 * np.random.rand() - 1
                E = E1 * E0
                J = 2 * (1 - np.random.rand())

                if abs(E) >= 1:
                    rand_idx = np.random.randint(POPULATION_SIZE)
                    rand_agent = population[rand_idx]
                    population[i] = rand_agent - np.random.rand() * np.abs(
                        rand_agent - 2 * np.random.rand() * population[i]
                    )
                else:
                    population[i] = alpha - E * np.abs(J * alpha - population[i])

        mask, params = decode_solution(alpha, n_features)
        history.append({
            "Iteration": t + 1,
            "Best_Fitness": alpha_score,
            "Selected_Features": int(mask.sum()),
            "Model_Type": params["model_type"],
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": str(params["max_features"])
        })

    final_mask, final_params = decode_solution(alpha, n_features)
    selected_idx = np.where(final_mask == 1)[0]

    return selected_idx, final_params, pd.DataFrame(history)

# ============================================================
# EVALUATION
# ============================================================

def evaluate_final_model(branch_name, X_train_full, y_train_full, X_test, y_test, selected_idx, params):
    model = build_model(params)

    start_train = time.time()
    model.fit(X_train_full[:, selected_idx], y_train_full)
    training_time = time.time() - start_train

    start_infer = time.time()
    pred = model.predict(X_test[:, selected_idx])
    inference_time = time.time() - start_infer

    prob = model.predict_proba(X_test[:, selected_idx])[:, 1]

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "Branch": branch_name,
        "Model_Type": params["model_type"],
        "Selected_Features": len(selected_idx),
        "Accuracy": accuracy_score(y_test, pred),
        "Macro_F1": f1_score(y_test, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "False_Positive_Rate": fpr,
        "ROC_AUC": roc_auc_score(y_test, prob),
        "Training_Time_Seconds": training_time,
        "Inference_Time_Seconds": inference_time,
        "Inference_Time_Per_Sample": inference_time / len(y_test),
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "min_samples_leaf": params["min_samples_leaf"],
        "max_features": str(params["max_features"])
    }

    report = classification_report(
        y_test,
        pred,
        target_names=["Benign", "DDoS"],
        output_dict=True,
        zero_division=0
    )

    return metrics, cm, pd.DataFrame(report).transpose()

# ============================================================
# PLOTTING
# ============================================================

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


def plot_optimization_history(history_df, branch_name):
    safe_branch = branch_name.replace("-", "_").replace(" ", "_")

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["Iteration"], history_df["Best_Fitness"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(f"Hybrid HHO-GWO Convergence — {branch_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"experiment4_{safe_branch}_convergence.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["Iteration"], history_df["Selected_Features"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Selected Features")
    plt.title(f"Selected Feature Count — {branch_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"experiment4_{safe_branch}_selected_features_history.png", dpi=300)
    plt.close()

# ============================================================
# MAIN EXECUTION
# ============================================================

print("=" * 80)
print("Experiment 4 — Proposed Hybrid HHO-GWO Lightweight Model")
print("=" * 80)

all_csv = sorted(DATA_ROOT.rglob("*.csv"))

target_files = [
    f for f in all_csv
    if infer_attack_family(f) in ["DDoS", "Benign"]
]

packet_files = [f for f in target_files if infer_branch(f) == "Packet-Based"]
flow_files = [f for f in target_files if infer_branch(f) == "Flow-Based"]

branches = {
    "Packet-Based": packet_files,
    "Flow-Based": flow_files
}

all_metrics = []
all_selected_features = []
all_histories = []

for branch_name, files in branches.items():
    print("\n" + "=" * 80)
    print(f"Branch: {branch_name}")
    print("=" * 80)

    X, y, feature_names = load_branch_data(branch_name, files)

    # Train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Train/validation split for optimization
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    if len(X_train) > FITNESS_SAMPLE_SIZE:
        idx = np.random.choice(len(X_train), FITNESS_SAMPLE_SIZE, replace=False)
        X_opt_train = X_train[idx]
        y_opt_train = y_train[idx]
    else:
        X_opt_train = X_train
        y_opt_train = y_train

    if len(X_val) > FITNESS_SAMPLE_SIZE:
        idx = np.random.choice(len(X_val), FITNESS_SAMPLE_SIZE, replace=False)
        X_opt_val = X_val[idx]
        y_opt_val = y_val[idx]
    else:
        X_opt_val = X_val
        y_opt_val = y_val

    start_opt = time.time()
    selected_idx, best_params, history_df = hybrid_hho_gwo_optimize(
        X_opt_train, y_opt_train, X_opt_val, y_opt_val
    )
    optimization_time = time.time() - start_opt

    history_df["Branch"] = branch_name
    all_histories.append(history_df)

    selected_names = [feature_names[i] for i in selected_idx]

    for f in selected_names:
        all_selected_features.append({
            "Branch": branch_name,
            "Feature": f
        })

    metrics, cm, cls_report = evaluate_final_model(
        branch_name,
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        selected_idx,
        best_params
    )

    metrics["Total_Features"] = len(feature_names)
    metrics["Reduction_Percent"] = 100 * (1 - len(selected_idx) / len(feature_names))
    metrics["Optimization_Time_Seconds"] = optimization_time
    metrics["Best_Hyperparameters_JSON"] = json.dumps(best_params)

    all_metrics.append(metrics)

    safe_branch = branch_name.replace("-", "_").replace(" ", "_")

    pd.DataFrame(cls_report).to_csv(
        TABLE_DIR / f"experiment4_{safe_branch}_classification_report.csv",
        encoding="utf-8-sig"
    )

    plot_confusion_matrix(
        cm,
        labels=["Benign", "DDoS"],
        title=f"Experiment 4 — {branch_name}",
        output_path=FIGURE_DIR / f"experiment4_{safe_branch}_confusion_matrix.png"
    )

    plot_optimization_history(history_df, branch_name)

# ============================================================
# SAVE OUTPUTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)
selected_features_df = pd.DataFrame(all_selected_features)
history_all_df = pd.concat(all_histories, ignore_index=True)

metrics_csv = TABLE_DIR / "experiment4_proposed_hybrid_hho_gwo_metrics.csv"
features_csv = TABLE_DIR / "experiment4_selected_features.csv"
history_csv = TABLE_DIR / "experiment4_optimization_history.csv"

metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
selected_features_df.to_csv(features_csv, index=False, encoding="utf-8-sig")
history_all_df.to_csv(history_csv, index=False, encoding="utf-8-sig")

excel_path = RESULT_ROOT / "Experiment4_Proposed_Hybrid_HHO_GWO_Results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
    selected_features_df.to_excel(writer, index=False, sheet_name="Selected_Features")
    history_all_df.to_excel(writer, index=False, sheet_name="Optimization_History")

# Summary figure
for metric in ["Accuracy", "Macro_F1", "Recall", "False_Positive_Rate", "Reduction_Percent"]:
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["Branch"], metrics_df[metric])
    if metric != "Reduction_Percent":
        plt.ylim(0, 1.05)
    plt.title(f"Experiment 4 — {metric}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"experiment4_{metric}.png", dpi=300)
    plt.close()

# ============================================================
# REPORT
# ============================================================

report = f"""
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

{metrics_df.to_string(index=False)}

## Output Files

- experiment4_proposed_hybrid_hho_gwo_metrics.csv
- experiment4_selected_features.csv
- experiment4_optimization_history.csv
- Experiment4_Proposed_Hybrid_HHO_GWO_Results.xlsx
"""

report_path = REPORT_DIR / "Experiment4_Proposed_Hybrid_HHO_GWO_Report.md"
report_path.write_text(report, encoding="utf-8")

print("\n" + "=" * 80)
print("EXPERIMENT 4 COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Results folder: {RESULT_ROOT}")
print(f"Metrics CSV: {metrics_csv}")
print(f"Excel: {excel_path}")
print(f"Report: {report_path}")
print("=" * 80)