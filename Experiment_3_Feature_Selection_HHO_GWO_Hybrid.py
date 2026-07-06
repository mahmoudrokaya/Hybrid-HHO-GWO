import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(r"D:\47\472\New-Papers\Atlam_4_2026\Data")

RESULT_ROOT = Path(
    r"D:\47\472\New-Papers\Atlam_4_2026\CIC IoT\Experiment3_Feature_Selection_HHO_GWO_Hybrid"
)

TABLE_DIR = RESULT_ROOT / "tables"
FIGURE_DIR = RESULT_ROOT / "figures"
REPORT_DIR = RESULT_ROOT / "reports"

for folder in [RESULT_ROOT, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

MAX_ROWS_PER_FILE = 12000
RANDOM_STATE = 42

TOP_K_RATIO = 0.30
MIN_FEATURES = 8

META_POPULATION = 10
META_ITERATIONS = 12

FITNESS_SAMPLE_SIZE = 30000

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


def evaluate_model(X_train, X_test, y_train, y_test, selected_idx, model_name):
    if len(selected_idx) == 0:
        return None

    Xtr = X_train[:, selected_idx]
    Xte = X_test[:, selected_idx]

    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        model = ExtraTreesClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    start = time.time()
    model.fit(Xtr, y_train)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(Xte)
    infer_time = time.time() - start

    probs = model.predict_proba(Xte)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probs),
        "Training_Time_Seconds": train_time,
        "Inference_Time_Seconds": infer_time
    }


def fitness_function(mask, X, y):
    selected = np.where(mask == 1)[0]

    if len(selected) < MIN_FEATURES:
        return 1.0

    if len(X) > FITNESS_SAMPLE_SIZE:
        idx = np.random.choice(len(X), FITNESS_SAMPLE_SIZE, replace=False)
        X_fit = X[idx][:, selected]
        y_fit = y[idx]
    else:
        X_fit = X[:, selected]
        y_fit = y

    Xtr, Xva, ytr, yva = train_test_split(
        X_fit,
        y_fit,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_fit
    )

    clf = RandomForestClassifier(
        n_estimators=60,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)

    f1 = f1_score(yva, pred, zero_division=0)
    reduction_penalty = len(selected) / X.shape[1]

    return (1.0 - f1) + 0.05 * reduction_penalty


def continuous_to_binary(position):
    prob = 1 / (1 + np.exp(-position))
    return (prob > 0.5).astype(int)


# ============================================================
# FEATURE SELECTION METHODS
# ============================================================

def select_all_features(X, y):
    return np.arange(X.shape[1])


def select_mutual_information(X, y):
    k = max(MIN_FEATURES, int(X.shape[1] * TOP_K_RATIO))
    scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return np.argsort(scores)[-k:]


def select_rf_importance(X, y):
    k = max(MIN_FEATURES, int(X.shape[1] * TOP_K_RATIO))
    clf = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X, y)
    scores = clf.feature_importances_
    return np.argsort(scores)[-k:]


def select_hho(X, y):
    dim = X.shape[1]
    pop = np.random.uniform(-1, 1, (META_POPULATION, dim))

    rabbit = None
    rabbit_score = np.inf

    for t in range(META_ITERATIONS):
        for i in range(META_POPULATION):
            mask = continuous_to_binary(pop[i])
            score = fitness_function(mask, X, y)

            if score < rabbit_score:
                rabbit_score = score
                rabbit = pop[i].copy()

        E1 = 2 * (1 - (t / META_ITERATIONS))

        for i in range(META_POPULATION):
            E0 = 2 * np.random.rand() - 1
            E = E1 * E0
            J = 2 * (1 - np.random.rand())

            if abs(E) >= 1:
                rand_idx = np.random.randint(META_POPULATION)
                X_rand = pop[rand_idx]
                pop[i] = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * pop[i])
            else:
                pop[i] = rabbit - E * abs(J * rabbit - pop[i])

    final_mask = continuous_to_binary(rabbit)
    selected = np.where(final_mask == 1)[0]

    if len(selected) < MIN_FEATURES:
        selected = select_rf_importance(X, y)

    return selected


def select_gwo(X, y):
    dim = X.shape[1]
    wolves = np.random.uniform(-1, 1, (META_POPULATION, dim))

    alpha = beta = delta = None
    alpha_score = beta_score = delta_score = np.inf

    for t in range(META_ITERATIONS):
        for i in range(META_POPULATION):
            mask = continuous_to_binary(wolves[i])
            score = fitness_function(mask, X, y)

            if score < alpha_score:
                delta_score, delta = beta_score, beta
                beta_score, beta = alpha_score, alpha
                alpha_score, alpha = score, wolves[i].copy()
            elif score < beta_score:
                delta_score, delta = beta_score, beta
                beta_score, beta = score, wolves[i].copy()
            elif score < delta_score:
                delta_score, delta = score, wolves[i].copy()

        a = 2 - t * (2 / META_ITERATIONS)

        for i in range(META_POPULATION):
            new_pos_parts = []

            for leader in [alpha, beta, delta]:
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = abs(C * leader - wolves[i])
                new_pos_parts.append(leader - A * D)

            wolves[i] = np.mean(new_pos_parts, axis=0)

    final_mask = continuous_to_binary(alpha)
    selected = np.where(final_mask == 1)[0]

    if len(selected) < MIN_FEATURES:
        selected = select_rf_importance(X, y)

    return selected


def select_hybrid_hho_gwo(X, y):
    hho_selected = set(select_hho(X, y))
    gwo_selected = set(select_gwo(X, y))
    hybrid = sorted(list(hho_selected | gwo_selected))

    max_features = max(MIN_FEATURES, int(X.shape[1] * 0.45))

    if len(hybrid) > max_features:
        rf_selected = select_rf_importance(X, y)
        hybrid = sorted(list(set(hybrid) & set(rf_selected)))

    if len(hybrid) < MIN_FEATURES:
        hybrid = sorted(list(select_rf_importance(X, y)))

    return np.array(hybrid)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("=" * 80)
print("Experiment 3 — Feature Selection: MI, RF, HHO, GWO, Hybrid HHO-GWO")
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

selection_methods = {
    "All_Features": select_all_features,
    "Mutual_Information": select_mutual_information,
    "RandomForest_Importance": select_rf_importance,
    "HHO": select_hho,
    "GWO": select_gwo,
    "Hybrid_HHO_GWO": select_hybrid_hho_gwo
}

all_results = []
selected_features_records = []

for branch_name, files in branches.items():
    print("\n" + "=" * 80)
    print(f"Branch: {branch_name}")
    print("=" * 80)

    X, y, feature_names = load_branch_data(branch_name, files)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    for method_name, selector in selection_methods.items():
        print(f"\nFeature selection method: {method_name}")

        start = time.time()
        selected_idx = selector(X_train, y_train)
        selection_time = time.time() - start

        selected_idx = sorted(list(set(selected_idx)))
        selected_feature_names = [feature_names[i] for i in selected_idx]

        reduction_percent = 100 * (1 - len(selected_idx) / len(feature_names))

        for feat in selected_feature_names:
            selected_features_records.append({
                "Branch": branch_name,
                "Selection_Method": method_name,
                "Feature": feat
            })

        for model_name in ["RandomForest", "ExtraTrees"]:
            metrics = evaluate_model(
                X_train,
                X_test,
                y_train,
                y_test,
                selected_idx,
                model_name
            )

            metrics.update({
                "Branch": branch_name,
                "Selection_Method": method_name,
                "Total_Features": len(feature_names),
                "Selected_Features": len(selected_idx),
                "Reduction_Percent": reduction_percent,
                "Feature_Selection_Time_Seconds": selection_time
            })

            all_results.append(metrics)

results_df = pd.DataFrame(all_results)
features_df = pd.DataFrame(selected_features_records)

results_csv = TABLE_DIR / "experiment3_feature_selection_results.csv"
features_csv = TABLE_DIR / "experiment3_selected_features.csv"

results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
features_df.to_csv(features_csv, index=False, encoding="utf-8-sig")

excel_path = RESULT_ROOT / "Experiment3_Feature_Selection_HHO_GWO_Hybrid_Results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    results_df.to_excel(writer, index=False, sheet_name="Results")
    features_df.to_excel(writer, index=False, sheet_name="Selected_Features")

# ============================================================
# FIGURES
# ============================================================

for branch in results_df["Branch"].unique():
    subset = results_df[results_df["Branch"] == branch]

    for metric in ["F1", "ROC_AUC", "Accuracy", "Reduction_Percent"]:
        plt.figure(figsize=(12, 5))
        labels = subset["Selection_Method"] + " — " + subset["Model"]
        plt.bar(labels, subset[metric])
        plt.xticks(rotation=70, ha="right")
        if metric != "Reduction_Percent":
            plt.ylim(0, 1.05)
        plt.title(f"Experiment 3 — {branch} — {metric}")
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / f"experiment3_{branch.replace('-', '_')}_{metric}.png",
            dpi=300
        )
        plt.close()

# ============================================================
# REPORT
# ============================================================

report = f"""
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

{results_df.to_string(index=False)}
"""

(REPORT_DIR / "Experiment3_Feature_Selection_Report.md").write_text(
    report,
    encoding="utf-8"
)

print("\n" + "=" * 80)
print("EXPERIMENT 3 COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Results folder: {RESULT_ROOT}")
print(f"Results CSV: {results_csv}")
print(f"Excel: {excel_path}")
print("=" * 80)