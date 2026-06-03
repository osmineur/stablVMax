import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn import clone
from stabl.stabl import Stabl
from stabl.adaptive import ALasso, ALogitLasso
from stabl.multi_omic_pipelines import multi_omic_stabl_cv

# ── Chargement des données ─────────────────────────────────────────────────────
X_density  = pd.read_csv("../Jakob1/data/PerioII_PatientFeature_03052026_IMCdensity.csv",  index_col=0)
X_function = pd.read_csv("../Jakob1/data/PerioII_PatientFeature_03052026_IMCfunction.csv", index_col=0)
X_neighbor = pd.read_csv("../Jakob1/data/PerioII_PatientFeature_03052026_IMCneighbor.csv", index_col=0)
X_shape    = pd.read_csv("../Jakob1/data/PerioII_PatientFeature_03052026_IMCshape.csv",    index_col=0)

penalization_matrix = pd.read_csv("../Jakob1/data/penalization_matrix_perio_imc.csv", index_col=0)

# ── Outcome ────────────────────────────────────────────────────────────────────
def make_y(df):
    return pd.Series(
        [0 if x.startswith("HG") else 1 for x in df.index],
        index=df.index,
        name="outcome"
    )

y = make_y(X_density)
print(y)

# ── Filtrage par densité ───────────────────────────────────────────────────────
median_densities = 1000000 * X_density.median(axis=0)
threshold = 10

prefixe_above = set(
    "_".join(col.split("_")[:2]) for col in median_densities[median_densities >= threshold].index
)

def matches_prefix_neighbor(col, prefixes):
    parts = col.split("_")
    start = "_".join(parts[:2])
    end = parts[-1] + "_" + parts[1]
    return start in prefixes and end in prefixes

def matches_prefix_other(col, prefixes):
    parts = col.split("_")
    start = "_".join(parts[:2])
    return start in prefixes

X_neighbor_filtered     = X_neighbor[[col for col in X_neighbor.columns  if matches_prefix_neighbor(col, prefixe_above)]]
X_shape_filtered        = X_shape[[col for col in X_shape.columns         if matches_prefix_other(col, prefixe_above)]]
X_shape_filtered_no_ecc = X_shape_filtered[[col for col in X_shape_filtered.columns if not col.endswith("eccentricity")]]

# ── Filtrage par pénalisation ──────────────────────────────────────────────────
def filter_by_penalization(df, pen_matrix):
    cols_to_keep = []
    for col in df.columns:
        parts = col.split("_")
        cell_type = parts[0]
        marker = parts[-1]
        if cell_type not in pen_matrix.index or marker not in pen_matrix.columns:
            continue
        if pen_matrix.loc[cell_type, marker] != 0:
            cols_to_keep.append(col)
    return df[cols_to_keep]

X_function_pen          = filter_by_penalization(X_function, penalization_matrix)
X_function_filtered_pen = X_function_pen[[col for col in X_function_pen.columns if matches_prefix_other(col, prefixe_above)]]
X_function_filtered     = X_function[[col for col in X_function.columns          if matches_prefix_other(col, prefixe_above)]]

# ── Data dicts ────────────────────────────────────────────────────────────────
data_dict_penalized = {
    "Density":  X_density,
    "Function": X_function_filtered_pen,
    "Neighbor": X_neighbor_filtered,
    "Shape":    X_shape_filtered_no_ecc,
}
data_dict_unpenalized = {
    "Density":  X_density,
    "Function": X_function_filtered,
    "Neighbor": X_neighbor_filtered,
    "Shape":    X_shape_filtered_no_ecc,
}

print("Penalized feature counts:")
for k, v in data_dict_penalized.items():
    print(f"  {k}: {v.shape[1]} features")
print("Unpenalized feature counts:")
for k, v in data_dict_unpenalized.items():
    print(f"  {k}: {v.shape[1]} features")

# ── Estimateurs ───────────────────────────────────────────────────────────────
C_grid = np.logspace(-2, 0, 10)

lasso_cv = GridSearchCV(
    LogisticRegression(l1_ratio=1.0, solver="liblinear", class_weight="balanced", max_iter=1000000),
    param_grid={"C": C_grid}, cv=5, scoring="roc_auc", n_jobs=-1
)
alasso_cv = GridSearchCV(
    ALogitLasso(),
    param_grid={"C": C_grid}, cv=5, scoring="roc_auc", n_jobs=-1
)
en_cv = GridSearchCV(
    LogisticRegression(l1_ratio=0.5, solver="saga", class_weight="balanced", max_iter=int(1e6)),
    param_grid={"C": C_grid}, cv=5, scoring="roc_auc", n_jobs=-1
)

# ============================================================
# QUICK TEST — paramètres réduits pour tester localement
# Pour Sherlock : utilise make_stabl_estimators() + outer_splitter_100
# ============================================================

def make_stabl_estimators_quick():
    """50 bootstraps + random_permutation : rapide pour tester localement."""
    return {
        "stabl_lasso": Stabl(
            base_estimator=LogisticRegression(l1_ratio=1.0, solver="saga", tol=1e-2,
                                              class_weight="balanced", max_iter=int(1e6)),
            lambda_grid="auto",
            artificial_type="random_permutation",
            n_bootstraps=50,
        ),
        "stabl_alasso": Stabl(
            base_estimator=ALasso(tol=1e-3),
            lambda_grid="auto",
            artificial_type="random_permutation",
            n_bootstraps=50,
        ),
        "stabl_en": Stabl(
            base_estimator=ElasticNet(l1_ratio=0.5, tol=1e-3),
            lambda_grid="auto",
            artificial_type="random_permutation",
            n_bootstraps=50,
        ),
    }

def make_stabl_estimators():
    """1000 bootstraps + knockoff : version complète pour Sherlock."""
    return {
        "lasso":        clone(lasso_cv),
        "alasso":       clone(alasso_cv),
        "en":           clone(en_cv),
        "stabl_lasso":  Stabl(
            base_estimator=LogisticRegression(l1_ratio=1.0, solver="saga", tol=1e-2,
                                              class_weight="balanced", max_iter=int(1e6)),
            lambda_grid="auto", artificial_type="knockoff", n_bootstraps=1000,
        ),
        "stabl_alasso": Stabl(
            base_estimator=ALasso(tol=1e-3),
            lambda_grid="auto", artificial_type="knockoff", n_bootstraps=1000,
        ),
        "stabl_en": Stabl(
            base_estimator=ElasticNet(l1_ratio=0.5, tol=1e-3),
            lambda_grid="auto", artificial_type="knockoff", n_bootstraps=1000,
        ),
    }

outer_splitter_100   = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)  # 100 folds (Sherlock)
outer_splitter_quick = RepeatedStratifiedKFold(n_splits=5, n_repeats=2,  random_state=42)  # 10 folds  (local)

STABL_MODELS = ["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]

# ── Run 1 : Penalized ─────────────────────────────────────────────────────────
print("\n=== Run 1 : Penalized ===")
preds_penalized = multi_omic_stabl_cv(
    data_dict=data_dict_penalized,
    y=y,
    outer_splitter=outer_splitter_quick,
    estimators=make_stabl_estimators_quick(),
    task_type="binary",
    save_path="resultsJakob_penalized",
    models=STABL_MODELS.copy(),
    outer_groups=None,
    early_fusion=False,
    late_fusion=False,
    n_iter_lf=10000,
)

auc_path    = Path("resultsJakob_penalized/Training CV/auc_progress.csv")
scores_path = Path("resultsJakob_penalized/Summary/Scores training CV.csv")

if auc_path.exists():
    auc_df = pd.read_csv(auc_path)
    print(f"\n=== AUC cumulatif (penalized) — {len(auc_df)} folds complétés ===")
    print(auc_df.to_string())

if scores_path.exists():
    print("\n=== Scores finaux (penalized) ===")
    print(pd.read_csv(scores_path, index_col=0).to_string())

# ── Run 2 : Unpenalized ───────────────────────────────────────────────────────
print("\n=== Run 2 : Unpenalized ===")
preds_unpenalized = multi_omic_stabl_cv(
    data_dict=data_dict_unpenalized,
    y=y,
    outer_splitter=outer_splitter_quick,
    estimators=make_stabl_estimators_quick(),
    task_type="binary",
    save_path="resultsJakob_unpenalized",
    models=STABL_MODELS.copy(),
    outer_groups=None,
    early_fusion=False,
    late_fusion=False,
    n_iter_lf=10000,
)

auc_path    = Path("resultsJakob_unpenalized/Training CV/auc_progress.csv")
scores_path = Path("resultsJakob_unpenalized/Summary/Scores training CV.csv")

if auc_path.exists():
    auc_df = pd.read_csv(auc_path)
    print(f"\n=== AUC cumulatif (unpenalized) — {len(auc_df)} folds complétés ===")
    print(auc_df.to_string())

if scores_path.exists():
    print("\n=== Scores finaux (unpenalized) ===")
    print(pd.read_csv(scores_path, index_col=0).to_string())
