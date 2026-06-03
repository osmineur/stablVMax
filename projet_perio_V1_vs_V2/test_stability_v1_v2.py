"""
test_stability_v1_v2.py
=======================
Version locale du pipeline Sherlock (projet_perio_V1_vs_V2/sendOut.py).
Données : datalayers penalized sauvegardés localement.
Paramètres réduits pour run local rapide.

Structure des sorties (miroir de Sherlock) :
  test_results/
  ├── penalized_v1/          ← résultats multi_omic_stabl_cv
  ├── penalized_v2/          ← résultats multi_omic_stabl_cv
  ├── post_processing/
  │   ├── ROC/               ← courbes ROC par modèle
  │   ├── AUC/               ← boxplots AUC V1 vs V2
  │   ├── jaccard/           ← similarité des features sélectionnées
  │   ├── variance/          ← Var(score) vs B, V1 vs V2 (Théorème 2)
  │   ├── fdp_theory/        ← ε_B, B_0, terme résiduel |∂|/D
  │   ├── gaussianity/       ← d_TV proxy, condition ε < η_t/2
  │   └── lambda_min/        ← λ_min(Σ_emp) vs λ_min(Σ_LW)
  └── correlation_results_tests/
      ├── penalized_v1/<Model>/  ← ranking.csv, barplot, heatmaps, network
      ├── penalized_v2/<Model>/
      └── summary.csv
"""

import sys
import os
import shutil
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import clone
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.covariance import LedoitWolf
from scipy.stats import ks_1samp, norm as sp_norm, probplot
from xgboost import XGBClassifier

BASE     = Path(__file__).parent                           # projet_perio_V1_vs_V2/
DATA_DIR = BASE.parent.parent / "Jakob1" / "data"
OUT_DIR  = BASE / "test_results"

sys.path.insert(0, str(BASE.parent))                       # stablVMax/ pour les imports stabl.*
sys.path.insert(0, str(BASE))                              # projet_perio_V1_vs_V2/ pour correlation_IMC_CyTOF

from stabl.stabl              import Stabl as StablV1
from stabl.stablV2            import Stabl as StablV2
from stabl.adaptive           import ALasso, ALogitLasso
from stabl.preprocessing      import LowInfoFilter, CorrelationFilter
from stabl.multi_omic_pipelines import multi_omic_stabl_cv
import stabl.stabl   as _v1_module
import stabl.stablV2  as _v2_module
import stabl.multi_omic_pipelines as _mop

print("Imports OK")

# ── Paramètres locaux (réduits par rapport à Sherlock) ────────────────────────

RUN_CV        = False  # True → lance les runs ML + ROC/AUC ; False → analyses théoriques seulement
N_SPLITS      = 2
N_REPEATS     = 3    # → 100 folds
RANDOM_STATE  = 42
N_JOBS        = -1

# CV — B fixe
N_BOOTSTRAPS_CV = 50
# Théorie — grille de B
B_VALUES      = [50000]
K_SEEDS       = 1

# FDP+ theory
DELTA_VALUES  = [0.01]

# Augmentation du dataset (après preprocessing)
AUGMENT_DATA   = False
AUGMENT_FACTOR = 5      # copies synthétiques par sample
AUGMENT_NOISE  = 0.05   # bruit gaussien (en unités std, features déjà scalées)


# Palette couleurs (même que Sherlock)
PALETTE = {
    "penalized_v1": "#C41E3A",
    "penalized_v2": "#001A7B",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _patch_pipeline(version):
    module = _v1_module if version == 1 else _v2_module
    _mop.save_stabl_results = module.save_stabl_results

def _subdir(name):
    d = OUT_DIR / "post_processing" / name
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── Données ───────────────────────────────────────────────────────────────────

X_density  = pd.read_csv(DATA_DIR / "PerioII_PatientFeature_03052026_IMCdensity.csv",  index_col=0)
X_function = pd.read_csv(DATA_DIR / "PerioII_PatientFeature_03052026_IMCfunction.csv", index_col=0)
X_neighbor = pd.read_csv(DATA_DIR / "PerioII_PatientFeature_03052026_IMCneighbor.csv", index_col=0)
pen_matrix = pd.read_csv(DATA_DIR / "penalization_matrix_perio_imc.csv", index_col=0)

y = pd.Series(
    [0 if idx.startswith("HG") else 1 for idx in X_density.index],
    index=X_density.index, name="outcome"
)

# Filtration identique à sendOut.py (density_threshold=10, use_penalization=True)
DENSITY_THRESHOLD = 10

def _filter_by_penalization(df, pm):
    return df[[c for c in df.columns
               if (p := c.split("_"))[0] in pm.index
               and p[-1] in pm.columns
               and pm.loc[p[0], p[-1]] != 0]]

def _matches_neighbor(col, prefixes):
    parts = col.split("_")
    return "_".join(parts[:2]) in prefixes and parts[-1] + "_" + parts[1] in prefixes

def _matches_other(col, prefixes):
    return "_".join(col.split("_")[:2]) in prefixes

median_densities = 1_000_000 * X_density.median(axis=0)
prefixe_above = set(
    "_".join(c.split("_")[:2]) for c in median_densities[median_densities >= DENSITY_THRESHOLD].index
)
no_other = lambda df: df[[c for c in df.columns if "other" not in c.lower()]]

X_function_pen = _filter_by_penalization(X_function, pen_matrix)

X_density  = no_other(X_density)
X_function = no_other(X_function_pen[[c for c in X_function_pen.columns if _matches_other(c, prefixe_above)]])
X_neighbor = no_other(X_neighbor[[c for c in X_neighbor.columns if _matches_neighbor(c, prefixe_above)]])

data_dict = {
    "Density":  X_density,
    "Function": X_function,
    "Neighbor": X_neighbor,
}

print(f"\nDonnées (penalized) :")
for name, df in data_dict.items():
    print(f"  {name:10s} : {df.shape[0]} patients × {df.shape[1]} features")
print(f"  Classes : {dict(zip(*np.unique(y.values, return_counts=True)))}")

# ── Preprocessing ─────────────────────────────────────────────────────────────

def make_pipelines():
    density = Pipeline([
        ("to_million", FunctionTransformer(lambda X: X * 1e6, feature_names_out="one-to-one")),
        ("variance",   VarianceThreshold(0.01)),
        ("corr",       CorrelationFilter(threshold="auto")),
        ("lif",        LowInfoFilter()),
        ("impute",     SimpleImputer(strategy="median")),
        ("std",        StandardScaler()),
    ])
    noisy = Pipeline([
        ("variance", VarianceThreshold(0.01)),
        ("lif",      LowInfoFilter()),
        ("impute",   SimpleImputer(strategy="median")),
        ("std",      StandardScaler()),
    ])
    return density, noisy

density_pipe, noisy_pipe = make_pipelines()
preprocessing_overrides = {
    "Density":  density_pipe,
    "Function": noisy_pipe,
    "Neighbor": noisy_pipe,
}

# ── Estimateurs (6 modèles — miroir Sherlock) ─────────────────────────────────

C_grid = np.logspace(-2, 0, 5)   # réduit pour le local (Sherlock : 10)

def make_estimators(StablClass, n_bootstraps):
    inner_cv  = StratifiedKFold(n_splits=min(3, N_SPLITS), shuffle=True, random_state=RANDOM_STATE)
    lasso_cv  = GridSearchCV(LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=int(1e6)), {"C": C_grid}, cv=inner_cv, scoring="roc_auc", n_jobs=N_JOBS)
    alasso_cv = GridSearchCV(ALogitLasso(solver="liblinear"), {"C": C_grid}, cv=inner_cv, scoring="roc_auc", n_jobs=N_JOBS)
    en_cv     = GridSearchCV(LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", class_weight="balanced", max_iter=int(1e6)), {"C": C_grid}, cv=inner_cv, scoring="roc_auc", n_jobs=N_JOBS)
    xgb_cv    = GridSearchCV(XGBClassifier(n_jobs=1, eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE),
                             {"max_depth": [3, 5], "n_estimators": [50, 100]},
                             cv=inner_cv, scoring="roc_auc", n_jobs=N_JOBS)
    return {
        "lasso":          clone(lasso_cv),
        "alasso":         clone(alasso_cv),
        "en":             clone(en_cv),
        "xgboost":        clone(xgb_cv),
        "stabl_lasso":    StablClass(base_estimator=LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=int(1e6)), lambda_grid="auto", artificial_type="knockoff", n_bootstraps=n_bootstraps, n_jobs=N_JOBS),
        "stabl_alasso":   StablClass(base_estimator=ALasso(tol=1e-3),                   lambda_grid="auto", artificial_type="knockoff", n_bootstraps=n_bootstraps, n_jobs=N_JOBS),
        "stabl_en":       StablClass(base_estimator=ElasticNet(l1_ratio=0.5, tol=1e-3), lambda_grid="auto", artificial_type="knockoff", n_bootstraps=n_bootstraps, n_jobs=N_JOBS),
    }

def jac(a, b): return len(a & b) / len(a | b) if (a or b) else 1.0

# ── Nettoyage complet de test_results à chaque run ───────────────────────────

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Runs V1 et V2 + Post-processing AUC/ROC/Jaccard ──────────────────────────
# Désactivé : mettre RUN_CV = True pour lancer les runs ML

STABL_MODELS = ["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]
ALL_MODELS   = STABL_MODELS + ["Lasso", "ALasso", "ElasticNet", "XGBoost"]

if RUN_CV:
    for version, StablClass in [(1, StablV1), (2, StablV2)]:
        label = f"penalized_v{version}"
        print(f"\n{'='*65}")
        print(f"RUN {label.upper()} — {N_BOOTSTRAPS_CV} bootstraps, {N_SPLITS}×{N_REPEATS} folds")
        print(f"{'='*65}")

        save_path = OUT_DIR / label
        if save_path.exists():
            shutil.rmtree(save_path)

        _patch_pipeline(version)
        splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

        multi_omic_stabl_cv(
            data_dict=data_dict, y=y,
            outer_splitter=splitter,
            estimators=make_estimators(StablClass, n_bootstraps=N_BOOTSTRAPS_CV),
            task_type="binary",
            save_path=str(save_path),
            models=ALL_MODELS,
            outer_groups=None,
            early_fusion=False, late_fusion=False, n_iter_lf=10000,
            preprocessing_overrides=preprocessing_overrides,
        )
        print(f"Résultats sauvegardés → {save_path}/")

    runs = [
        {"name": f"penalized_v{v}", "save_path": OUT_DIR / f"penalized_v{v}",
         "stabl_version": f"v{v}"}
        for v in [1, 2]
    ]
    auc_data  = {}
    pred_data = {}
    feat_sets = {}
    for run in runs:
        sp  = run["save_path"]
        key = run["stabl_version"]
        auc_data[key]  = {}
        pred_data[key] = {}
        auc_path = sp / "Training CV" / "auc_progress.csv"
        if auc_path.exists():
            df_auc = pd.read_csv(auc_path, index_col=0)
            for m in ALL_MODELS:
                if m in df_auc.columns:
                    auc_data[key][m] = df_auc[m].dropna().tolist()
        for m in ALL_MODELS:
            pred_path = sp / "Training CV" / m / f"{m} predictions.csv"
            if pred_path.exists():
                df_pred = pd.read_csv(pred_path)
                score_cols = [c for c in df_pred.columns if c not in ("Patient", "outcome")]
                df_pred[score_cols] = df_pred[score_cols].fillna(0.5)
                pred_data[key][m] = df_pred
        for m in ALL_MODELS:
            feat_path = sp / "Training CV" / f"Selected Features {m}.csv"
            fkey = f"{run['stabl_version']} / {m}"
            feat_sets[fkey] = set(pd.read_csv(feat_path).iloc[:, 0].tolist()) if feat_path.exists() else set()

    from sklearn.metrics import roc_curve, auc as sk_auc
    C_V1 = "#C41E3A"; C_V2 = "#001A7B"
    out_roc = _subdir("ROC")
    for m in ALL_MODELS:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        for v, color in [("v1", C_V1), ("v2", C_V2)]:
            if v not in pred_data or m not in pred_data[v]: continue
            df_p = pred_data[v][m]
            score_col = [c for c in df_p.columns if c not in ("Patient", "outcome")][0]
            fpr, tpr, _ = roc_curve(df_p["outcome"], df_p[score_col])
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{v.upper()}  AUC={roc_auc:.3f}")
        ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC — {m} (V1 vs V2, B={N_BOOTSTRAPS_CV})")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_roc / f"ROC_{m.replace(' ','_')}.pdf", dpi=150)
        plt.close(fig)
    print(f"\nROC curves → {out_roc}/")

    out_auc = _subdir("AUC")
    width = 0.35; x_pos = np.arange(len(ALL_MODELS))
    fig, ax = plt.subplots(figsize=(max(8, len(ALL_MODELS) * 1.5), 5))
    for v, color, offset in [("v1", C_V1, -width/2), ("v2", C_V2, +width/2)]:
        means = [np.mean(auc_data.get(v, {}).get(m, [np.nan])) for m in ALL_MODELS]
        stds  = [np.std(auc_data.get(v,  {}).get(m, [np.nan])) for m in ALL_MODELS]
        ax.bar(x_pos + offset, means, width, color=color, alpha=0.75, label=v.upper())
        ax.errorbar(x_pos + offset, means, yerr=stds, fmt="none", color="black", capsize=4, lw=1)
    ax.set_xticks(x_pos); ax.set_xticklabels(ALL_MODELS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ROC AUC (moyenne ± std sur folds)"); ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"AUC V1 vs V2 — B={N_BOOTSTRAPS_CV}, {N_SPLITS}×{N_REPEATS} folds", fontsize=11)
    fig.tight_layout(); fig.savefig(out_auc / "AUC_V1_V2.pdf", dpi=150); plt.close(fig)
    print(f"AUC → {out_auc}/")

    out_jac = _subdir("jaccard")
    labels_jac = list(feat_sets.keys()); n_j = len(labels_jac)
    M = np.array([[jac(feat_sets[l1], feat_sets[l2]) for l2 in labels_jac] for l1 in labels_jac])
    pd.DataFrame(M, index=labels_jac, columns=labels_jac).to_csv(out_jac / "jaccard_similarity.csv")
    fig, ax = plt.subplots(figsize=(max(8, n_j), max(6, n_j - 2)))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n_j)); ax.set_yticks(range(n_j))
    ax.set_xticklabels(labels_jac, rotation=45, ha="right", fontsize=5)
    ax.set_yticklabels(labels_jac, fontsize=5)
    for i in range(n_j):
        for j in range(n_j):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=4)
    plt.colorbar(im, ax=ax, label="Jaccard"); ax.set_title("Feature similarity — V1 vs V2")
    fig.tight_layout(); fig.savefig(out_jac / "jaccard_heatmap.pdf", dpi=150); plt.close(fig)
    print(f"Jaccard → {out_jac}/")

else:
    print("\nRUN_CV=False — runs ML et AUC/ROC/Jaccard ignorés.")

# ── 4. λ_min diagnostics (V2 / LW) ───────────────────────────────────────────

out_lmin = _subdir("lambda_min")
lw_records = []
for layer_name, (df_layer, pipe) in [("Density",  (X_density,  density_pipe)),
                                      ("Function", (X_function, noisy_pipe)),
                                      ("Neighbor", (X_neighbor, noisy_pipe))]:
    density_pipe2, noisy_pipe2 = make_pipelines()
    pipe2 = density_pipe2 if layer_name == "Density" else noisy_pipe2
    X_prep = pipe2.fit_transform(df_layer.values, y.values)
    n, p = X_prep.shape
    Sigma_emp = np.cov(X_prep.T)
    lmin_emp  = float(np.linalg.eigvalsh(Sigma_emp).min())
    lw        = LedoitWolf().fit(X_prep)
    lmin_lw   = float(np.linalg.eigvalsh(lw.covariance_).min())
    delta_lw  = min(2 * lmin_lw, 1.0)
    diag      = np.diag(lw.covariance_)
    ko_corr   = float(np.nanmean(np.abs(1.0 - delta_lw / np.where(diag > 0, diag, np.nan))))
    lw_records.append({"layer": layer_name, "n": n, "p": p,
                        "lambda_min_emp": lmin_emp, "lambda_min_LW": lmin_lw,
                        "shrinkage": float(lw.shrinkage_), "mean_ko_corr": ko_corr})
    print(f"  [{layer_name}] n={n}, p={p} | λ_min(Σ)={lmin_emp:.4f} | λ_min(Σ_LW)={lmin_lw:.4f} | shrinkage={lw.shrinkage_:.3f}")

df_lmin = pd.DataFrame(lw_records)
df_lmin.to_csv(out_lmin / "lambda_min.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(df_lmin))
w = 0.35
ax.bar(x - w/2, df_lmin["lambda_min_emp"], w, color="tomato",    alpha=0.7, label="Σ_empirique (dégénère si n<p)")
ax.bar(x + w/2, df_lmin["lambda_min_LW"],  w, color="steelblue", alpha=0.8, label="Σ_LW (non-dégénéré)")
ax.set_xticks(x); ax.set_xticklabels(df_lmin["layer"])
ax.set_ylabel("λ_min"); ax.set_title("λ_min(Σ_emp) vs λ_min(Σ_LW) par datalayer")
ax.axhline(0, color="black", lw=0.8); ax.legend()
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(out_lmin / "lambda_min.pdf", dpi=150)
plt.close(fig)
print(f"λ_min → {out_lmin}/")


# ── 5-8. Analyses théoriques — 3 layers (Density, Function, Neighbor) ─────────

from itertools import combinations as _comb

def _t_star_constrained(m, t_min=0.0):
    """Borne supérieure de l'intervalle où FDP+(t) est minimal (t >= t_min)."""
    thresh = np.array(m.fdr_threshold_range)
    fdrs   = np.array(m.FDRs_)
    mask   = thresh >= t_min
    if mask.any():
        fdrs_m  = fdrs[mask]
        thresh_m = thresh[mask]
        return float(thresh_m[fdrs_m == fdrs_m.min()][0])
    return float(thresh[-1])


def _t_star_opt_constrained(m, scores, eps_B, eps_tv=0.0, t_min=0.0):
    """
    argmin_{t : ∂+(t) = ∅} FDP+(t)
    ∂+(t) = {j : score(j) ∈ (t, t + 2*(ε_B + ε_tv)]}  (condition nouvelle, basée sur les scores)
    ε_tv  = proxy TV (KS moyen) ; 0 si données gaussiennes

    Fallback sans contrainte si aucun t ne satisfait ∂+(t) = ∅ (exact=False).
    Retourne (t_star, bound, eta_t, exact).
    """
    eps_total = eps_B + eps_tv
    thresh = np.array(m.fdr_threshold_range)
    fdrs   = np.array(m.FDRs_)
    mask   = thresh >= t_min
    if not mask.any():
        t = float(thresh[-1])
        return t, float("inf"), 0.0, False
    thresh_m = thresh[mask]
    fdrs_m   = fdrs[mask]

    objs     = np.empty(len(thresh_m))
    feasible = np.empty(len(thresh_m), dtype=bool)
    etas     = np.empty(len(thresh_m))
    for i, t in enumerate(thresh_m):
        above      = scores[scores > t]
        etas[i]    = float(above.min() - t) if len(above) > 0 else float(1.0 - t)
        D          = max(1, len(above))
        bdry       = int(np.sum((scores > t) & (scores <= t + 2 * eps_total)))
        objs[i]    = fdrs_m[i] + bdry / D
        feasible[i] = bdry == 0   # ∂+(t) = ∅

    if feasible.any():
        sub_objs = np.where(feasible, objs, np.inf)
        min_obj  = sub_objs.min()
        best_idx = np.where(sub_objs == min_obj)[0][0]
        return float(thresh_m[best_idx]), float(min_obj), float(etas[best_idx]), True

    # fallback : aucun t ne satisfait ∂+(t) = ∅
    min_obj  = objs.min()
    best_idx = np.where(objs == min_obj)[0][0]
    return float(thresh_m[best_idx]), float(objs[best_idx]), float(etas[best_idx]), False

def _bernstein_eps_featurewise(sigma2_vec, B, log_t):
    """Maurer-Pontil empirical Bernstein tolerance eps_j per feature (shape (p,)).
    Uses B-1 and 7/3 instead of B and 2/3 because sigma2 is estimated from the same bootstraps."""
    B_eff = max(B - 1, 1)
    disc  = (7/3)**2 * log_t**2 + 8 * B_eff * log_t * sigma2_vec
    return ((7/3) * log_t + np.sqrt(np.maximum(disc, 0.0))) / (2 * B_eff)


def _t_star_unconstrained(m, t_min=0.0):
    """argmin_{t >= t_min} FDP+(t)  — minimisation pure sans contrainte ni terme de barrière.
    Un seul t*, commun à Hoeffding et Bernstein FW.
    Retourne (t_star, fdp_value).
    """
    thresh = np.array(m.fdr_threshold_range)
    fdrs   = np.array(m.FDRs_)
    mask   = thresh >= t_min
    if not mask.any():
        return float(thresh[-1]), float(fdrs[-1])
    thresh_m = thresh[mask]
    fdrs_m   = fdrs[mask]
    best_idx = int(np.where(fdrs_m == fdrs_m.min())[0][0])
    return float(thresh_m[best_idx]), float(fdrs_m[best_idx])


def _t_star_opt_bernstein(m, scores, sigma2_vec, B, log_t, sigma2_ko_vec=None, eps_tv=0.0, t_min=0.0):
    """
    Maurer-Pontil empirical Bernstein feature-wise constrained minimization (Thm 5 Bis).
    eps_j    = MP-Bernstein(sigma2_j, B, log_t)      — feature-wise depuis score_variance_
    eps_ko_j = MP-Bernstein(sigma2_ko_j, B, log_t)  — feature-wise depuis ko_score_variance_
    Constraint: ∀j ∈ S(t): score(j) > t + eps_j + eps_ko_j + eps_tv  <=> ∂+(t) = ∅.
    Returns (t_star, bound, min_margin, exact).
    """
    eps_j = _bernstein_eps_featurewise(sigma2_vec, B, log_t)          # (p,)
    if sigma2_ko_vec is None:
        sigma2_ko_vec = sigma2_vec                                      # fallback symétrique
    eps_ko_j = _bernstein_eps_featurewise(sigma2_ko_vec, B, log_t)    # (p,)
    eps_tot  = eps_j + eps_ko_j + eps_tv                               # (p,)

    thresh = np.array(m.fdr_threshold_range)
    fdrs   = np.array(m.FDRs_)
    mask   = thresh >= t_min
    if not mask.any():
        return float(thresh[-1]), float("inf"), 0.0, False
    thresh_m = thresh[mask]
    fdrs_m   = fdrs[mask]

    objs     = np.empty(len(thresh_m))
    feasible = np.empty(len(thresh_m), dtype=bool)
    for i, t in enumerate(thresh_m):
        sel         = scores > t
        bdry        = int(np.sum(sel & (scores <= t + eps_tot)))
        D           = max(1, int(sel.sum()))
        objs[i]     = fdrs_m[i] + bdry / D
        feasible[i] = bdry == 0

    if feasible.any():
        sub_objs = np.where(feasible, objs, np.inf)
        best_idx = int(np.where(sub_objs == sub_objs.min())[0][0])
        t_star   = float(thresh_m[best_idx])
        sel      = scores > t_star
        margin   = float(np.min(scores[sel] - t_star - eps_tot[sel])) if sel.any() else 0.0
        return t_star, float(sub_objs[best_idx]), margin, True

    best_idx = int(np.where(objs == objs.min())[0][0])
    t_star   = float(thresh_m[best_idx])
    sel      = scores > t_star
    margin   = float(np.min(scores[sel] - t_star - eps_tot[sel])) if sel.any() else 0.0
    return t_star, float(objs[best_idx]), margin, False


def _t_star_composite_hoeff(m, scores, eps_B, eps_tv=0.0, t_min=0.0):
    """argmin FDP+(t) + |∂+(t, 2ε)|/D — sans contrainte bdry==0 (Hoeffding)."""
    eps_total = eps_B + eps_tv
    thresh  = np.array(m.fdr_threshold_range)
    fdrs    = np.array(m.FDRs_)
    mask    = thresh >= t_min
    if not mask.any():
        return float(thresh[-1]), float(fdrs[-1])
    thresh_m = thresh[mask]
    fdrs_m   = fdrs[mask]
    objs = np.array([
        fdrs_m[i] + int(np.sum((scores > t) & (scores <= t + 2*eps_total))) / max(1, int(np.sum(scores > t)))
        for i, t in enumerate(thresh_m)
    ])
    best_idx = int(np.where(objs == objs.min())[0][0])
    return float(thresh_m[best_idx]), float(objs[best_idx])


def _t_star_composite_bfw(m, scores, sigma2_vec, B, log_t, sigma2_ko_vec=None, eps_tv=0.0, t_min=0.0):
    """argmin FDP+(t) + |∂+(t, ε_j+ε_ko_j)|/D — sans contrainte bdry==0 (Bernstein FW)."""
    eps_j = _bernstein_eps_featurewise(sigma2_vec, B, log_t)
    if sigma2_ko_vec is None:
        sigma2_ko_vec = sigma2_vec
    eps_ko_j = _bernstein_eps_featurewise(sigma2_ko_vec, B, log_t)
    eps_tot  = eps_j + eps_ko_j + eps_tv
    thresh  = np.array(m.fdr_threshold_range)
    fdrs    = np.array(m.FDRs_)
    mask    = thresh >= t_min
    if not mask.any():
        return float(thresh[-1]), float(fdrs[-1])
    thresh_m = thresh[mask]
    fdrs_m   = fdrs[mask]
    objs = np.array([
        fdrs_m[i] + int(np.sum((scores > t) & (scores <= t + eps_tot))) / max(1, int(np.sum(scores > t)))
        for i, t in enumerate(thresh_m)
    ])
    best_idx = int(np.where(objs == objs.min())[0][0])
    return float(thresh_m[best_idx]), float(objs[best_idx])


BASE_ESTIMATORS = {
    "alasso": ALogitLasso(solver="liblinear"),
}
colors_est = {"lasso": "#C41E3A", "alasso": "#001A7B", "en": "#2CA02C"}

_ls_cycle = ["-", "--", ":", "-.", (0,(3,1,1,1)), (0,(5,2)), (0,(1,1))]
_mk_cycle = ["o", "s", "^", "D", "v", "P", "X"]
_c_cycle  = ["#C41E3A","#001A7B","#2CA02C","#FF7F0E","#9467BD","#8C564B","#E377C2"]
ls_delta  = {d: _ls_cycle[i % len(_ls_cycle)] for i, d in enumerate(DELTA_VALUES)}
mk_delta  = {d: _mk_cycle[i % len(_mk_cycle)] for i, d in enumerate(DELTA_VALUES)}
c_delta   = {d: _c_cycle[i  % len(_c_cycle)]  for i, d in enumerate(DELTA_VALUES)}

rng_seeds = np.random.default_rng(RANDOM_STATE).integers(0, 2**31, size=K_SEEDS).tolist()

# Layers à analyser (même filtrage que la CV)
layers_theory = {
    "Density": (X_density, "density"),
}

all_synthesis = []

for layer_name, (X_df, pipe_type) in layers_theory.items():
    print(f"\n{'='*65}")
    print(f"LAYER : {layer_name.upper()}")
    print(f"{'='*65}")

    out_var   = _subdir(f"variance/{layer_name}")
    out_fdp   = _subdir(f"fdp_theory/{layer_name}")
    out_gauss = _subdir(f"gaussianity/{layer_name}")

    # Fresh pipeline (pas de gaussianisation)
    dp, np_ = make_pipelines()
    pipe = dp if pipe_type == "density" else np_
    X = pipe.fit_transform(X_df.values, y.values)
    n_l, p_l = X.shape
    print(f"  Shape après preprocessing : n={n_l}, p={p_l}")

    # ε_tv : 0 si on suppose X Gaussien (hypothèse théorique), KS sinon
    GAUSSIAN = True    # True → cas Gaussien (eps_tv=0), False → cas non-Gaussien (eps_tv via KS)
    if GAUSSIAN:
        eps_tv = 0.0
        print(f"  ε_tv = 0.0 (hypothèse Gaussienne)")
    else:
        ks_stats = np.array([ks_1samp(X[:, j], sp_norm.cdf).statistic for j in range(p_l)])
        eps_tv   = float(ks_stats.mean())
        print(f"  ε_tv (mean KS) = {eps_tv:.5f} | ε_tv_max = {ks_stats.max():.5f}")

    # ── 5. Variance V1 vs V2 ──────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  VARIANCE — B={B_VALUES}, {K_SEEDS} seeds")
    print(f"{'─'*55}")

    scores_all     = {est: {"v1": {B: [] for B in B_VALUES},
                            "v2": {B: [] for B in B_VALUES}} for est in BASE_ESTIMATORS}
    thresholds_all = {est: {"v1": {B: [] for B in B_VALUES},
                            "v2": {B: [] for B in B_VALUES}} for est in BASE_ESTIMATORS}

    for est_name, base_est in BASE_ESTIMATORS.items():
        print(f"\n  Estimateur : {est_name}")
        for seed in rng_seeds:
            for B in B_VALUES:
                for version, StablClass in [("v1", StablV1), ("v2", StablV2)]:
                    m = StablClass(
                        base_estimator=clone(base_est),
                        lambda_grid="auto", n_lambda=5,
                        n_bootstraps=B, artificial_type="knockoff",
                        n_jobs=N_JOBS, random_state=int(seed),
                    )
                    m.fit(X, y.values)
                    scores_all[est_name][version][B].append(m.get_importances())
                    thresholds_all[est_name][version][B].append(_t_star_constrained(m))

    rows_var = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for est_name in BASE_ESTIMATORS:
        var_mean = {"v1": [], "v2": []}
        var_std  = {"v1": [], "v2": []}
        for version in ["v1", "v2"]:
            for B in B_VALUES:
                S  = np.vstack(scores_all[est_name][version][B])
                vv = S.var(axis=0)
                var_mean[version].append(vv.mean())
                var_std[version].append(vv.std())

        print(f"\n  {est_name} | {'B':>6} | {'Var V1':>10} | {'Var V2':>10} | {'Réduction':>10}")
        for i, B in enumerate(B_VALUES):
            v1, v2 = var_mean["v1"][i], var_mean["v2"][i]
            red = (v1 - v2) / v1 * 100 if v1 > 0 else 0
            print(f"  {est_name} | {B:>6} | {v1:>10.6f} | {v2:>10.6f} | {red:>+9.1f}%")
            rows_var.append({"layer": layer_name, "estimateur": est_name, "B": B, "version": "v1",
                             "var_mean": v1, "var_std": var_std["v1"][i]})
            rows_var.append({"layer": layer_name, "estimateur": est_name, "B": B, "version": "v2",
                             "var_mean": v2, "var_std": var_std["v2"][i]})

        c = colors_est[est_name]
        for ax, version, ls in zip(axes, ["v1", "v2"], ["-", "--"]):
            m_arr = np.array(var_mean[version])
            s_arr = np.array(var_std[version])
            ax.plot(B_VALUES, m_arr, "o-", color=c, lw=2, ls=ls, label=est_name)
            ax.fill_between(B_VALUES, m_arr - s_arr, m_arr + s_arr, color=c, alpha=0.1)

    for ax, title in zip(axes, ["V1 — knockoffs fixes", "V2 — fresh knockoffs + LW"]):
        ax.set_xlabel("B"); ax.set_ylabel(f"Var(score_j) moyenne ({K_SEEDS} seeds)")
        ax.set_title(f"Var(score_j) vs B — {title}")
        ax.set_xscale("log"); ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"Var(score_j) vs B — {layer_name} — V1 vs V2", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_var / "variance_vs_B.pdf", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows_var).to_csv(out_var / "variance_vs_B.csv", index=False)

    # Jaccard inter-seeds
    B_fixed = max(B_VALUES)
    EST_REF = next(iter(BASE_ESTIMATORS))
    selections = {}
    for version in ["v1", "v2"]:
        for k, (sc, thr) in enumerate(zip(scores_all[EST_REF][version][B_fixed],
                                          thresholds_all[EST_REF][version][B_fixed])):
            selections[(version, k)] = (np.array(sc) >= thr).astype(int)

    # Heatmap features×seeds
    if K_SEEDS > 1:
        fig_h, axes_h = plt.subplots(1, 2, figsize=(14, max(3, K_SEEDS + 1)))
        for ax_h, version, color, vtitle in zip(
                axes_h, ["v1","v2"], ["#C41E3A","#001A7B"],
                ["V1 — knockoffs fixes","V2 — fresh knockoffs"]):
            sel_matrix = np.vstack([selections[(version, k)] for k in range(K_SEEDS)])
            ever = sel_matrix.any(axis=0)
            sm   = sel_matrix[:, ever]
            im = ax_h.imshow(sm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax_h.set_yticks(range(K_SEEDS))
            ax_h.set_yticklabels([f"seed {k+1}" for k in range(K_SEEDS)], fontsize=8)
            ax_h.set_xlabel(f"Features sélectionnées ≥1 fois ({ever.sum()} / {p_l})", fontsize=8)
            ax_h.set_title(vtitle, fontsize=8)
            ax_h.tick_params(bottom=False, labelbottom=False)
            plt.colorbar(im, ax=ax_h, ticks=[0,1], shrink=0.6, label="sélectionnée")
        fig_h.suptitle(f"Reproductibilité — {layer_name}, B={B_fixed}", fontsize=10)
        fig_h.tight_layout()
        fig_h.savefig(out_var / f"feature_stability_heatmap_B{B_fixed}.pdf", dpi=150)
        plt.close(fig_h)

    jac_records = []
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for i, (version, color) in enumerate([("v1","#C41E3A"), ("v2","#001A7B")]):
        sel_sets = [set(np.where(selections[(version, k)])[0]) for k in range(K_SEEDS)]
        jac_vals = [jac(sel_sets[a], sel_sets[b]) for a, b in _comb(range(K_SEEDS), 2)] if K_SEEDS > 1 else [1.0]
        jm, js   = float(np.mean(jac_vals)), float(np.std(jac_vals))
        print(f"  {version.upper()} Jaccard inter-seeds : {jm:.3f} ± {js:.3f}")
        jac_records.append({"layer": layer_name, "version": version, "B_fixed": B_fixed,
                             "jaccard_mean": round(jm,4), "jaccard_std": round(js,4)})
        ax2.bar(i, jm, yerr=js, color=color, alpha=0.8, capsize=6, width=0.5, label=version.upper())
    ax2.set_xticks([0,1]); ax2.set_xticklabels(["V1","V2"])
    ax2.set_ylabel("Jaccard inter-seeds"); ax2.set_ylim(0, 1.1)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax2.set_title(f"Reproductibilité — {layer_name}\nB={B_fixed}, {K_SEEDS} seeds")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(out_var / f"jaccard_inter_seeds_B{B_fixed}.pdf", dpi=150)
    plt.close(fig2)
    pd.DataFrame(jac_records).to_csv(out_var / "jaccard_inter_seeds.csv", index=False)
    print(f"  Variance → {out_var}/")

    # ── 6. FDP+ theory ────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  FDP+ THEORY — fit V2 par B")
    print(f"{'─'*55}")

    fdp_per_est_B = {}
    for est_name, base_est in BASE_ESTIMATORS.items():
        print(f"\n  Estimateur : {est_name}")
        for B in B_VALUES:
            m = StablV2(
                base_estimator=clone(base_est),
                lambda_grid="auto", n_lambda=5,
                n_bootstraps=B, artificial_type="knockoff",
                n_jobs=N_JOBS, random_state=RANDOM_STATE,
            )
            m.fit(X, y.values)
            scores = m.get_importances()
            # Recalcule FDP+ sur la grille des vrais scores pour avoir un η_t significatif
            m.fdr_threshold_range = np.sort(np.unique(scores))
            m._compute_FDPplus()
            K_lambda  = m.stabl_scores_.shape[1]
            sigma2    = m.score_variance_.max(axis=1)                  # (p,)
            p_        = sigma2.shape[0]
            ko_sigma2 = (m.ko_score_variance_.max(axis=1)              # (p,) feature-wise
                         if m.ko_score_variance_ is not None and m.ko_score_variance_.shape[0] == p_
                         else np.full(p_, 0.25))
            fdp_per_est_B[(est_name, B)] = {
                "scores":    scores,
                "K_lambda":  K_lambda,
                "sigma2":    sigma2,
                "ko_sigma2": ko_sigma2,
                "m":         m,
            }
            # diagnostics (t* indicatif via argmin FDP+, sans contrainte)
            t_diag = _t_star_constrained(m)
            D_diag = max(1, int(np.sum(scores > t_diag)))
            print(f"    B={B:5d} | t*_diag={t_diag:.4f} | D(t*_diag)={D_diag} | K={K_lambda}")

            scores_sorted = np.sort(scores)[::-1]
            print(f"\n    --- Diagnostic scores (top 30) ---")
            for rank, s in enumerate(scores_sorted[:30], 1):
                print(f"    {rank:>5}  {s:>10.4f}")
            fdrs        = np.array(m.FDRs_)
            thresh_grid = m.fdr_threshold_range
            idx_min     = np.argmin(fdrs)
            print(f"\n    FDRs autour du minimum :")
            for i in range(max(0, idx_min-3), min(len(fdrs), idx_min+4)):
                marker = " ← min" if i == idx_min else ""
                print(f"      thresh={thresh_grid[i]:.4f}  FDP+={fdrs[i]:.4f}{marker}")
            print(f"\n    Distribution des scores :")
            for label, lo, hi in [("<0.05",0,0.05),("[0.05,0.10)",0.05,0.10),
                                   ("[0.10,0.20)",0.10,0.20),("[0.20,0.50)",0.20,0.50),("≥0.50",0.50,1.01)]:
                print(f"      {label:>15} : {np.sum((scores>=lo)&(scores<hi)):>3} features")

    records_fdp = []
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for est_name in BASE_ESTIMATORS:
        c = colors_est[est_name]
        for delta in DELTA_VALUES:
            eps_pts       = []
            bern_pts      = []
            bdry_hoeff    = []
            bdry_bern_fw  = []
            for B in B_VALUES:
                d          = fdp_per_est_B[(est_name, B)]
                log_t      = np.log(4 * p_l * d["K_lambda"] / delta)
                eps_B      = float(np.sqrt(log_t / (2 * B)))
                sigma2_vec = d["sigma2"]                          # shape (p,)
                sigma2_max = float(sigma2_vec.max())
                disc       = (2/3)**2 * log_t**2 + 8 * B * log_t * sigma2_max
                eps_bern_unif = ((2/3) * log_t + np.sqrt(disc)) / (2 * B)

                # ── Minimisation sans contrainte (référence commune) ─────────
                t_star_unc, fdp_unc = _t_star_unconstrained(d["m"])
                D_unc = max(1, int(np.sum(d["scores"] > t_star_unc)))
                # termes résiduels au t* non-contraint, fenêtre Hoeffding et Bern FW
                bdry_unc_h   = int(np.sum((d["scores"] > t_star_unc) &
                                          (d["scores"] <= t_star_unc + 2*(eps_B + eps_tv))))
                residual_unc_h   = bdry_unc_h / D_unc

                # ── eps Bernstein FW (calculés une fois) ──────────────────────
                eps_fw_pre     = _bernstein_eps_featurewise(d["sigma2"], B, log_t)
                eps_ko_fw_pre  = _bernstein_eps_featurewise(d["ko_sigma2"], B, log_t)
                eps_tot_fw_pre = eps_fw_pre + eps_ko_fw_pre + eps_tv        # (p,)
                bdry_unc_bfw   = int(np.sum((d["scores"] > t_star_unc) &
                                            (d["scores"] <= t_star_unc + eps_tot_fw_pre)))
                residual_unc_bfw = bdry_unc_bfw / D_unc

                # ── Hoeffding contraint : argmin FDP+(t) s.t. bdry==0 ────────
                eps_total = eps_B + eps_tv
                t_star_h, bound_h, eta_t_h, exact_h = _t_star_opt_constrained(
                    d["m"], d["scores"], eps_B, eps_tv=eps_tv)
                D_h = max(1, int(np.sum(d["scores"] > t_star_h)))

                # ── Hoeffding composite (sans contrainte) : argmin FDP+(t) + bdry_H/D
                t_star_h_comp, obj_h_comp = _t_star_composite_hoeff(
                    d["m"], d["scores"], eps_B, eps_tv=eps_tv)
                D_h_comp = max(1, int(np.sum(d["scores"] > t_star_h_comp)))
                bdry_h_comp = int(np.sum((d["scores"] > t_star_h_comp) &
                                         (d["scores"] <= t_star_h_comp + 2*eps_total)))
                bdry_r_h_comp = bdry_h_comp / D_h_comp

                # ── Bernstein FW contraint : argmin FDP+(t) s.t. bdry==0 ──────
                t_star_bfw, bound_bfw, margin_bfw, exact_bfw = _t_star_opt_bernstein(
                    d["m"], d["scores"], sigma2_vec, B, log_t,
                    sigma2_ko_vec=d["ko_sigma2"], eps_tv=eps_tv)
                D_bfw     = max(1, int(np.sum(d["scores"] > t_star_bfw)))
                eps_fw    = eps_fw_pre
                eps_ko_fw = eps_ko_fw_pre
                eps_tot_fw = eps_tot_fw_pre

                # ── Bernstein FW composite (sans contrainte) : argmin FDP+(t) + bdry_BFW/D
                t_star_bfw_comp, obj_bfw_comp = _t_star_composite_bfw(
                    d["m"], d["scores"], sigma2_vec, B, log_t,
                    sigma2_ko_vec=d["ko_sigma2"], eps_tv=eps_tv)
                D_bfw_comp = max(1, int(np.sum(d["scores"] > t_star_bfw_comp)))
                bdry_bfw_comp = int(np.sum((d["scores"] > t_star_bfw_comp) &
                                           (d["scores"] <= t_star_bfw_comp + eps_tot_fw)))
                bdry_r_bfw_comp = bdry_bfw_comp / D_bfw_comp

                eps_fw_mean = float(eps_fw.mean())

                eps_pts.append(eps_B)
                bern_pts.append(float(eps_fw_mean))
                bdry_hoeff.append(residual_unc_h)
                bdry_bern_fw.append(residual_unc_bfw)

                records_fdp.append({
                    "layer": layer_name, "estimateur": est_name,
                    "delta": delta, "B": B,
                    "eps_B_hoeffding": round(eps_B, 5),
                    "eps_fw_mean":     round(eps_fw_mean, 5),
                    # Référence pure FDP+
                    "t_star_unc":          round(t_star_unc, 4),
                    "D_unc":               D_unc,
                    "fdp_unc":             round(fdp_unc, 4),
                    "residual_unc_hoeff":  round(residual_unc_h, 4),
                    "residual_unc_bfw":    round(residual_unc_bfw, 4),
                    # Hoeffding contraint (bdry==0)
                    "t_star_hoeffding":    round(t_star_h, 4),
                    "D_hoeffding":         D_h,
                    "fdp_hoeffding":       round(bound_h, 4),
                    "exact_hoeffding":     exact_h,
                    "D_loss_hoeff_vs_unc": D_h - D_unc,
                    # Hoeffding composite (sans contrainte)
                    "t_star_hoeff_comp":    round(t_star_h_comp, 4),
                    "D_hoeff_comp":         D_h_comp,
                    "obj_hoeff_comp":       round(obj_h_comp, 4),
                    "bdry_r_hoeff_comp":    round(bdry_r_h_comp, 4),
                    "D_diff_hoeff_comp_vs_constr": D_h_comp - D_h,
                    # Bernstein FW contraint (bdry==0)
                    "t_star_bern_fw":      round(t_star_bfw, 4),
                    "D_bern_fw":           D_bfw,
                    "fdp_bern_fw":         round(bound_bfw, 4),
                    "exact_bern_fw":       exact_bfw,
                    "min_margin_bern_fw":  round(margin_bfw, 5),
                    "D_loss_bern_vs_unc":  D_bfw - D_unc,
                    "D_gain_bern_vs_hoeff": D_bfw - D_h,
                    # Bernstein FW composite (sans contrainte)
                    "t_star_bfw_comp":     round(t_star_bfw_comp, 4),
                    "D_bfw_comp":          D_bfw_comp,
                    "obj_bfw_comp":        round(obj_bfw_comp, 4),
                    "bdry_r_bfw_comp":     round(bdry_r_bfw_comp, 4),
                    "D_diff_bfw_comp_vs_constr": D_bfw_comp - D_bfw,
                })
                if delta == DELTA_VALUES[0]:
                    print(f"      δ={delta} | ε_H={eps_B:.5f} | ε_fw̄={eps_fw_mean:.5f}")
                    print(f"      [Unc]        t*={t_star_unc:.4f} | D={D_unc} | FDP+={fdp_unc:.4f} | "
                          f"résidu_H={residual_unc_h:.4f} | résidu_BFW={residual_unc_bfw:.4f}")
                    print(f"      [Hoeff]    t*={t_star_h:.4f} | D={D_h} | FDP+={bound_h:.4f} | "
                          f"exact={exact_h} | ΔD={D_h - D_unc:+d}")
                    print(f"      [Bern FW]  t*={t_star_bfw:.4f} | D={D_bfw} | FDP+={bound_bfw:.4f} | "
                          f"exact={exact_bfw} | ΔD vs unc={D_bfw - D_unc:+d} | ΔD vs H={D_bfw - D_h:+d}")

            axes[0].plot(B_VALUES, eps_pts,  "o-", color=c, lw=1.5, ls=ls_delta[delta],
                         label=f"{est_name} Hoeffding" if delta==0.05 else None)
            axes[0].plot(B_VALUES, bern_pts, "s:", color=c, lw=1,
                         label=f"{est_name} Bern. uniform" if delta==0.05 else None)
            axes[1].plot(B_VALUES, bdry_hoeff,   "o-", color=c, lw=1.5, ls=ls_delta[delta],
                         label=f"{est_name} Hoeffding" if delta==0.05 else None)
            axes[2].plot(B_VALUES, bdry_bern_fw, "s-", color=c, lw=1.5, ls=ls_delta[delta],
                         label=f"{est_name} Bern. FW" if delta==0.05 else None)

    axes[0].set(xlabel="B", ylabel="ε", title="ε_H Hoeffding (trait) vs ε̄_fw Bernstein FW (pointillé)")
    axes[0].legend(fontsize=7); axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
    for ax, title in zip(axes[1:], [
            "résidu Hoeffding au t* unc : |∂+(t*,2ε_H)|/D",
            "résidu Bernstein FW au t* unc : |∂+(t*,ε_j+ε_ko_j)|/D"]):
        ax.axhline(0, color="black", lw=0.5)
        ax.set(xlabel="B", ylabel="résidu |∂+(t*_unc)|/D", title=title)
        ax.legend(fontsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"Garanties FDP+ — {layer_name} — p={p_l}, n={n_l}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fdp / "fdp_theory_curves.pdf", dpi=150)
    plt.close(fig)

    B_ref = B_VALUES[-1]
    delta_ref = DELTA_VALUES[0]
    fig2, axes2 = plt.subplots(1, len(BASE_ESTIMATORS), figsize=(14, 4))
    axes2 = np.atleast_1d(axes2)
    for ax2, est_name in zip(axes2, BASE_ESTIMATORS):
        d_ref      = fdp_per_est_B[(est_name, B_ref)]
        log_ref    = np.log(4 * p_l * d_ref["K_lambda"] / delta_ref)
        eps_ref    = float(np.sqrt(log_ref / (2 * B_ref)))
        sigma2_vec = d_ref["sigma2"]

        eps_fw    = _bernstein_eps_featurewise(sigma2_vec, B_ref, log_ref)
        eps_ko_fw = _bernstein_eps_featurewise(d_ref["ko_sigma2"], B_ref, log_ref)
        eps_tot_j = eps_fw + eps_ko_fw + eps_tv
        t_unc_ref, _ = _t_star_unconstrained(d_ref["m"])
        t_h, _, eta_h, exact_h = _t_star_opt_constrained(
            d_ref["m"], d_ref["scores"], eps_ref, eps_tv=eps_tv)
        t_bfw, _, margin_bfw, exact_bfw = _t_star_opt_bernstein(
            d_ref["m"], d_ref["scores"], sigma2_vec, B_ref, log_ref,
            sigma2_ko_vec=d_ref["ko_sigma2"], eps_tv=eps_tv)

        sc = d_ref["scores"]
        D_unc_ref     = max(1, int(np.sum(sc > t_unc_ref)))
        D_h_ref       = max(1, int(np.sum(sc > t_h)))
        D_bfw_ref     = max(1, int(np.sum(sc > t_bfw)))
        eps_tot_h_ref = eps_ref + eps_tv
        res_unc_h_ref   = int(np.sum((sc > t_unc_ref) & (sc <= t_unc_ref + 2*eps_tot_h_ref))) / D_unc_ref
        res_unc_bfw_ref = int(np.sum((sc > t_unc_ref) & (sc <= t_unc_ref + eps_tot_j))) / D_unc_ref

        ax2.hist(sc, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
        ax2.axvline(t_unc_ref, color="gray", lw=1.5, ls=":",
                    label=f"t* unc={t_unc_ref:.3f} (D={D_unc_ref}, résidu H={res_unc_h_ref:.3f}, BFW={res_unc_bfw_ref:.3f})")
        ax2.axvline(t_h,       color="#C41E3A", lw=2, ls="--",
                    label=f"t* Hoeff contraint={t_h:.3f} ({'✓' if exact_h else '✗'}, D={D_h_ref})")
        ax2.axvline(t_bfw,     color="#2CA02C", lw=2, ls="-.",
                    label=f"t* BernFW contraint={t_bfw:.3f} ({'✓' if exact_bfw else '✗'}, D={D_bfw_ref})")
        ax2.axvspan(t_h, t_h + 2*eps_tot_h_ref, alpha=0.12, color="#C41E3A",
                    label=f"barrière Hoeffding (2ε={2*eps_tot_h_ref:.4f})")
        eps_fw_above = eps_tot_j[sc > t_bfw]
        bfw_zone = float(eps_fw_above.max()) if len(eps_fw_above) > 0 else 0.0
        ax2.axvspan(t_bfw, t_bfw + bfw_zone, alpha=0.10, color="#2CA02C",
                    label=f"barrière BernFW (max ε_tot={bfw_zone:.4f})")

        ax2.set(xlabel="Max stability score", ylabel="Nb features",
                title=(f"{est_name} (B={B_ref}, δ={delta_ref})\n"
                       f"unc D={D_unc_ref} | Hoeff D={D_h_ref} (Δ={D_h_ref-D_unc_ref:+d}) | "
                       f"BernFW D={D_bfw_ref} (Δ={D_bfw_ref-D_unc_ref:+d})"))
        ax2.legend(fontsize=7)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig2.suptitle(f"Distribution des scores — {layer_name}\nHoeffding vs Bernstein FW", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(out_fdp / "score_distribution.pdf", dpi=150)
    plt.close(fig2)
    pd.DataFrame(records_fdp).to_csv(out_fdp / "fdp_theory_table.csv", index=False)

    # ── Graphique décomposé : FDP+(t) / D(t) / |∂+|/D(t) / objectif vs t ──────
    B_ref_obj     = B_VALUES[-1]
    delta_ref_obj = DELTA_VALUES[0]
    n_est  = len(BASE_ESTIMATORS)
    fig3, axes3 = plt.subplots(4, n_est,
                               figsize=(max(6, 5 * n_est), 14),
                               sharex="col")
    axes3 = np.array(axes3).reshape(4, n_est)   # shape (4, n_est) garanti

    _ROW_LABELS = ["FDP⁺(t)", "D(t)", "|∂⁺(t, 2ε)| / D(t)",
                   "FDP⁺(t)  +  |∂⁺| / D(t)  [objectif]"]
    _ROW_COLORS = ["#4A90D9", "#2CA02C", "#FF7F0E", "steelblue"]

    for col, est_name in enumerate(BASE_ESTIMATORS):
        d_o      = fdp_per_est_B[(est_name, B_ref_obj)]
        log_o    = np.log(4 * p_l * d_o["K_lambda"] / delta_ref_obj)
        eps_o    = float(np.sqrt(log_o / (2 * B_ref_obj)))
        eps_tot  = eps_o + eps_tv
        thresh   = np.array(d_o["m"].fdr_threshold_range)
        fdrs_arr = np.array(d_o["m"].FDRs_)
        sc       = d_o["scores"]

        D_t    = np.array([int(np.sum(sc > t)) for t in thresh])          # D(t) brut
        bdry_t = np.array([int(np.sum((sc > t) & (sc <= t + 2*eps_tot)))
                           for t in thresh])                               # |∂+|(t)
        bdry_D = bdry_t / np.maximum(D_t, 1)                              # |∂+|/D(t)
        objs   = fdrs_arr + bdry_D                                         # objectif

        feasible = bdry_t == 0   # ∂+(t, Hoeffding) = ∅

        sigma2_o     = d_o["sigma2"]
        eps_fw_o     = _bernstein_eps_featurewise(sigma2_o, B_ref_obj, log_o)
        eps_ko_fw_o  = _bernstein_eps_featurewise(d_o["ko_sigma2"], B_ref_obj, log_o)
        eps_tot_fw_o = eps_fw_o + eps_ko_fw_o + eps_tv

        # Courbe de barrière Bernstein FW : |∂+(t, ε_j+ε_ko_j)| / D(t)
        bdry_bfw_t = np.array([int(np.sum((sc > t) & (sc <= t + eps_tot_fw_o))) for t in thresh])
        bdry_bfw_D = bdry_bfw_t / np.maximum(D_t, 1)

        t_star_unc_o, _ = _t_star_unconstrained(d_o["m"])
        t_star_o, _, _, exact_o = _t_star_opt_constrained(
            d_o["m"], sc, eps_o, eps_tv=eps_tv)
        t_star_bfw_o, _, margin_bfw_o, exact_bfw_o = _t_star_opt_bernstein(
            d_o["m"], sc, sigma2_o, B_ref_obj, log_o,
            sigma2_ko_vec=d_o["ko_sigma2"], eps_tv=eps_tv)
        t_star_h_comp_o, _ = _t_star_composite_hoeff(
            d_o["m"], sc, eps_o, eps_tv=eps_tv)
        t_star_bfw_comp_o, _ = _t_star_composite_bfw(
            d_o["m"], sc, sigma2_o, B_ref_obj, log_o,
            sigma2_ko_vec=d_o["ko_sigma2"], eps_tv=eps_tv)
        # Bernstein FW feasibility per threshold
        feasible_bfw = np.array([
            np.sum((sc > t) & (sc <= t + eps_tot_fw_o)) == 0
            for t in thresh
        ])

        # Valeurs des barrières au t* unconstrained
        matches = np.where(thresh == t_star_unc_o)[0]
        idx_unc = int(matches[0]) if len(matches) > 0 else int(np.argmin(np.abs(thresh - t_star_unc_o)))
        res_h_unc   = float(bdry_D[idx_unc])
        res_bfw_unc = float(bdry_bfw_D[idx_unc])

        curves = [fdrs_arr, D_t.astype(float), bdry_D, objs]

        for row, (curve, color, ylabel) in enumerate(
                zip(curves, _ROW_COLORS, _ROW_LABELS)):
            ax = axes3[row, col]
            ax.step(thresh, curve, where="post", color=color, lw=1.5)

            # zones faisables : axvspan par threshold faisable, largeur = fenêtre ε
            ylo = curve.min(); yhi = curve.max()
            pad = (yhi - ylo) * 0.12 if yhi > ylo else 0.05
            ax.set_ylim(ylo - pad, yhi + pad)
            _lbl_h   = "faisable Hoeffding" if row == 0 else None
            _lbl_bfw = "faisable Bern FW"   if row == 0 else None
            for t_f in thresh[feasible]:
                ax.axvspan(t_f, t_f + 2 * eps_tot, alpha=0.18, color="#4A90D9",
                           label=_lbl_h, zorder=0)
                _lbl_h = None  # n'étiqueter qu'une fois
                ax.axvline(t_f, ymin=0, ymax=0.08, color="#4A90D9", lw=1.2, alpha=0.9)
            bfw_win = float(eps_tot_fw_o.max())
            for t_f in thresh[feasible_bfw]:
                ax.axvspan(t_f, t_f + bfw_win, alpha=0.18, color="#2CA02C",
                           label=_lbl_bfw, zorder=0)
                _lbl_bfw = None
                ax.axvline(t_f, ymin=0, ymax=0.08, color="#2CA02C", lw=1.2, alpha=0.9)

            ax.axvline(t_star_unc_o,      color="gray",    lw=1,   ls=":",
                       label=f"t* unc={t_star_unc_o:.3f}" if row == 0 else None)
            ax.axvline(t_star_o,          color="#C41E3A", lw=1.5, ls="--",
                       label=f"t* Hoeff contraint={t_star_o:.3f} ({'✓' if exact_o else '✗'})" if row == 0 else None)
            ax.axvline(t_star_bfw_o,      color="#2CA02C", lw=1.5, ls="-.",
                       label=f"t* BernFW contraint={t_star_bfw_o:.3f} ({'✓' if exact_bfw_o else '✗'})" if row == 0 else None)
            ax.axvline(t_star_h_comp_o,   color="#4A90D9", lw=1.2, ls="--",
                       label=f"t* Hoeff composite={t_star_h_comp_o:.3f}" if row == 0 else None)
            ax.axvline(t_star_bfw_comp_o, color="#85C785", lw=1.2, ls="--",
                       label=f"t* BernFW composite={t_star_bfw_comp_o:.3f}" if row == 0 else None)

            # Panneau 2 : superposer la courbe BernFW + annoter les deux barrières au t* unc
            if row == 2:
                ax.step(thresh, bdry_bfw_D, where="post",
                        color="#2CA02C", lw=1.2, ls="--", alpha=0.85,
                        label="|∂⁺|/D BernFW")
                ax.plot(t_star_unc_o, res_h_unc,   "o", color="#4A90D9", ms=7, zorder=5,
                        label=f"Hoeff au t*_unc : {res_h_unc:.3f}")
                ax.plot(t_star_unc_o, res_bfw_unc, "s", color="#2CA02C", ms=7, zorder=5,
                        label=f"BernFW au t*_unc : {res_bfw_unc:.3f}")
                ax.legend(fontsize=6, loc="upper right")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                eps_fw_mean_o = float(eps_fw_o.mean())
                ax.set_title(f"{est_name}\nB={B_ref_obj}, δ={delta_ref_obj}, "
                             f"ε_H={eps_o:.4f} | ε_fw̄={eps_fw_mean_o:.4f}",
                             fontsize=8)
                ax.legend(fontsize=7, loc="upper right")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row == 3:
                ax.set_xlabel("t", fontsize=8)

    fig3.suptitle(
        f"Décomposition FDP⁺ / D(t) / |∂⁺| — {layer_name} — ε_tv={eps_tv:.5f}  "
        f"(bleu clair = faisable Hoeffding, vert = faisable BernFW | "
        f"rouge-- = Hoeff contraint, vert-. = BernFW contraint, "
        f"bleu-- = Hoeff composite, vert clair-- = BernFW composite)",
        fontsize=8)
    fig3.tight_layout()
    fig3.savefig(out_fdp / "objective_function.pdf", dpi=150)
    plt.close(fig3)
    print(f"  FDP+ theory → {out_fdp}/")

    # ── 7. Gaussianité ────────────────────────────────────────────────────────
    rows_gauss = []
    for est_name in BASE_ESTIMATORS:
        for B in B_VALUES:
            d = fdp_per_est_B[(est_name, B)]
            for delta in DELTA_VALUES:
                log_t     = np.log(4 * p_l * d["K_lambda"] / delta)
                eps_B     = float(np.sqrt(log_t / (2 * B)))
                eps_total = eps_B + eps_tv
                _, _, eta_t, exact = _t_star_opt_constrained(
                    d["m"], d["scores"], eps_B, eps_tv=eps_tv)
                cond  = exact
                rows_gauss.append({
                    "layer": layer_name, "estimateur": est_name, "B": B, "delta": delta,
                    "eta_t": round(eta_t, 5), "eps_B": round(eps_B, 5),
                    "eps_tv": round(eps_tv, 5), "eps_total": round(eps_total, 5),
                    "condition (2*(eps_B+eps_tv) < eta_t)": cond,
                    "marge": round(eta_t - 2*eps_total, 5),
                })
            print(f"  {est_name} B={B:5d} | η_t={eta_t:.5f} | ε_B={eps_B:.5f} | "
                  f"ε_tv={eps_tv:.5f} | 2ε_total={2*eps_total:.5f} | condition : {cond}")

    df_gauss = pd.DataFrame(rows_gauss)
    df_gauss.to_csv(out_gauss / "gaussianity_summary.csv", index=False)

    fig, axes = plt.subplots(1, len(BASE_ESTIMATORS), figsize=(14, 4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, est_name in zip(axes, BASE_ESTIMATORS):
        for delta in DELTA_VALUES:
            sub = df_gauss[(df_gauss["estimateur"]==est_name) & (df_gauss["delta"]==delta)]
            ax.plot(sub["B"], sub["eps_total"], marker="o", lw=2,
                    ls=ls_delta[delta], color=c_delta[delta], label=f"δ={delta} ε_B+ε_tv")
            ax.plot(sub["B"], sub["eta_t"] / 2, marker="s", lw=1.5,
                    ls=ls_delta[delta], color=c_delta[delta], alpha=0.5,
                    label=f"δ={delta} η_t/2")
        ax.set_xlabel("B"); ax.set_ylabel("valeur")
        ax.set_title(f"{est_name} — {layer_name}\n(garantie si ε_B+ε_tv < η_t/2)")
        ax.set_xscale("log"); ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"ε_B vs η_t/2 — {layer_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_gauss / "gaussianity.pdf", dpi=150)
    plt.close(fig)
    print(f"  Gaussianité → {out_gauss}/")

    # ── Accumuler pour synthesis ───────────────────────────────────────────────
    var_lookup = {(r["estimateur"], r["B"], r["version"]): (r["var_mean"], r["var_std"])
                  for r in rows_var}
    for row in records_fdp:
        est, B = row["estimateur"], row["B"]
        v1m, v1s = var_lookup.get((est, B, "v1"), (np.nan, np.nan))
        v2m, v2s = var_lookup.get((est, B, "v2"), (np.nan, np.nan))
        all_synthesis.append({**row,
            "var_v1_mean": round(v1m,7) if not np.isnan(v1m) else np.nan,
            "var_v1_std":  round(v1s,7) if not np.isnan(v1s) else np.nan,
            "var_v2_mean": round(v2m,7) if not np.isnan(v2m) else np.nan,
            "var_v2_std":  round(v2s,7) if not np.isnan(v2s) else np.nan,
        })

# ── 8. CSV unifié global ──────────────────────────────────────────────────────
out_synthesis = _subdir("synthesis")
df_unified = pd.DataFrame(all_synthesis)
df_unified.to_csv(out_synthesis / "theory_analysis.csv", index=False)
print(f"\nCSV unifié → {out_synthesis}/theory_analysis.csv")
print(df_unified.to_string(index=False))

# ── 9. Corrélation IMC × CyTOF (post-CV uniquement) ─────────────────────────

if RUN_CV:
    from correlation_IMC_CyTOF import (
        _load_imc, _load_cytof, _get_features_for_model,
        _spearman_matrix, _fdr_correct, _build_ranking,
        plot_heatmap, plot_top_pairs_barplot, plot_network,
        FDR_THRESHOLD,
    )

    CORR_OUT = OUT_DIR / "correlation_results_tests"
    CORR_OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("CORRÉLATION IMC × CyTOF")
    print(f"{'='*65}")

    X_imc_full = _load_imc(DATA_DIR)
    cytof_dict = _load_cytof()

    if not cytof_dict:
        print("  Données CyTOF introuvables — corrélation ignorée.")
    else:
        cytof_all  = pd.concat(cytof_dict.values(), axis=1)
        common_pts = X_imc_full.index.intersection(cytof_all.index)
        print(f"  IMC : {X_imc_full.shape}  |  CyTOF : {cytof_all.shape[1]} features  |  Patients communs : {len(common_pts)}")

        summary_corr = []
        for version in [1, 2]:
            run_name  = f"penalized_v{version}"
            cv_dir    = OUT_DIR / run_name / "Training CV"
            if not cv_dir.exists():
                print(f"\n  {run_name} — pas de résultats CV, ignoré.")
                continue
            print(f"\n  {run_name.upper()}")
            for model in ALL_MODELS:
                feat_df   = _get_features_for_model(cv_dir, model)
                if feat_df.empty:
                    print(f"    [{model}] Aucune feature sélectionnée, ignoré.")
                    continue
                imc_feats = [f for f in feat_df["feature"] if f in X_imc_full.columns]
                if not imc_feats:
                    print(f"    [{model}] Features introuvables dans l'IMC, ignoré.")
                    continue

                X_imc   = X_imc_full.loc[common_pts, imc_feats]
                X_cytof = cytof_all.loc[common_pts]

                df_r, df_p = _spearman_matrix(X_imc, X_cytof)
                df_fdr     = _fdr_correct(df_p)
                df_ranking = _build_ranking(df_r, df_p, df_fdr, cytof_dict)

                out_dir = CORR_OUT / run_name / model.replace(" ", "_")
                out_dir.mkdir(parents=True, exist_ok=True)
                df_ranking.to_csv(out_dir / "ranking.csv", index=False)

                n_sig = int(df_ranking["significant"].sum())
                top_r = float(df_ranking["abs_r"].iloc[0]) if not df_ranking.empty else 0.0
                print(f"    [{model}] {len(imc_feats)} features IMC, {len(df_ranking)} paires, "
                      f"{n_sig} FDR<{FDR_THRESHOLD}, top|r|={top_r:.3f}")

                plot_top_pairs_barplot(df_ranking, out_dir / "top_pairs_barplot.pdf", model, run_name)
                for cytof_name, df_layer in cytof_dict.items():
                    cols = [c for c in df_layer.columns if c in df_r.columns]
                    if cols:
                        plot_heatmap(df_r[cols], df_fdr[cols], cytof_name,
                                     out_dir / f"heatmap_{cytof_name}.pdf",
                                     len(common_pts), model, run_name)
                plot_network(df_ranking, out_dir / "network.pdf", model, run_name)

                summary_corr.append({
                    "version":          run_name,
                    "model":            model,
                    "n_imc_features":   len(imc_feats),
                    "n_pairs":          len(df_ranking),
                    "n_significant":    n_sig,
                    "top_abs_r":        round(top_r, 4),
                    "top_imc_feature":  df_ranking["imc_feature"].iloc[0]   if not df_ranking.empty else "",
                    "top_cytof_feature":df_ranking["cytof_feature"].iloc[0] if not df_ranking.empty else "",
                    "top_cytof_source": df_ranking["cytof_source"].iloc[0]  if not df_ranking.empty else "",
                })

        if summary_corr:
            df_corr_summary = pd.DataFrame(summary_corr).sort_values("top_abs_r", ascending=False)
            df_corr_summary.to_csv(CORR_OUT / "summary.csv", index=False)
            print(f"\n  Summary → {CORR_OUT}/summary.csv")
        print(f"  Corrélation → {CORR_OUT}/")

# ── Résumé final ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("RÉSULTATS SAUVEGARDÉS")
print(f"{'='*65}")
for path in sorted((OUT_DIR / "post_processing").rglob("*.pdf")):
    print(f"  {path.relative_to(OUT_DIR)}")
print(f"\nDone.")
