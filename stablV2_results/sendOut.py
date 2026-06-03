"""
sendOut.py — Synthetic StablV2 benchmark
=========================================
Compare StablV1 vs StablV2 (unconstrained FDP+) vs StablV2 (BernFW constrained)
on 5 synthetic datasets with known ground truth.

Modes:
  0 : run_experiment(idx)   — CV for run index idx
  1 : post_process()        — ROC, AUC, feature recovery plots
  2 : run_theory_analysis() — FDP+, BernFW boundary (no Hoeffding)
  3 : parse_params()        — generate SLURM scripts
"""

import os
import sys
import shutil
import json
import argparse
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).parent.parent))  # stablVMax/

# ── SLURM templates ────────────────────────────────────────────────────────────

_arrayTemplate = Template("""#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --error=./logs/${name}_%a.err
#SBATCH --output=./logs/${name}_%a.out
#SBATCH --array=0-${rep}
#SBATCH --time=${time}
#SBATCH -p normal
#SBATCH -c ${cpu}
#SBATCH --mem=${mem}GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${email}

ml python/3.12.1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=$$SLURM_CPUS_PER_TASK

source ${venv}/bin/activate
cd ${workdir}

time python3 ./sendOut.py 0 $${SLURM_ARRAY_TASK_ID}
""")

_endTemplate = Template("""#!/bin/bash
#SBATCH --job-name=${name}_end
#SBATCH --error=./logs/${name}_end.err
#SBATCH --output=./logs/${name}_end.out
#SBATCH --time=24:00:00
#SBATCH -p normal
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${email}

ml python/3.12.1

source ${venv}/bin/activate
cd ${workdir}

echo "=== Post-processing ==="
time python3 ./sendOut.py 1

echo "=== Theory analysis ==="
time python3 ./sendOut.py 2
""")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_json(path):
    with open(path) as f:
        return json.load(f)

def _write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def _expand_runs(params):
    """Generate flat list of (dataset, version) run configs from params."""
    runs = []
    for ds in params["datasets"]:
        for version in params["versions"]:
            name = f"dataset_{ds['id']}_{version}"
            runs.append({
                "name":    name,
                "version": version,
                "dataset": ds,
                "save_path": f"results/dataset_{ds['id']}/{version}",
            })
    return runs


# ── parse_params ───────────────────────────────────────────────────────────────

def parse_params(params_path="./params.json"):
    params  = _read_json(params_path)
    g       = params["general"]
    name    = params["Experiment_Name"].replace(" ", "_")
    email   = params.get("email", "")
    venv    = params.get("venv",    "$HOME/stablVMax/.venv")
    workdir = params.get("workdir", "$HOME/stablVMax/stablV2_results")
    runs    = _expand_runs(params)

    os.makedirs("./temp/",    exist_ok=True)
    os.makedirs("./logs/",    exist_ok=True)
    os.makedirs("./results/", exist_ok=True)

    for idx, run in enumerate(runs):
        run_dir = Path(f"./results/run_{idx}")
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json({**run, **g, "idx": idx}, run_dir / "params.json")

    script = _arrayTemplate.substitute(
        name=name, rep=len(runs)-1,
        cpu=g["cpusHigh"], mem=g["memHighGB"], time=g["time"],
        email=email, venv=venv, workdir=workdir
    )
    with open("./temp/array.sh", "w") as f:
        f.write(script)

    end_script = _endTemplate.substitute(
        name=name, email=email, venv=venv, workdir=workdir
    )
    with open("./temp/end.sh", "w") as f:
        f.write(end_script)

    # Script de lancement avec dependency automatique
    launch = (
        "#!/bin/bash\n"
        "set -e\n"
        f"ARRAY_ID=$(sbatch --parsable temp/array.sh)\n"
        f'echo "Array job submitted: $ARRAY_ID"\n'
        f"sbatch --dependency=afterok:$ARRAY_ID temp/end.sh\n"
        f'echo "End job submitted (runs after $ARRAY_ID completes)"\n'
    )
    with open("./temp/launch.sh", "w") as f:
        f.write(launch)

    print(f"parse_params: {len(runs)} runs ({len(params['datasets'])} datasets × "
          f"{len(params['versions'])} versions) — scripts written to ./temp/")
    print(f"  → Sur Sherlock : bash temp/launch.sh")


# ── Data generation ────────────────────────────────────────────────────────────

def generate_dataset(n, p, k, signal, seed):
    """
    Synthetic binary classification dataset.
    - Block correlation structure (blocks of 10, within-block rho=0.5)
    - First k features carry signal (alternating ±signal)
    - Classes balanced at median logit
    Returns: X (DataFrame, n×p), y (Series, n), true_features (list), Sigma_true (p×p ndarray)
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    block_size = 10

    # Matrice de covariance exacte : block-diagonale, rho=0.5 intra-bloc
    Sigma_true = np.zeros((p, p))
    n_full_blocks = p // block_size
    for b in range(n_full_blocks):
        i0, i1 = b * block_size, (b + 1) * block_size
        Sigma_true[i0:i1, i0:i1] = 0.5
        np.fill_diagonal(Sigma_true[i0:i1, i0:i1], 1.0)
    remainder = p % block_size
    if remainder > 0:
        i0 = n_full_blocks * block_size
        np.fill_diagonal(Sigma_true[i0:i0+remainder, i0:i0+remainder], 1.0)

    blocks = []
    for b in range(n_full_blocks):
        cov = 0.5 * np.ones((block_size, block_size))
        np.fill_diagonal(cov, 1.0)
        blocks.append(rng.multivariate_normal(np.zeros(block_size), cov, n))
    if remainder > 0:
        blocks.append(rng.standard_normal((n, remainder)))
    X = np.hstack(blocks)

    true_idx = np.arange(k)
    coefs = np.zeros(p)
    coefs[true_idx] = signal * np.array([1 if i % 2 == 0 else -1 for i in range(k)])

    logits = X @ coefs
    y = (logits > np.median(logits)).astype(int)

    feat_names    = [f"feat_{i:04d}" for i in range(p)]
    true_features = [f"feat_{i:04d}" for i in true_idx]

    X_df = pd.DataFrame(X, columns=feat_names)
    y_s  = pd.Series(y, name="outcome")

    return X_df, y_s, true_features, Sigma_true


# ── ML imports ─────────────────────────────────────────────────────────────────

def _import_ml():
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn import clone
    from stabl.stabl  import Stabl as StablV1
    from stabl.stablV2 import Stabl as StablV2
    from stabl.adaptive import ALasso, ALogitLasso
    from stabl.multi_omic_pipelines import multi_omic_stabl_cv
    from xgboost import XGBClassifier
    return (np, pd, ElasticNet, LogisticRegression, GridSearchCV,
            RepeatedStratifiedKFold, StratifiedKFold, clone,
            StablV1, StablV2, ALasso, ALogitLasso, multi_omic_stabl_cv,
            Pipeline, StandardScaler, XGBClassifier)


def _get_stabl_class(version, StablV1, StablV2):
    return StablV1 if version == "v1" else StablV2

def _stabl_kwargs(version):
    if version == "v1":      return {}
    if version == "v2_unc":  return {"selection_mode": "unconstrained"}
    if version == "v2_constr": return {"selection_mode": "constrained"}
    return {}

def _make_estimators(p, np, GridSearchCV, LogisticRegression, ElasticNet,
                     clone, StablClass, stabl_kw, ALasso, ALogitLasso,
                     XGBClassifier, StratifiedKFold, version):
    C_grid   = np.logspace(-2, 0, 10)
    nb       = p["n_bootstraps"]
    inner_cv = StratifiedKFold(n_splits=min(3, p["n_splits"]),
                               shuffle=True, random_state=p["random_state"])
    n_jobs   = p["n_jobs"]

    lasso_cv = GridSearchCV(
        LogisticRegression(penalty="l1", solver="liblinear",
                           class_weight="balanced", max_iter=int(1e6)),
        {"C": C_grid}, cv=inner_cv, scoring="roc_auc", n_jobs=n_jobs)
    alasso_cv = GridSearchCV(
        ALogitLasso(solver="liblinear"),
        {"C": C_grid}, cv=inner_cv, scoring="roc_auc", n_jobs=n_jobs)
    en_cv = GridSearchCV(
        LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga",
                           class_weight="balanced", max_iter=int(1e6)),
        {"C": C_grid}, cv=inner_cv, scoring="roc_auc", n_jobs=n_jobs)
    xgb_cv = GridSearchCV(
        XGBClassifier(n_jobs=1, eval_metric="logloss",
                      verbosity=0, random_state=p["random_state"]),
        {"max_depth": [3, 5], "n_estimators": [50, 100, 200]},
        cv=inner_cv, scoring="roc_auc", n_jobs=n_jobs)

    # multi_omic_stabl_cv requires "lasso"/"alasso"/"en" keys to always exist
    return {
        "lasso":        clone(lasso_cv),
        "alasso":       clone(alasso_cv),
        "en":           clone(en_cv),
        "xgboost":      clone(xgb_cv),
        "stabl_lasso":  StablClass(
            base_estimator=LogisticRegression(penalty="l1", solver="liblinear",
                                              class_weight="balanced", max_iter=int(1e6)),
            lambda_grid="auto", artificial_type="knockoff", n_bootstraps=nb, **stabl_kw),
        "stabl_alasso": StablClass(
            base_estimator=ALasso(tol=1e-3),
            lambda_grid="auto", artificial_type="knockoff", n_bootstraps=nb, **stabl_kw),
        "stabl_en":     StablClass(
            base_estimator=ElasticNet(l1_ratio=0.5, tol=1e-3),
            lambda_grid="auto", artificial_type="knockoff", n_bootstraps=nb, **stabl_kw),
    }


# ── Mode 0 : run one experiment ────────────────────────────────────────────────

def run_experiment(idx):
    (np, pd, ElasticNet, LogisticRegression, GridSearchCV,
     RepeatedStratifiedKFold, StratifiedKFold, clone,
     StablV1, StablV2, ALasso, ALogitLasso, multi_omic_stabl_cv,
     Pipeline, StandardScaler, XGBClassifier) = _import_ml()

    p       = _read_json(Path(f"./results/run_{idx}/params.json"))
    version = p["version"]
    ds      = p["dataset"]
    print(f"Run {idx}: dataset_{ds['id']} | version={version} | "
          f"n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']}")

    StablClass = _get_stabl_class(version, StablV1, StablV2)
    stabl_kw   = _stabl_kwargs(version)

    # Patch save_stabl_results
    import stabl.multi_omic_pipelines as _mop_mod
    if version != "v1":
        from stabl.stablV2 import save_stabl_results as _ssr
    else:
        from stabl.stabl import save_stabl_results as _ssr
    _mop_mod.save_stabl_results = _ssr

    X_df, y, true_features, _ = generate_dataset(
        ds["n"], ds["p"], ds["k"], ds["signal"], ds["seed"])

    save_path = Path(p["save_path"])
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save ground truth
    _write_json({"true_features": true_features,
                 "n": ds["n"], "p": ds["p"], "k": ds["k"],
                 "signal": ds["signal"], "seed": ds["seed"]},
                save_path / "ground_truth.json")

    outer_splitter = RepeatedStratifiedKFold(
        n_splits=p["n_splits"], n_repeats=p["n_repeats"],
        random_state=p["random_state"])

    estimators = _make_estimators(
        p, np, GridSearchCV, LogisticRegression, ElasticNet,
        clone, StablClass, stabl_kw, ALasso, ALogitLasso,
        XGBClassifier, StratifiedKFold, version)

    if version == "v1":
        models_to_run = ["STABL Lasso", "STABL ALasso", "STABL ElasticNet",
                         "Lasso", "ALasso", "ElasticNet", "XGBoost"]
    else:
        models_to_run = ["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]

    preprocessing = Pipeline([("std", StandardScaler())])

    multi_omic_stabl_cv(
        data_dict={"Synthetic": X_df},
        y=y,
        outer_splitter=outer_splitter,
        estimators=estimators,
        task_type=p["taskType"],
        save_path=str(save_path),
        models=models_to_run,
        outer_groups=None,
        early_fusion=False, late_fusion=False, n_iter_lf=10000,
        preprocessing_overrides={"Synthetic": preprocessing},
    )
    print(f"Run {idx} done → {save_path}/")


# ── Mode 1 : post-processing ───────────────────────────────────────────────────

def post_process(params_path="./params.json"):
    import numpy as np
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sk_auc

    params = _read_json(params_path)
    runs   = _expand_runs(params)

    stabl_models = ["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]
    base_models  = ["Lasso", "ALasso", "ElasticNet", "XGBoost"]
    all_models   = stabl_models + base_models

    palette = {"v1": "#C41E3A", "v2_unc": "#2CA02C", "v2_constr": "#001A7B"}

    out = Path("post_processing")
    for sub in ["ROC", "AUC", "feature_recovery"]:
        d = out / sub
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # ── Load predictions and AUC progress ────────────────────────────────────
    auc_data  = {}   # (dataset_id, version, model) -> list of AUC per fold
    pred_data = {}   # (dataset_id, version, model) -> DataFrame
    feat_data = {}   # (dataset_id, version, model) -> set of selected features
    gt_data   = {}   # dataset_id -> list of true features

    for run in runs:
        ds_id   = run["dataset"]["id"]
        version = run["version"]
        sp      = Path(run["save_path"])

        if ds_id not in gt_data:
            gt_path = sp / "ground_truth.json"
            if gt_path.exists():
                gt_data[ds_id] = _read_json(gt_path)["true_features"]

        auc_path = sp / "Training CV" / "auc_progress.csv"
        if auc_path.exists():
            df_auc = pd.read_csv(auc_path, index_col=0)
            for m in all_models:
                if m in df_auc.columns:
                    auc_data[(ds_id, version, m)] = df_auc[m].dropna().tolist()

        for m in all_models:
            pred_path = sp / "Training CV" / m / f"{m} predictions.csv"
            if pred_path.exists():
                pred_data[(ds_id, version, m)] = pd.read_csv(pred_path, index_col=0)

            feat_path = sp / "Training CV" / f"Selected Features {m}.csv"
            if feat_path.exists():
                df_f = pd.read_csv(feat_path)
                feat_data[(ds_id, version, m)] = set(df_f.iloc[:, 0].tolist())
            else:
                feat_data[(ds_id, version, m)] = set()

    datasets = params["datasets"]
    versions = params["versions"]

    # ── ROC curves — one figure per dataset per model ─────────────────────────
    for ds in datasets:
        ds_id = ds["id"]
        for m in all_models:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
            any_curve = False
            for version in versions:
                key = (ds_id, version, m)
                if key not in pred_data: continue
                df_p = pred_data[key]
                score_col = [c for c in df_p.columns if c not in ("Patient", "outcome")][0]
                fpr, tpr, _ = roc_curve(df_p["outcome"], df_p[score_col])
                roc_auc = sk_auc(fpr, tpr)
                ax.plot(fpr, tpr, color=palette.get(version, "gray"), lw=2,
                        label=f"{version}  AUC={roc_auc:.3f}")
                any_curve = True
            if not any_curve:
                plt.close(fig); continue
            ax.set(xlabel="FPR", ylabel="TPR",
                   title=f"ROC — {m}\ndataset {ds_id} (n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']})")
            ax.legend(loc="lower right", fontsize=8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(out / "ROC" / f"ROC_ds{ds_id}_{m.replace(' ','_')}.pdf", dpi=150)
            plt.close(fig)
    print("ROC curves saved → post_processing/ROC/")

    # ── AUC boxplots — per model, one box per (dataset, version) ─────────────
    for m in all_models:
        fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, datasets):
            ds_id = ds["id"]
            box_data, box_labels, box_colors = [], [], []
            for version in versions:
                key = (ds_id, version, m)
                if key in auc_data and auc_data[key]:
                    box_data.append(auc_data[key])
                    box_labels.append(version)
                    box_colors.append(palette.get(version, "gray"))
            if not box_data:
                ax.set_title(f"ds{ds_id}", fontsize=8); continue
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
            for patch, c in zip(bp["boxes"], box_colors):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            for med in bp["medians"]: med.set_color("black")
            ax.set_xticklabels(box_labels, fontsize=7, rotation=15)
            ax.set_title(f"ds{ds_id}\nn={ds['n']}, p={ds['p']}\nk={ds['k']}, sig={ds['signal']}",
                         fontsize=7)
            ax.set_ylim(0.3, 1.05)
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("ROC AUC (per fold)")
        fig.suptitle(f"AUC — {m} — V1 vs V2_unc vs V2_constr", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "AUC" / f"AUC_{m.replace(' ','_')}.pdf", dpi=150)
        plt.close(fig)
    print("AUC boxplots saved → post_processing/AUC/")

    # ── Feature recovery — precision & recall vs ground truth ─────────────────
    recovery_records = []
    for ds in datasets:
        ds_id = ds["id"]
        true_set = set(gt_data.get(ds_id, []))
        if not true_set: continue
        for version in versions:
            for m in stabl_models:
                key = (ds_id, version, m)
                sel = feat_data.get(key, set())
                if not sel:
                    precision = recall = f1 = 0.0
                    n_sel = 0
                else:
                    tp = len(sel & true_set)
                    precision = tp / len(sel)
                    recall    = tp / len(true_set)
                    f1        = (2 * precision * recall / (precision + recall)
                                 if (precision + recall) > 0 else 0.0)
                    n_sel = len(sel)
                recovery_records.append({
                    "dataset_id": ds_id, "n": ds["n"], "p": ds["p"],
                    "k": ds["k"], "signal": ds["signal"],
                    "version": version, "model": m,
                    "n_selected": n_sel, "precision": round(precision, 3),
                    "recall": round(recall, 3), "f1": round(f1, 3),
                })

    if recovery_records:
        df_rec = pd.DataFrame(recovery_records)
        df_rec.to_csv(out / "feature_recovery" / "recovery_table.csv", index=False)

        for m in stabl_models:
            df_m = df_rec[df_rec["model"] == m]
            if df_m.empty: continue
            fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
            metrics = ["precision", "recall", "f1"]
            for ax, metric in zip(axes, metrics):
                x = np.arange(len(datasets))
                w = 0.25
                for i, version in enumerate(versions):
                    vals = [df_m[(df_m["dataset_id"] == ds["id"]) &
                                 (df_m["version"] == version)][metric].values
                            for ds in datasets]
                    vals = [v[0] if len(v) > 0 else 0 for v in vals]
                    ax.bar(x + i * w, vals, w, label=version,
                           color=palette.get(version, "gray"), alpha=0.8)
                ax.set_xticks(x + w)
                ax.set_xticklabels([f"ds{ds['id']}" for ds in datasets], fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.axhline(1.0, color="gray", ls="--", lw=0.5)
                ax.set_ylabel(metric); ax.set_title(metric.upper())
                ax.legend(fontsize=8)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.suptitle(f"Feature recovery — {m}\n(precision/recall/F1 vs ground truth)", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "feature_recovery" / f"recovery_{m.replace(' ','_')}.pdf", dpi=150)
            plt.close(fig)
    print("Feature recovery saved → post_processing/feature_recovery/")
    print("\nPost-processing complete.")


# ── Mode 2 : theory analysis ───────────────────────────────────────────────────

def run_theory_analysis(params_path="./params.json", ds_idx=None):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from stabl.stablV2 import Stabl as StablV2
    from stabl.stabl import Stabl as StablV1
    from stabl.adaptive import ALasso

    params = _read_json(params_path)
    datasets = params["datasets"]
    if ds_idx is not None:
        datasets = [datasets[ds_idx]]
    g = params["general"]

    B_VALUES  = [200, 500, 1000, 2000, 5000, 10000]
    B_SWEEP   = B_VALUES
    B_REF     = 10000
    K_STAB    = 20
    DELTA     = 0.05
    RANDOM_STATE = g["random_state"]

    out = Path("results") / "post_processing" / "fdp_theory"
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    def _bernstein_eps_fw(sigma2_vec, B, log_t):
        B_eff = max(B - 1, 1)
        disc  = (7/3)**2 * log_t**2 + 8 * B_eff * log_t * sigma2_vec
        return ((7/3) * log_t + np.sqrt(np.maximum(disc, 0.0))) / (2 * B_eff)

    def _t_star_unc(m):
        fdrs   = np.array(m.FDRs_)
        thresh = np.array(m.fdr_threshold_range)
        idx    = int(np.where(fdrs == fdrs.min())[0][0])
        return float(thresh[idx]), float(fdrs[idx])

    def _t_star_constr_bfw(m, scores, sigma2_vec, B, log_t, sigma2_ko=None):
        eps_j    = _bernstein_eps_fw(sigma2_vec, B, log_t)
        eps_ko_j = _bernstein_eps_fw(sigma2_ko if sigma2_ko is not None else sigma2_vec, B, log_t)
        eps_tot  = eps_j + eps_ko_j
        thresh   = np.array(m.fdr_threshold_range)
        fdrs     = np.array(m.FDRs_)
        feasible = np.array([
            np.sum((scores > t) & (scores <= t + eps_tot)) == 0
            for t in thresh
        ])
        if feasible.any():
            sub = np.where(feasible, fdrs, np.inf)
            idx = int(np.where(sub == sub.min())[0][0])
            return float(thresh[idx]), float(fdrs[idx]), True
        idx = int(np.where(fdrs == fdrs.min())[0][0])
        return float(thresh[idx]), float(fdrs[idx]), False

    BASE_ESTIMATORS = {
        "lasso":   LogisticRegression(penalty="l1", solver="liblinear",
                                      class_weight="balanced", max_iter=int(1e6)),
        "alasso":  ALasso(tol=1e-3),
    }

    all_records = []

    for ds in datasets:
        ds_id = ds["id"]
        print(f"\n{'='*60}")
        print(f"DATASET {ds_id}  n={ds['n']}  p={ds['p']}  k={ds['k']}  signal={ds['signal']}")
        print(f"{'='*60}")

        out_ds = out / f"dataset_{ds_id}"
        out_ds.mkdir(parents=True, exist_ok=True)

        _sweep_path = out_ds / "scores_all_B.npz"

        # ── Phase 1 : score collection uniquement si cache absent ──────────────
        if not _sweep_path.exists():
            print(f"  Pas de cache — sweep score collection uniquement…")
            X_df2, y2, true_features2, Sigma_true2 = generate_dataset(
                ds["n"], ds["p"], ds["k"], ds["signal"], ds["seed"])
            X2 = StandardScaler().fit_transform(X_df2.values, y2)
            lasso_base  = list(BASE_ESTIMATORS.values())[0]
            stab_seeds  = np.random.default_rng(RANDOM_STATE).integers(0, 2**31, K_STAB).tolist()
            versions_sweep = {
                "v1":       (StablV1, {}),
                "v2unc":    (StablV2, {"selection_mode": "unconstrained", "delta": DELTA, "cov_matrix": Sigma_true2}),
                "v2constr": (StablV2, {"selection_mode": "constrained",  "delta": DELTA, "cov_matrix": Sigma_true2}),
            }
            sweep_data = {
                "B_values"  : np.array(B_VALUES),
                "feat_names": np.array(X_df2.columns.tolist()),
                "true_mask" : np.array([f in set(true_features2) for f in X_df2.columns]),
                "DELTA"     : np.array([DELTA]),
            }
            for B in B_VALUES:
                for v_name, (VCls, vkw) in versions_sweep.items():
                    m_sw = VCls(base_estimator=clone(lasso_base), lambda_grid="auto",
                                artificial_type="knockoff", n_bootstraps=B,
                                random_state=RANDOM_STATE, **vkw)
                    try:
                        m_sw.fit(X2, y2)
                    except Exception as e:
                        print(f"    [B={B} {v_name}] failed: {e}"); continue
                    sv_v = getattr(m_sw, "score_variance_", None)
                    ko_v = getattr(m_sw, "ko_score_variance_", None)
                    if sv_v is None:
                        sv_v = np.zeros_like(m_sw.stabl_scores_)
                    pfx  = f"{v_name}_B{B}"
                    sweep_data[f"{pfx}_scores"]    = m_sw.stabl_scores_
                    sweep_data[f"{pfx}_ko_scores"] = m_sw.stabl_scores_artificial_
                    sweep_data[f"{pfx}_score_var"] = sv_v
                    sweep_data[f"{pfx}_ko_var"]    = ko_v if ko_v is not None else sv_v
                    sweep_data[f"{pfx}_fdrs"]      = np.array(m_sw.FDRs_)
                    sweep_data[f"{pfx}_thresh"]    = np.array(m_sw.fdr_threshold_range)
                    sweep_data[f"{pfx}_t_star"]    = np.array([m_sw.fdr_min_threshold_])
                    # scores multi-seeds pour variance/Jaccard en Phase 2
                    _seed_sc = []
                    for _s in stab_seeds:
                        try:
                            _m_s = VCls(base_estimator=clone(lasso_base), lambda_grid="auto",
                                        artificial_type="knockoff", n_bootstraps=B,
                                        random_state=int(_s), **vkw)
                            _m_s.fit(X2, y2)
                            _seed_sc.append(np.max(_m_s.stabl_scores_, axis=1))
                        except Exception:
                            pass
                    if _seed_sc:
                        sweep_data[f"{pfx}_seed_sc"] = np.stack(_seed_sc, axis=0)
                print(f"    B={B} ✓")
            np.savez(_sweep_path, **sweep_data)
            print(f"  Scores sauvegardés → {_sweep_path}")
            continue  # pas de plots cette fois

        # ── Phase 2 : génération des PDFs depuis le cache ──────────────────────
        print(f"  Cache trouvé — génération des PDFs…")
        X_df, y, true_features, Sigma_true = generate_dataset(
            ds["n"], ds["p"], ds["k"], ds["signal"], ds["seed"])
        X = X_df.values
        true_set = set(true_features)
        p_dim = X.shape[1]
        n_dim = X.shape[0]

        _d_cache = np.load(_sweep_path, allow_pickle=True)

        def _make_proxy(data, prefix):
            from types import SimpleNamespace
            p = SimpleNamespace()
            p.stabl_scores_            = data[f"{prefix}_scores"]
            p.stabl_scores_artificial_ = data[f"{prefix}_ko_scores"]
            p.score_variance_          = data[f"{prefix}_score_var"]
            p.ko_score_variance_       = data[f"{prefix}_ko_var"]
            p.FDRs_                    = list(data[f"{prefix}_fdrs"])
            p.fdr_threshold_range      = data[f"{prefix}_thresh"]
            p.fdr_min_threshold_       = float(data[f"{prefix}_t_star"][0])
            p.n_features_in_           = int(p.stabl_scores_.shape[0])
            return p

        feat_names = X_df.columns.tolist()
        records_B  = []

        for est_name, base_est in BASE_ESTIMATORS.items():
            print(f"  Estimateur : {est_name}")
            eps_pts_h, eps_pts_fw = [], []
            bdry_unc_pts = []
            t_unc_pts, t_constr_pts = [], []
            D_unc_pts, D_constr_pts = [], []
            prec_unc_pts, prec_constr_pts = [], []
            rec_unc_pts,  rec_constr_pts  = [], []
            f1_unc_pts,   f1_constr_pts   = [], []

            for B in B_VALUES:
                pfx_unc = f"v2unc_B{B}"
                if f"{pfx_unc}_scores" not in _d_cache.files:
                    for lst in [eps_pts_h, eps_pts_fw, bdry_unc_pts, t_unc_pts,
                                 t_constr_pts, D_unc_pts, D_constr_pts,
                                 prec_unc_pts, prec_constr_pts,
                                 rec_unc_pts, rec_constr_pts,
                                 f1_unc_pts, f1_constr_pts]:
                        lst.append(np.nan)
                    continue

                m_b      = _make_proxy(_d_cache, pfx_unc)
                scores   = np.max(m_b.stabl_scores_, axis=1)
                p_feat   = m_b.n_features_in_
                K_lambda = m_b.stabl_scores_.shape[1]
                log_t    = max(np.log(4 * p_feat * K_lambda / DELTA), 0.0)
                eps_B    = float(np.sqrt(log_t / (2 * B))) if B > 0 else 0.0

                sigma2_vec = m_b.score_variance_.max(axis=1)
                sigma2_ko  = (m_b.ko_score_variance_.max(axis=1)
                              if m_b.ko_score_variance_ is not None else sigma2_vec)
                eps_fw_j   = _bernstein_eps_fw(sigma2_vec, B, log_t)
                eps_ko_j   = _bernstein_eps_fw(sigma2_ko,  B, log_t)
                eps_tot_j  = eps_fw_j + eps_ko_j

                t_unc, fdp_unc = _t_star_unc(m_b)
                t_constr, fdp_constr, exact = _t_star_constr_bfw(
                    m_b, scores, sigma2_vec, B, log_t, sigma2_ko)

                D_unc    = int(np.sum(scores > t_unc))
                D_constr = int(np.sum(scores > t_constr))

                bdry_unc_ratio = int(np.sum(
                    (scores > t_unc) & (scores <= t_unc + eps_tot_j)
                )) / max(1, D_unc)

                sel_unc    = set(f for f, s in zip(feat_names, scores) if s > t_unc)
                sel_constr = set(f for f, s in zip(feat_names, scores) if s > t_constr)
                k_    = max(1, ds["k"])
                tp_u  = len(sel_unc & true_set);    tp_c = len(sel_constr & true_set)
                p_u   = tp_u / max(1, len(sel_unc)); r_u = tp_u / k_
                p_c   = tp_c / max(1, len(sel_constr)); r_c = tp_c / k_

                eps_pts_h.append(eps_B)
                eps_pts_fw.append(float(eps_fw_j.mean()))
                bdry_unc_pts.append(bdry_unc_ratio)
                t_unc_pts.append(t_unc);    t_constr_pts.append(t_constr)
                D_unc_pts.append(D_unc);    D_constr_pts.append(D_constr)
                prec_unc_pts.append(p_u);   prec_constr_pts.append(p_c)
                rec_unc_pts.append(r_u);    rec_constr_pts.append(r_c)
                f1_unc_pts.append(2*p_u*r_u/(p_u+r_u) if p_u+r_u > 0 else 0.0)
                f1_constr_pts.append(2*p_c*r_c/(p_c+r_c) if p_c+r_c > 0 else 0.0)

                records_B.append({
                    "dataset_id": ds_id, "estimateur": est_name,
                    "B": B,
                    "eps_B_hoeff": round(eps_B, 5),
                    "eps_fw_mean": round(float(eps_fw_j.mean()), 5),
                    "t_star_unc":    round(t_unc, 4),
                    "D_unc":         D_unc,
                    "fdp_unc":       round(fdp_unc, 4),
                    "bdry_unc_ratio": round(bdry_unc_ratio, 4),
                    "t_star_constr": round(t_constr, 4),
                    "D_constr":      D_constr,
                    "fdp_constr":    round(fdp_constr, 4),
                    "exact_constr":  exact,
                    "D_diff":        D_constr - D_unc,
                    "precision_unc":    round(p_u, 3),
                    "precision_constr": round(p_c, 3),
                    "recall_unc":       round(r_u, 3),
                    "recall_constr":    round(r_c, 3),
                })

            B_arr = np.array(B_VALUES[:len(eps_pts_h)])

            # ── Plot: epsilon + D + boundary + precision + recall + F1 vs B ──
            fig, axes = plt.subplots(2, 3, figsize=(18, 8))
            axes = axes.flatten()

            axes[0].plot(B_arr, eps_pts_h,  "o-", color="#FF7F0E", label="ε_Hoeff (référence)")
            axes[0].plot(B_arr, eps_pts_fw, "s-", color="#2CA02C", label="ε̄_fw BernFW")
            axes[0].set(xlabel="B", ylabel="ε", title="Décroissance des epsilons vs B")
            axes[0].legend(fontsize=8); axes[0].set_xscale("log")
            axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

            axes[1].plot(B_arr, D_unc_pts,    "o-", color="#C41E3A", label="D(t* unc)")
            axes[1].plot(B_arr, D_constr_pts, "s-", color="#001A7B", label="D(t* constr)")
            axes[1].axhline(ds["k"], color="gray", ls="--", lw=0.8, label=f"k={ds['k']}")
            axes[1].set(xlabel="B", ylabel="D(t*)", title="Features sélectionnées vs B")
            axes[1].legend(fontsize=8); axes[1].set_xscale("log")
            axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

            axes[2].plot(B_arr, bdry_unc_pts, "o-", color="#2CA02C", label="|∂⁺(t*_unc)|/D")
            axes[2].axhline(0, color="black", lw=0.5)
            axes[2].set(xlabel="B", ylabel="|∂⁺|/D", title="Frontière BernFW au t* unc")
            axes[2].legend(fontsize=8); axes[2].set_xscale("log")
            axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)

            for ax_i, (pu, pc, ylabel, title) in zip(axes[3:], [
                (prec_unc_pts, prec_constr_pts, "Précision", "Précision vs B"),
                (rec_unc_pts,  rec_constr_pts,  "Rappel",    "Rappel vs B"),
                (f1_unc_pts,   f1_constr_pts,   "F1",        "F1 vs B"),
            ]):
                ax_i.plot(B_arr, pu, "o-", color="#C41E3A", label="t* unc")
                ax_i.plot(B_arr, pc, "s-", color="#001A7B", label="t* constr")
                ax_i.axhline(1.0, color="gray", ls="--", lw=0.5)
                ax_i.set(xlabel="B", ylabel=ylabel, title=title, ylim=(-0.05, 1.05))
                ax_i.legend(fontsize=8); ax_i.set_xscale("log")
                ax_i.spines["top"].set_visible(False); ax_i.spines["right"].set_visible(False)

            fig.suptitle(
                f"Théorie BernFW — dataset {ds_id} (n={ds['n']}, p={ds['p']}, "
                f"k={ds['k']}, signal={ds['signal']}) — {est_name}",
                fontsize=10)
            fig.tight_layout()
            fig.savefig(out_ds / f"theory_curves_{est_name}.pdf", dpi=150)
            plt.close(fig)

        # ── Modèles B_REF depuis le cache ─────────────────────────────────────
        B_ref        = B_REF
        X2           = StandardScaler().fit_transform(X, y)
        y2           = y
        true_set2    = true_set
        feat_names2  = feat_names
        true_mask2   = np.array([f in true_set2 for f in feat_names2])
        Sigma_true2  = Sigma_true
        lasso_base   = list(BASE_ESTIMATORS.values())[0]

        _pfx         = f"B{B_ref}"
        _d           = _d_cache
        m_v1_ref     = _make_proxy(_d, f"v1_{_pfx}")
        m_constr_ref = _make_proxy(_d, f"v2constr_{_pfx}")
        scores_v1_ref     = m_v1_ref.stabl_scores_.max(axis=1)
        scores_constr_ref = m_constr_ref.stabl_scores_.max(axis=1)
        print(f"  V1    t*={m_v1_ref.fdr_min_threshold_:.3f}")
        print(f"  V2_constr t*={m_constr_ref.fdr_min_threshold_:.3f}")

        n_est = len(BASE_ESTIMATORS)
        fig3, axes3 = plt.subplots(4, n_est, figsize=(6 * n_est, 15), sharex="col")
        axes3 = np.array(axes3).reshape(4, n_est)
        _ROW_LABELS = ["FDP⁺(t)", "D(t)", "|∂⁺(t)| absolu", "|∂⁺(t)| / D(t)"]
        _ROW_COLORS = ["#4A90D9", "#C41E3A", "#2CA02C", "#1a7b1a"]

        fig_pr, axes_pr = plt.subplots(3, n_est, figsize=(6 * n_est, 10), sharex="col")
        axes_pr = np.array(axes_pr).reshape(3, n_est)

        for col, (est_name, base_est) in enumerate(BASE_ESTIMATORS.items()):
            m = _make_proxy(_d, f"v2unc_{_pfx}")

            scores   = np.max(m.stabl_scores_, axis=1)
            p_feat   = m.n_features_in_
            K_lambda = m.stabl_scores_.shape[1]
            log_t    = max(np.log(4 * p_feat * K_lambda / DELTA), 0.0)
            sigma2_v = m.score_variance_.max(axis=1)
            sigma2_k = (m.ko_score_variance_.max(axis=1)
                        if m.ko_score_variance_ is not None else sigma2_v)
            eps_fw_j = _bernstein_eps_fw(sigma2_v, B_ref, log_t)
            eps_ko_j = _bernstein_eps_fw(sigma2_k, B_ref, log_t)
            eps_tot  = eps_fw_j + eps_ko_j

            thresh   = np.array(m.fdr_threshold_range)
            fdrs_arr = np.array(m.FDRs_)
            D_t      = np.array([int(np.sum(scores > t)) for t in thresh])
            bdry_t   = np.array([int(np.sum((scores > t) & (scores <= t + eps_tot)))
                                  for t in thresh])
            bdry_D   = bdry_t / np.maximum(D_t, 1)

            feasible_bfw = bdry_t == 0

            t_unc, _       = _t_star_unc(m)
            t_constr, _, _ = _t_star_constr_bfw(m, scores, sigma2_v, B_ref, log_t, sigma2_k)

            curves = [fdrs_arr, D_t.astype(float), bdry_t.astype(float), bdry_D]

            for row, (curve, color, ylabel) in enumerate(
                    zip(curves, _ROW_COLORS, _ROW_LABELS)):
                ax = axes3[row, col]
                ax.step(thresh, curve, where="post", color=color, lw=1.5)

                ylo = curve.min(); yhi = curve.max()
                pad = (yhi - ylo) * 0.12 if yhi > ylo else 0.05
                ax.set_ylim(ylo - pad, yhi + pad)

                _lbl = "faisable BernFW" if row == 0 else None
                for t_f in thresh[feasible_bfw]:
                    ax.axvspan(t_f, t_f + float(eps_tot.max()),
                               alpha=0.15, color="#2CA02C", label=_lbl, zorder=0)
                    _lbl = None
                    ax.axvline(t_f, ymin=0, ymax=0.08, color="#2CA02C", lw=1.0, alpha=0.8)

                if row in (2, 3):
                    ax.axhline(0, color="black", lw=0.8, ls="--",
                               label="∂⁺=∅" if row == 2 else None)

                ax.axvline(t_unc,    color="#C41E3A", lw=1.5, ls="--",
                           label=f"t* unc={t_unc:.3f}" if row == 0 else None)
                ax.axvline(t_constr, color="#001A7B", lw=1.5, ls="-.",
                           label=f"t* BernFW constr={t_constr:.3f}" if row == 0 else None)

                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                if row == 0:
                    true_fdp_t2 = np.array([
                        np.sum((scores > t) & ~true_mask2) / max(1, int(np.sum(scores > t)))
                        for t in thresh
                    ])
                    ax.step(thresh, true_fdp_t2, where="post", color="#FF7F0E",
                            lw=1.2, ls="--", alpha=0.85, label="Vrai FDP(t)")
                    ax.set_title(
                        f"{est_name}\nB={B_ref}, δ={DELTA}, ε̄_fw={float(eps_fw_j.mean()):.4f}",
                        fontsize=8)
                    ax.legend(fontsize=7, loc="upper right")
                if row == 2:
                    ax.legend(fontsize=7)
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=8)
                if row == 3:
                    ax.set_xlabel("t", fontsize=8)

            # ── Précision / Recall / F1 vs t ──────────────────────────────
            prec_t_arr, rec_t_arr = [], []
            for t in thresh:
                sel = set(f for f, s in zip(feat_names2, scores) if s > t)
                tp  = len(sel & true_set2)
                prec_t_arr.append(tp / len(sel) if sel else 0.0)
                rec_t_arr.append(tp / len(true_set2))
            prec_t_arr = np.array(prec_t_arr)
            rec_t_arr  = np.array(rec_t_arr)
            f1_t_arr   = np.where(prec_t_arr + rec_t_arr > 0,
                                   2 * prec_t_arr * rec_t_arr / (prec_t_arr + rec_t_arr), 0.0)

            def _at_t(arr, t_q, _thresh=thresh):
                idx = max(np.searchsorted(_thresh, t_q, side="right") - 1, 0)
                return float(arr[idx])

            prec_u = _at_t(prec_t_arr, t_unc)
            rec_u  = _at_t(rec_t_arr,  t_unc)
            f1_u   = _at_t(f1_t_arr,   t_unc)
            prec_c = _at_t(prec_t_arr, t_constr)
            rec_c  = _at_t(rec_t_arr,  t_constr)
            f1_c   = _at_t(f1_t_arr,   t_constr)
            print(f"    {est_name} — t* unc    → prec={prec_u:.3f}  rec={rec_u:.3f}  F1={f1_u:.3f}")
            print(f"    {est_name} — t* constr → prec={prec_c:.3f}  rec={rec_c:.3f}  F1={f1_c:.3f}")

            for row_pr, (curve_pr, ylabel_pr, yvals) in enumerate([
                (prec_t_arr, "Précision", (prec_u, prec_c)),
                (rec_t_arr,  "Recall",    (rec_u,  rec_c)),
                (f1_t_arr,   "F1",        (f1_u,   f1_c)),
            ]):
                ax_pr = axes_pr[row_pr, col]
                ax_pr.step(thresh, curve_pr, where="post", color="#555555", lw=1.5)
                ax_pr.axvline(t_unc,    color="#C41E3A", lw=1.5, ls="--",
                              label=f"t* unc → {yvals[0]:.2f}")
                ax_pr.axvline(t_constr, color="#001A7B", lw=1.5, ls="-.",
                              label=f"t* constr → {yvals[1]:.2f}")
                ax_pr.set_ylabel(ylabel_pr, fontsize=8)
                ax_pr.set_ylim(-0.05, 1.1)
                ax_pr.axhline(1.0, color="gray", ls="--", lw=0.5)
                ax_pr.legend(fontsize=7)
                ax_pr.spines["top"].set_visible(False); ax_pr.spines["right"].set_visible(False)
                if row_pr == 0:
                    ax_pr.set_title(f"{est_name}", fontsize=8)
                if row_pr == 2:
                    ax_pr.set_xlabel("t", fontsize=8)

        fig3.suptitle(
            f"FDP⁺ / D(t) / Frontière BernFW — dataset {ds_id} "
            f"(n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']})\n"
            f"rouge-- = t* unconstrained | bleu-. = t* BernFW contraint | vert = zone faisable",
            fontsize=9)
        fig3.tight_layout()
        fig3.savefig(out_ds / "objective_function.pdf", dpi=150)
        plt.close(fig3)

        fig_pr.suptitle(
            f"Précision, Recall & F1 vs t — dataset {ds_id} "
            f"(n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']}, B={B_ref})\n"
            f"rouge-- = t* unc | bleu-. = t* BernFW constr",
            fontsize=9)
        fig_pr.tight_layout()
        fig_pr.savefig(out_ds / "precision_recall_vs_t.pdf", dpi=150)
        plt.close(fig_pr)

        # ── FDP+(t) et vrai FDP(t) pour les 3 versions ─────────────────────
        m_unc_fdp      = _make_proxy(_d, f"v2unc_{_pfx}")
        scores_unc_fdp = np.max(m_unc_fdp.stabl_scores_, axis=1)
        versions_fdp_t = {
            "V1":        (m_v1_ref,     scores_v1_ref,     "#C41E3A"),
            "V2_unc":    (m_unc_fdp,    scores_unc_fdp,    "#2CA02C"),
            "V2_constr": (m_constr_ref, scores_constr_ref, "#001A7B"),
        }
        fig_ft, axes_ft = plt.subplots(1, 3, figsize=(16, 5))
        for ax_ft, (v_name, (mv, sc_v, color)) in zip(axes_ft, versions_fdp_t.items()):
            th_v   = np.array(mv.fdr_threshold_range)
            fp_v   = np.array(mv.FDRs_)
            tfdp_v = np.array([
                np.sum((sc_v > t) & ~true_mask2) / max(1, int(np.sum(sc_v > t)))
                for t in th_v
            ])
            t_star_v = float(mv.fdr_min_threshold_)
            fdp_lbl = "FDP+(t)  (+1 num.)" if v_name == "V1" else "FDP+(t)  (sans +1)"
            ax_ft.step(th_v, fp_v,   where="post", color=color, lw=1.8, label=fdp_lbl)
            ax_ft.step(th_v, tfdp_v, where="post", color=color, lw=1.8, ls="--",
                       alpha=0.7, label="Vrai FDP(t)")
            # Augmented FDP+(t) — V2 uniquement (|∂⁺|/D remplace le +1 de V1)
            if v_name in ("V2_unc", "V2_constr"):
                log_t_v = max(np.log(4 * mv.n_features_in_ * mv.stabl_scores_.shape[1] / DELTA), 0.0)
                sv_v    = mv.score_variance_.max(axis=1)
                sk_v    = (mv.ko_score_variance_.max(axis=1)
                           if mv.ko_score_variance_ is not None else sv_v)
                eps_v   = (_bernstein_eps_fw(sv_v, B_ref, log_t_v)
                           + _bernstein_eps_fw(sk_v, B_ref, log_t_v))
                D_v     = np.array([max(1, int(np.sum(sc_v > t))) for t in th_v])
                bdry_v  = np.array([int(np.sum((sc_v > t) & (sc_v <= t + eps_v))) for t in th_v])
                corrected = np.minimum(fp_v + bdry_v / D_v, 1.0)
                ax_ft.step(th_v, corrected, where="post", color=color, lw=1.4, ls="-.",
                           alpha=0.85, label="FDP+_aug(t) = FDP+(t) + |∂⁺|/D")
            ax_ft.axvline(t_star_v, color="black", lw=1.2, ls=":", alpha=0.8,
                          label=f"t*={t_star_v:.3f}")
            ax_ft.set(xlabel="t", ylabel="FDP", title=v_name,
                      xlim=(0, 1), ylim=(-0.05, 1.1))
            ax_ft.axhline(0.2, color="gray", ls="--", lw=0.6, alpha=0.4, label="FDR=0.2")
            ax_ft.legend(fontsize=8)
            ax_ft.spines["top"].set_visible(False); ax_ft.spines["right"].set_visible(False)
        fig_ft.suptitle(
            f"FDP+(t) (solide) vs Vrai FDP(t) (tiret) — B={B_ref}\n"
            f"dataset {ds_id} (n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']})",
            fontsize=10)
        fig_ft.tight_layout()
        fig_ft.savefig(out_ds / "fdp_plus_vs_true_fdp.pdf", dpi=150)
        plt.close(fig_ft)
        print(f"  FDP+ vs vrai FDP → {out_ds}/fdp_plus_vs_true_fdp.pdf")

        # ── Calibration FDP+ : depuis le cache (seed unique) ──────────────
        fdp_palette     = {"v1": "#C41E3A", "v2_unc": "#2CA02C", "v2_constr": "#001A7B"}
        cache_fdp_vers  = {"v1": (m_v1_ref, scores_v1_ref),
                           "v2_unc": (m_unc_fdp, scores_unc_fdp),
                           "v2_constr": (m_constr_ref, scores_constr_ref)}
        fdp_records = []
        for v_name, (mv, sc_c) in cache_fdp_vers.items():
            t_sel    = float(mv.fdr_min_threshold_)
            fdp_plus = float(np.array(mv.FDRs_).min())
            sel      = set(f for f, s in zip(feat_names2, sc_c) if s > t_sel)
            true_fdp = len(sel - true_set2) / len(sel) if sel else 0.0
            tp       = len(sel & true_set2)
            prec_c   = tp / len(sel) if sel else 0.0
            rec_c    = tp / len(true_set2)
            f1_c     = 2*prec_c*rec_c/(prec_c+rec_c) if prec_c+rec_c > 0 else 0.0
            fdp_records.append({"version": v_name,
                                "fdp_plus": round(fdp_plus, 4),
                                "true_fdp": round(true_fdp, 4),
                                "n_selected": len(sel), "f1": round(f1_c, 3)})
        df_fdp = pd.DataFrame(fdp_records)
        df_fdp.to_csv(out_ds / "fdp_calibration.csv", index=False)

        v_names_fdp = list(cache_fdp_vers.keys())
        x_fdp = np.arange(len(v_names_fdp))
        fig_fdp, axes_fdp = plt.subplots(1, 3, figsize=(15, 5))
        ax = axes_fdp[0]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="borne exacte")
        ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.04, color="green",
                        label="zone valide (FDP+ ≥ FDP)")
        for v_name, row in zip(v_names_fdp, fdp_records):
            ax.scatter([row["fdp_plus"]], [row["true_fdp"]],
                       color=fdp_palette[v_name], label=v_name, alpha=0.9, s=80, zorder=3)
        ax.set(xlabel="FDP+ estimé", ylabel="Vrai FDP", xlim=(0, 1), ylim=(0, 1),
               title="Calibration FDP+\n(points sous diagonale = borne valide)")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        ax = axes_fdp[1]
        ax.bar(x_fdp, [r["f1"] for r in fdp_records],
               color=[fdp_palette[v] for v in v_names_fdp], alpha=0.85)
        ax.set_xticks(x_fdp); ax.set_xticklabels(v_names_fdp, fontsize=8)
        ax.set_ylim(0, 1.1); ax.axhline(1.0, color="gray", ls="--", lw=0.5)
        ax.set_ylabel("F1"); ax.set_title(f"F1 score (B={B_ref})")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        ax = axes_fdp[2]
        ax.axhline(ds["k"], color="gray", ls="--", lw=0.8, label=f"k={ds['k']}")
        ax.bar(x_fdp, [r["n_selected"] for r in fdp_records],
               color=[fdp_palette[v] for v in v_names_fdp], alpha=0.85)
        ax.set_xticks(x_fdp); ax.set_xticklabels(v_names_fdp, fontsize=8)
        ax.set_ylabel("n features sélectionnées")
        ax.set_title("Features sélectionnées vs vérité terrain")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig_fdp.suptitle(
            f"Calibration FDP+ — dataset {ds_id} "
            f"(n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']}) — B={B_ref}",
            fontsize=10)
        fig_fdp.tight_layout()
        fig_fdp.savefig(out_ds / "fdp_calibration.pdf", dpi=150)
        plt.close(fig_fdp)
        print(f"  FDP+ calibration → {out_ds}/fdp_calibration.pdf")

        # ── Comparaison sélections V1 / V2_unc / V2_constr à B_ref ──────────
        sc_unc_cmp   = scores_unc_fdp
        t_unc_cmp    = float(m_unc_fdp.fdr_min_threshold_)
        t_v1_cmp     = float(m_v1_ref.fdr_min_threshold_)
        t_constr_cmp = float(m_constr_ref.fdr_min_threshold_)

        sel_v1_cmp     = set(f for f, s in zip(feat_names2, scores_v1_ref) if s > t_v1_cmp)
        sel_unc_cmp    = set(f for f, s in zip(feat_names2, sc_unc_cmp)    if s > t_unc_cmp)
        sel_constr_cmp = set(f for f, s in zip(feat_names2, scores_constr_ref) if s > t_constr_cmp)

        def _cmp_metrics(sel, ts):
            tp = len(sel & ts)
            p_ = tp / len(sel) if sel else 0.0
            r_ = tp / len(ts)
            f_ = 2*p_*r_/(p_+r_) if p_+r_ > 0 else 0.0
            return p_, r_, f_

        v_labels_cmp  = ["V1", "V2_unc", "V2_constr"]
        sel_sets_cmp  = [sel_v1_cmp, sel_unc_cmp, sel_constr_cmp]
        t_stars_cmp   = [t_v1_cmp, t_unc_cmp, t_constr_cmp]
        n_sels_cmp    = [len(s) for s in sel_sets_cmp]
        metrics_cmp   = [_cmp_metrics(s, true_set2) for s in sel_sets_cmp]
        pal_cmp_v     = {"V1": "#C41E3A", "V2_unc": "#2CA02C", "V2_constr": "#001A7B"}

        all_sel_cmp  = sel_v1_cmp | sel_unc_cmp | sel_constr_cmp | true_set2
        feat_order_c = sorted(all_sel_cmp, key=lambda f: int(f.split("_")[1]))
        n_feat_c     = len(feat_order_c)

        fig_cmp, axes_cmp = plt.subplots(
            1, 3, figsize=(17, max(5, n_feat_c * 0.35 + 2)),
            gridspec_kw={"width_ratios": [1.2, 1.2, 2]})

        ax = axes_cmp[0]
        xc = np.arange(3); wc = 0.35
        ax.bar(xc - wc/2, t_stars_cmp, wc,
               color=[pal_cmp_v[v] for v in v_labels_cmp], alpha=0.85)
        ax_r2 = ax.twinx()
        ax_r2.bar(xc + wc/2, n_sels_cmp, wc,
                  color=[pal_cmp_v[v] for v in v_labels_cmp], alpha=0.35)
        ax_r2.axhline(ds["k"], color="gray", ls="--", lw=0.8, label=f"k={ds['k']}")
        ax_r2.legend(fontsize=7)
        ax.set_xticks(xc); ax.set_xticklabels(v_labels_cmp, fontsize=8)
        ax.set_ylabel("t* (threshold)", fontsize=8)
        ax_r2.set_ylabel("n sélectionnées", fontsize=8, color="gray")
        ax.set_title(f"Threshold & n_sel — B={B_ref}", fontsize=8)
        ax.spines["top"].set_visible(False)

        ax = axes_cmp[1]
        metric_names_c = ["Précision", "Recall", "F1"]
        for i, v_label in enumerate(v_labels_cmp):
            ax.bar(np.arange(3) + (i-1)*0.25, metrics_cmp[i], 0.25,
                   label=v_label, color=pal_cmp_v[v_label], alpha=0.85)
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(metric_names_c, fontsize=8)
        ax.set_ylim(0, 1.15); ax.axhline(1.0, color="gray", ls="--", lw=0.5)
        ax.set_ylabel("Score"); ax.legend(fontsize=7)
        ax.set_title("Précision / Recall / F1", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        ax = axes_cmp[2]
        mat_c = np.array([[1.0 if f in sel else 0.0 for sel in sel_sets_cmp]
                          for f in feat_order_c])
        ax.imshow(mat_c, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(v_labels_cmp, fontsize=8)
        ax.set_yticks(range(n_feat_c)); ax.set_yticklabels(feat_order_c, fontsize=7)
        for j, f in enumerate(feat_order_c):
            lbl = ax.get_yticklabels()[j]
            if f in true_set2:
                lbl.set_color("#C41E3A"); lbl.set_fontweight("bold")
        ax.set_title("Features sélectionnées\n(rouge gras = vraie feature)", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        fig_cmp.suptitle(
            f"Comparaison sélections — dataset {ds_id} "
            f"(n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']}) — B={B_ref}",
            fontsize=10)
        fig_cmp.tight_layout()
        fig_cmp.savefig(out_ds / "selection_comparison.pdf", dpi=150)
        plt.close(fig_cmp)
        print(f"  Selection comparison → {out_ds}/selection_comparison.pdf")

        # ── Stabilité V1 vs V2_unc vs V2_constr : variance + Jaccard + Précision ──
        STAB_THRESH = 0.5
        palette_cmp = {"v1": "#C41E3A", "v2_unc": "#2CA02C", "v2_constr": "#001A7B"}
        versions_cmp = list(palette_cmp.keys())
        _pfx_map_stab = {"v1": "v1", "v2_unc": "v2unc", "v2_constr": "v2constr"}

        def _jaccard(s1, s2):
            u = len(s1 | s2)
            return len(s1 & s2) / u if u > 0 else 1.0

        var_rec  = {v: [] for v in versions_cmp}
        jac_rec  = {v: [] for v in versions_cmp}
        prec_rec = {v: [] for v in versions_cmp}
        rec_rec  = {v: [] for v in versions_cmp}
        f1_rec   = {v: [] for v in versions_cmp}

        for B_s in B_SWEEP:
            for v_name in versions_cmp:
                c_pfx = f"{_pfx_map_stab[v_name]}_B{B_s}"
                # var + Jaccard depuis le cache multi-seeds
                if f"{c_pfx}_seed_sc" in _d_cache.files:
                    seed_sc = _d_cache[f"{c_pfx}_seed_sc"]
                    var_rec[v_name].append(float(seed_sc.var(axis=0).mean())
                                           if seed_sc.shape[0] >= 2 else np.nan)
                    sets = [frozenset(np.where(r > STAB_THRESH)[0]) for r in seed_sc]
                    pairs = [_jaccard(a, b) for i, a in enumerate(sets) for b in sets[i+1:]]
                    jac_rec[v_name].append(np.mean(pairs) if pairs else np.nan)
                else:
                    var_rec[v_name].append(np.nan)
                    jac_rec[v_name].append(np.nan)
                # prec/rec/f1 depuis le cache single-seed
                if f"{c_pfx}_scores" in _d_cache.files:
                    m_c   = _make_proxy(_d_cache, c_pfx)
                    sc_c  = np.max(m_c.stabl_scores_, axis=1)
                    t_c   = m_c.fdr_min_threshold_
                    sel_c = set(f for f, s in zip(feat_names2, sc_c) if s > t_c)
                    tp_c  = len(sel_c & true_set2)
                    p_c   = tp_c / len(sel_c) if sel_c else 0.0
                    r_c   = tp_c / max(1, ds["k"])
                    prec_rec[v_name].append(p_c)
                    rec_rec[v_name].append(r_c)
                    f1_rec[v_name].append(2*p_c*r_c/(p_c+r_c) if p_c+r_c > 0 else 0.0)
                else:
                    prec_rec[v_name].append(np.nan)
                    rec_rec[v_name].append(np.nan)
                    f1_rec[v_name].append(np.nan)

        B_sw = np.array(B_SWEEP[:len(var_rec["v1"])])
        fig_s, axes_s = plt.subplots(1, 5, figsize=(25, 5))
        for v_name in versions_cmp:
            c = palette_cmp[v_name]
            axes_s[0].plot(B_sw, var_rec[v_name],  "o-", color=c, label=v_name)
            axes_s[1].plot(B_sw, jac_rec[v_name],  "o-", color=c, label=v_name)
            axes_s[2].plot(B_sw, prec_rec[v_name], "o-", color=c, label=v_name)
            axes_s[3].plot(B_sw, rec_rec[v_name],  "o-", color=c, label=v_name)
            axes_s[4].plot(B_sw, f1_rec[v_name],   "o-", color=c, label=v_name)
        axes_s[0].set(xlabel="B", ylabel="Variance moyenne des scores",
                      title="Variance des scores vs B")
        axes_s[0].set_xscale("log"); axes_s[0].legend(fontsize=8)
        axes_s[0].spines["top"].set_visible(False); axes_s[0].spines["right"].set_visible(False)
        axes_s[1].set(xlabel="B", ylabel="Jaccard moyen inter-seeds",
                      title="Stabilité Jaccard vs B", ylim=(0, 1.05))
        axes_s[1].axhline(1.0, color="gray", ls="--", lw=0.5)
        axes_s[1].set_xscale("log"); axes_s[1].legend(fontsize=8)
        axes_s[1].spines["top"].set_visible(False); axes_s[1].spines["right"].set_visible(False)
        for ax_i, (metric_d, ylabel, title) in zip(axes_s[2:], [
            (prec_rec, "Précision", "Précision vs B"),
            (rec_rec,  "Rappel",    "Rappel vs B"),
            (f1_rec,   "F1",        "F1 vs B"),
        ]):
            for v_name in versions_cmp:
                ax_i.plot(B_sw, metric_d[v_name], "o-", color=palette_cmp[v_name], label=v_name)
            ax_i.set(xlabel="B", ylabel=ylabel, title=title, ylim=(-0.05, 1.05))
            ax_i.axhline(1.0, color="gray", ls="--", lw=0.5)
            ax_i.set_xscale("log"); ax_i.legend(fontsize=8)
            ax_i.spines["top"].set_visible(False); ax_i.spines["right"].set_visible(False)
        fig_s.suptitle(
            f"Stabilité V1 vs V2_unc vs V2_constr — dataset {ds_id} "
            f"(n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']})",
            fontsize=10)
        fig_s.tight_layout()
        out_ds.mkdir(parents=True, exist_ok=True)
        fig_s.savefig(out_ds / "stability_comparison.pdf", dpi=150)
        plt.close(fig_s)

        # ── ε_{B,W,j} et W_j en fonction de B (v2_constr uniquement) ─────────
        true_mask_ds = np.array([f in true_set for f in feat_names])
        eps_med_sig, eps_q1_sig, eps_q3_sig = [], [], []
        eps_med_nul, eps_q1_nul, eps_q3_nul = [], [], []
        wj_med_sig,  wj_med_nul             = [], []
        B_eps_sw = []

        for B in B_SWEEP:
            c_pfx = f"v2constr_B{B}"
            if f"{c_pfx}_scores" not in _d_cache.files:
                continue
            sc  = _d_cache[f"{c_pfx}_scores"]     # (p, K)
            sv  = _d_cache[f"{c_pfx}_score_var"]  # (p, K)
            kov = _d_cache[f"{c_pfx}_ko_var"]     # (n_inj, K)
            sko = _d_cache[f"{c_pfx}_ko_scores"]  # (n_inj, K)
            p_, K_ = sc.shape[0], sc.shape[1]
            if kov.shape[0] != p_:
                continue

            idx_p  = np.arange(p_)
            k_st   = np.argmax(sc, axis=1)
            sig2_W = sv[idx_p, k_st] + kov[idx_p, k_st]
            w_j    = sc[idx_p, k_st] - sko[idx_p, k_st]

            L_W   = max(np.log(2.0 * p_ * K_ / DELTA), 0.0)
            B_eff = max(B - 1, 1)
            disc  = (7/3)**2 * L_W**2 + 8 * B_eff * L_W * sig2_W
            eps_W = ((7/3) * L_W + np.sqrt(np.maximum(disc, 0.0))) / (2 * B_eff)

            B_eps_sw.append(B)
            for mask, me, q1, q3, mw in [
                (true_mask_ds,  eps_med_sig, eps_q1_sig, eps_q3_sig, wj_med_sig),
                (~true_mask_ds, eps_med_nul, eps_q1_nul, eps_q3_nul, wj_med_nul),
            ]:
                e = eps_W[mask]; w = w_j[mask]
                me.append(np.median(e));  q1.append(np.percentile(e, 25))
                q3.append(np.percentile(e, 75)); mw.append(np.median(w))

        if B_eps_sw:
            B_arr = np.array(B_eps_sw)
            fig_eps, ax_eps = plt.subplots(figsize=(7, 5))
            ax_eps.plot(B_arr, eps_med_sig, "o-",  color="#001A7B",
                        label=r"$\varepsilon_{B,W,j}$ — signal")
            ax_eps.fill_between(B_arr, eps_q1_sig, eps_q3_sig,
                                 color="#001A7B", alpha=0.15)
            ax_eps.plot(B_arr, wj_med_sig,  "s--", color="#001A7B",
                        label=r"$W_j$ — signal", alpha=0.7)
            ax_eps.plot(B_arr, eps_med_nul, "o-",  color="#C41E3A",
                        label=r"$\varepsilon_{B,W,j}$ — null")
            ax_eps.fill_between(B_arr, eps_q1_nul, eps_q3_nul,
                                 color="#C41E3A", alpha=0.15)
            ax_eps.plot(B_arr, wj_med_nul,  "s--", color="#C41E3A",
                        label=r"$W_j$ — null", alpha=0.7)
            ax_eps.axhline(0, color="gray", lw=0.5, ls=":")
            ax_eps.set_xscale("log")
            ax_eps.set(xlabel="B (bootstraps)", ylabel="Valeur",
                       title=(rf"$\varepsilon_{{B,W,j}}$ et $W_j$ vs B — v2_constr"
                              f"\nds{ds_id} (n={ds['n']}, p={ds['p']},"
                              f" k={ds['k']}, signal={ds['signal']})"))
            ax_eps.legend(fontsize=8)
            ax_eps.spines["top"].set_visible(False)
            ax_eps.spines["right"].set_visible(False)
            fig_eps.tight_layout()
            fig_eps.savefig(out_ds / "eps_W_vs_B.pdf", dpi=150)
            plt.close(fig_eps)
            print(f"  ε_W vs B → {out_ds}/eps_W_vs_B.pdf")

        print(f"  → {out_ds}/")

        all_records.extend(records_B)

    pd.DataFrame(all_records).to_csv(out / "theory_table.csv", index=False)
    print(f"\nThéorie → {out}/ (theory_table.csv + un dossier par dataset)")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",      type=int, help="0=experiment, 1=post-process, 2=theory, 3=parse-params")
    parser.add_argument("idx",       type=int, nargs="?", default=None)
    parser.add_argument("--params",  type=str, default="./params.json")
    args = parser.parse_args()

    if args.mode == 0:
        run_experiment(args.idx if args.idx is not None else 0)
    elif args.mode == 1:
        post_process(args.params)
    elif args.mode == 2:
        run_theory_analysis(args.params, ds_idx=args.idx)
    elif args.mode == 3:
        parse_params(args.params)
