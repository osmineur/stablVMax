import os
import shutil
import json
import argparse
from pathlib import Path
from string import Template


# ── SBATCH templates ───────────────────────────────────────────────────────────

_arrayTemplate = Template("""#!/bin/bash
#SBATCH --job-name=${name}_${variant}
#SBATCH --error=./logs/${name}_${variant}_%a.err
#SBATCH --output=./logs/${name}_${variant}_%a.out
#SBATCH --array=0-${rep}
#SBATCH --time=${time}
#SBATCH -p normal
#SBATCH -c ${cpu}
#SBATCH --mem=${mem}GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${email}

module load python/3.12.1
source ~/stablVMax/.venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=$$SLURM_CPUS_PER_TASK

time python3 ./sendOut.py 0 $${SLURM_ARRAY_TASK_ID} ${variant}
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

module load python/3.12.1
source ~/stablVMax/.venv/bin/activate

echo "=== Post-processing ==="
time python3 ./sendOut.py 1 --params ${params_file} --out ${out_dir}

echo "=== Theory analysis (variance, FDP+, gaussianity) ==="
time python3 ./sendOut.py 2 --params ${params_file}

echo "=== Correlation IMC x CyTOF ==="
time python3 ./correlation_IMC_CyTOF.py --params ${params_file}
""")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_json(path):
    with open(path) as f:
        return json.load(f)

def _write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


# ── parse_params ───────────────────────────────────────────────────────────────

def parse_params(paramsFile: str) -> None:
    params = _read_json(paramsFile)
    g = params["general"]
    name = params["Experiment_Name"].replace(" ", "_")
    email = params.get("email", "")

    os.makedirs("./temp/",    exist_ok=True)
    os.makedirs("./logs/",    exist_ok=True)
    os.makedirs("./results/", exist_ok=True)

    # Supprime les répertoires de résultats parents avant soumission des jobs
    # (parse_params tourne une seule fois sur le login node → pas de race condition)
    parent_dirs = set()
    for run in params["runs"]:
        parent_dirs.add(Path(run["save_path"]).parent)
    for d in parent_dirs:
        if d.exists():
            shutil.rmtree(d)
            print(f"parse_params: supprimé {d}/")
        d.mkdir(parents=True, exist_ok=True)

    high_count = 0
    for idx, run in enumerate(params["runs"]):
        run_dir = Path(f"./results/h/{idx}")
        run_dir.mkdir(parents=True, exist_ok=True)
        run_params = {
            **run,
            "data_path":         params["data_path"],
            "taskType":          g["taskType"],
            "n_bootstraps":      g["n_bootstraps"],
            "n_splits":          g["n_splits"],
            "n_repeats":         g["n_repeats"],
            "random_state":      g["random_state"],
            "density_threshold": g["density_threshold"],
            "n_jobs":            g["n_jobs"],
            "shorthand":         f"{idx}_h",
        }
        _write_json(run_params, run_dir / "params.json")
        high_count += 1

    script = _arrayTemplate.substitute(
        name=name, variant="h", rep=high_count - 1,
        cpu=g["cpusHigh"], mem=g["memHighGB"], time=g["time"], email=email
    )
    with open("./temp/arrayHigh.sh", "w") as f:
        f.write(script)

    params_file = Path(paramsFile).name
    out_dir = "post_processing_unc" if "unc" in params_file else "post_processing"
    end_script = _endTemplate.substitute(name=name, email=email,
                                         params_file=f"./{params_file}",
                                         out_dir=out_dir)
    with open("./temp/end.sh", "w") as f:
        f.write(end_script)

    print(f"parse_params: {high_count} high jobs — scripts written to ./temp/")


# ── ML imports ─────────────────────────────────────────────────────────────────

def _import_ml():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning,   module="sklearn")
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"

    import pandas as pd
    import numpy as np
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn import clone
    from stabl.stabl  import Stabl as StablV1
    from stabl.stablV2 import Stabl as StablV2
    from stabl.adaptive import ALasso, ALogitLasso
    from stabl.preprocessing import LowInfoFilter, CorrelationFilter
    from stabl.multi_omic_pipelines import multi_omic_stabl_cv
    from xgboost import XGBClassifier
    return (pd, np, ElasticNet, LogisticRegression, GridSearchCV,
            RepeatedStratifiedKFold, StratifiedKFold, clone, StablV1, StablV2,
            ALasso, ALogitLasso, multi_omic_stabl_cv,
            VarianceThreshold, Pipeline, SimpleImputer,
            StandardScaler, LowInfoFilter, FunctionTransformer, CorrelationFilter,
            XGBClassifier)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(pd, data_path):
    X_density  = pd.read_csv(f"{data_path}/PerioII_PatientFeature_03052026_IMCdensity.csv",  index_col=0)
    X_function = pd.read_csv(f"{data_path}/PerioII_PatientFeature_03052026_IMCfunction.csv", index_col=0)
    X_neighbor = pd.read_csv(f"{data_path}/PerioII_PatientFeature_03052026_IMCneighbor.csv", index_col=0)
    pen_matrix = pd.read_csv(f"{data_path}/penalization_matrix_perio_imc.csv", index_col=0)
    return X_density, X_function, X_neighbor, pen_matrix


def preprocess(X_density, X_function, X_neighbor, pen_matrix, threshold):
    def filter_by_penalization(df, pm):
        return df[[c for c in df.columns
                   if (p := c.split("_"))[0] in pm.index
                   and p[-1] in pm.columns
                   and pm.loc[p[0], p[-1]] != 0]]

    def matches_neighbor(col, prefixes):
        parts = col.split("_")
        return "_".join(parts[:2]) in prefixes and parts[-1] + "_" + parts[1] in prefixes

    def matches_other(col, prefixes):
        return "_".join(col.split("_")[:2]) in prefixes

    median_densities = 1_000_000 * X_density.median(axis=0)
    prefixe_above = set(
        "_".join(c.split("_")[:2]) for c in median_densities[median_densities >= threshold].index
    )
    no_other = lambda df: df[[c for c in df.columns if "other" not in c.lower()]]

    X_density               = no_other(X_density)
    X_function_pen          = filter_by_penalization(X_function, pen_matrix)
    X_neighbor_filtered     = no_other(X_neighbor[[c for c in X_neighbor.columns  if matches_neighbor(c, prefixe_above)]])
    X_function_filtered_pen = no_other(X_function_pen[[c for c in X_function_pen.columns if matches_other(c, prefixe_above)]])
    X_function_filtered     = no_other(X_function[[c for c in X_function.columns   if matches_other(c, prefixe_above)]])

    return X_density, X_function_filtered_pen, X_function_filtered, X_neighbor_filtered


def _make_pipelines(VarianceThreshold, Pipeline, SimpleImputer,
                    StandardScaler, FunctionTransformer, LowInfoFilter, CorrelationFilter):
    """Build the two preprocessing pipelines (density and noisy layers)."""
    noisy = Pipeline([
        ("variance", VarianceThreshold(0.01)),
        ("lif",      LowInfoFilter()),
        ("impute",   SimpleImputer(strategy="median")),
        ("std",      StandardScaler()),
    ])
    density = Pipeline([
        ("to_million", FunctionTransformer(lambda X: X * 1e6, feature_names_out="one-to-one")),
        ("variance",   VarianceThreshold(0.01)),
        ("corr",       CorrelationFilter(threshold="auto")),
        ("lif",        LowInfoFilter()),
        ("impute",     SimpleImputer(strategy="median")),
        ("std",        StandardScaler()),
    ])
    return density, noisy


def make_estimators(p, np, GridSearchCV, LogisticRegression, ElasticNet,
                    clone, StablClass, ALasso, ALogitLasso, XGBClassifier,
                    StratifiedKFold, stabl_extra_kwargs=None):
    C_grid   = np.logspace(-2, 0, 10)
    n_jobs   = p["n_jobs"]
    nb       = p["n_bootstraps"]
    inner_cv = StratifiedKFold(n_splits=min(3, p["n_splits"]), shuffle=True, random_state=p["random_state"])
    stabl_kw = stabl_extra_kwargs or {}

    lasso_cv  = GridSearchCV(LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=1_000_000), {"C": C_grid}, cv=inner_cv, scoring="roc_auc", n_jobs=n_jobs)
    alasso_cv = GridSearchCV(ALogitLasso(solver="liblinear"), {"C": C_grid}, cv=inner_cv, scoring="roc_auc", n_jobs=n_jobs)
    en_cv     = GridSearchCV(LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", class_weight="balanced", max_iter=int(1e6)), {"C": C_grid}, cv=inner_cv, scoring="roc_auc", n_jobs=n_jobs)
    xgb_cv    = GridSearchCV(XGBClassifier(n_jobs=1, eval_metric="logloss", verbosity=0, random_state=p["random_state"]),
                             {"max_depth": [3, 5], "n_estimators": [50, 100, 200]},
                             cv=inner_cv, scoring="roc_auc", n_jobs=n_jobs)

    def make():
        return {
            "lasso":          clone(lasso_cv),
            "alasso":         clone(alasso_cv),
            "en":             clone(en_cv),
            "xgboost":        clone(xgb_cv),
            "stabl_lasso":    StablClass(base_estimator=LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=int(1e6)), lambda_grid="auto", artificial_type="knockoff", n_bootstraps=nb, **stabl_kw),
            "stabl_alasso":   StablClass(base_estimator=ALasso(tol=1e-3),                   lambda_grid="auto", artificial_type="knockoff", n_bootstraps=nb, **stabl_kw),
            "stabl_en":       StablClass(base_estimator=ElasticNet(l1_ratio=0.5, tol=1e-3), lambda_grid="auto", artificial_type="knockoff", n_bootstraps=nb, **stabl_kw),
        }
    return make


# ── Mode 0 : run one experiment ────────────────────────────────────────────────

def run_experiment(idx: int, intensity: str) -> None:
    (pd, np, ElasticNet, LogisticRegression, GridSearchCV,
     RepeatedStratifiedKFold, StratifiedKFold, clone, StablV1, StablV2,
     ALasso, ALogitLasso, multi_omic_stabl_cv,
     VarianceThreshold, Pipeline, SimpleImputer,
     StandardScaler, LowInfoFilter, FunctionTransformer, CorrelationFilter,
     XGBClassifier) = _import_ml()

    p = _read_json(Path(f"./results/{intensity}/{idx}/params.json"))
    version    = p.get("stabl_version", "v1")
    StablClass = StablV2 if version in ("v2", "v2_unc") else StablV1
    print(f"Starting run {idx} ({intensity}): {p['name']} [STABL {version.upper()}]")

    # Patch multi_omic_pipelines.save_stabl_results avec la bonne version
    import stabl.multi_omic_pipelines as _mop_mod
    if version in ("v2", "v2_unc"):
        from stabl.stablV2 import save_stabl_results as _ssr
    else:
        from stabl.stabl import save_stabl_results as _ssr
    _mop_mod.save_stabl_results = _ssr

    X_density, X_function, X_neighbor, pen_matrix = load_data(pd, p["data_path"])
    y = pd.Series([0 if x.startswith("HG") else 1 for x in X_density.index],
                  index=X_density.index, name="outcome")

    X_density_f, X_fn_pen, X_fn, X_nb = preprocess(
        X_density, X_function, X_neighbor, pen_matrix, p["density_threshold"]
    )

    data_dict = ({"Density": X_density_f, "Function": X_fn_pen, "Neighbor": X_nb}
                 if p["use_penalization"] else
                 {"Density": X_density_f, "Function": X_fn,     "Neighbor": X_nb})

    density_pipe, noisy_pipe = _make_pipelines(
        VarianceThreshold, Pipeline, SimpleImputer,
        StandardScaler, FunctionTransformer, LowInfoFilter, CorrelationFilter
    )

    save_path = Path(p["save_path"])
    # Supprime uniquement ce run (pas le parent) — évite la race condition sur Sherlock
    # quand penalized_v1 et penalized_v2 tournent en parallèle sur le même parent
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    csv_dir = save_path / "datalayers"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for layer_name, df in data_dict.items():
        df.to_csv(csv_dir / f"{layer_name}.csv")

    # Save λ_min diagnostics for V2 runs
    if version in ("v2", "v2_unc"):
        _save_lambda_min_stats(np, data_dict, save_path)

    outer_splitter       = RepeatedStratifiedKFold(n_splits=p["n_splits"], n_repeats=p["n_repeats"], random_state=p["random_state"])
    stabl_extra_kwargs = {}
    if version in ("v2", "v2_unc"):
        stabl_extra_kwargs["selection_mode"] = p.get("selection_mode", "unconstrained" if version == "v2_unc" else "constrained")
    make_stabl_estimators = make_estimators(p, np, GridSearchCV, LogisticRegression,
                                            ElasticNet, clone, StablClass, ALasso, ALogitLasso,
                                            XGBClassifier, StratifiedKFold,
                                            stabl_extra_kwargs=stabl_extra_kwargs)

    if version in ("v2", "v2_unc"):
        models_to_run = ["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]
    else:
        models_to_run = ["STABL Lasso", "STABL ALasso", "STABL ElasticNet",
                         "Lasso", "ALasso", "ElasticNet", "XGBoost"]

    multi_omic_stabl_cv(
        data_dict=data_dict, y=y,
        outer_splitter=outer_splitter,
        estimators=make_stabl_estimators(),
        task_type=p["taskType"],
        save_path=str(save_path),
        models=models_to_run,
        outer_groups=None,
        early_fusion=False, late_fusion=False, n_iter_lf=10000,
        preprocessing_overrides={
            "Density": density_pipe,
            "Function": noisy_pipe,
            "Neighbor": noisy_pipe
        },
    )
    print(f"Run {p['name']} done → {save_path}/")


def _save_lambda_min_stats(np, data_dict, save_path):
    from sklearn.covariance import LedoitWolf
    stats = {}
    for layer, df in data_dict.items():
        X = df.values.astype(float)
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        n, p = X.shape
        Sigma_emp  = np.cov(X.T)
        try:
            lmin_emp = float(np.linalg.eigvalsh(Sigma_emp).min())
        except np.linalg.LinAlgError:
            lmin_emp = float("nan")
        try:
            lw = LedoitWolf().fit(X)
        except Exception:
            lmin_lw = float("nan")
            stats[layer] = {"n": n, "p": p,
                            "lambda_min_empirical": lmin_emp,
                            "lambda_min_LW": float("nan"),
                            "shrinkage_coef": float("nan"),
                            "mean_abs_knockoff_corr_LW": float("nan")}
            continue
        lmin_lw    = float(np.linalg.eigvalsh(lw.covariance_).min())
        delta_lw   = min(2 * lmin_lw, 1.0)
        diag       = np.diag(lw.covariance_)
        ko_corr    = float(np.nanmean(np.abs(1.0 - delta_lw / np.where(diag > 0, diag, np.nan))))
        stats[layer] = {"n": n, "p": p,
                        "lambda_min_empirical": lmin_emp,
                        "lambda_min_LW":        lmin_lw,
                        "shrinkage_coef":       float(lw.shrinkage_),
                        "mean_abs_knockoff_corr_LW": ko_corr}
        print(f"  [{layer}] n={n}, p={p} | λ_min(Σ)={lmin_emp:.4f} | λ_min(Σ_LW)={lmin_lw:.4f}")

    theory_dir = Path(save_path) / "v2_theory"
    theory_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stats, theory_dir / "lambda_min.json")


# ── Mode 1 : post-processing (plots from CV results) ──────────────────────────

def post_process(params_path: str = "./params.json", out_dir: str = "post_processing") -> None:
    import numpy as np
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from collections import Counter

    params     = _read_json(params_path)
    runs       = params["runs"]
    stabl_all  = ["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]
    other_all  = ["Lasso", "ALasso", "ElasticNet", "XGBoost"]
    all_models = stabl_all + other_all

    out = Path(out_dir)
    for _sub in ["ROC", "AUC", "lambda_min", "jaccard"]:
        _d = out / _sub
        if _d.exists():
            shutil.rmtree(_d)
        _d.mkdir(parents=True, exist_ok=True)

    # ── Color palette ──
    palette = {
        "penalized_v1":     "#C41E3A",
        "penalized_v2":     "#001A7B",
        "unpenalized_v1":   "#FF7F7F",
        "unpenalized_v2":   "#6B8CFF",
        "penalized_v2_unc": "#2CA02C",
    }

    # ── Load data ──
    auc_data  = {}
    pred_data = {}
    feat_sets = {}

    for run in runs:
        sp    = Path(run["save_path"])
        rname = run["name"]
        auc_data[rname]  = {}
        pred_data[rname] = {}

        auc_path = sp / "Training CV" / "auc_progress.csv"
        if auc_path.exists():
            df_auc = pd.read_csv(auc_path, index_col=0)
            for m in all_models:
                if m in df_auc.columns:
                    auc_data[rname][m] = df_auc[m].dropna().tolist()

        for m in all_models:
            pred_path = sp / "Training CV" / m / f"{m} predictions.csv"
            if pred_path.exists():
                pred_data[rname][m] = pd.read_csv(pred_path)

        for m in all_models:
            feat_path = sp / "Training CV" / f"Selected Features {m}.csv"
            label = f"{rname} / {m}"
            if feat_path.exists():
                df_f = pd.read_csv(feat_path)
                feat_sets[label] = set(df_f.iloc[:, 0].tolist())
            else:
                feat_sets[label] = set()

    # ── ROC curves — one PDF per model ──
    for m in all_models:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        for run in runs:
            rname = run["name"]
            if rname not in pred_data or m not in pred_data[rname]:
                continue
            df_p = pred_data[rname][m]
            score_col = [c for c in df_p.columns if c not in ("Patient", "outcome")][0]
            fpr, tpr, _ = roc_curve(df_p["outcome"], df_p[score_col])
            roc_auc = auc(fpr, tpr)
            v = run.get("stabl_version", "v1").upper()
            pen = "pen." if run.get("use_penalization") else "unpen."
            ax.plot(fpr, tpr, color=palette.get(rname, "gray"), lw=2,
                    label=f"{v} {pen}  AUC={roc_auc:.3f}")
        ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC — {m}")
        ax.legend(loc="lower right", fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "ROC" / f"ROC_{m.replace(' ','_')}.pdf", dpi=150)
        plt.close(fig)

    print("ROC curves saved → post_processing/ROC/")

    # ── AUC boxplots — one figure, all models ──
    n_models = len(all_models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True)
    for ax, m in zip(axes, all_models):
        box_data, box_labels, box_colors = [], [], []
        for run in runs:
            rname = run["name"]
            if rname in auc_data and m in auc_data[rname] and auc_data[rname][m]:
                v   = run.get("stabl_version", "v1").upper()
                pen = "pen." if run.get("use_penalization") else "unpen."
                box_data.append(auc_data[rname][m])
                box_labels.append(f"{v}\n{pen}")
                box_colors.append(palette.get(rname, "gray"))
        if not box_data:
            ax.set_title(m, fontsize=8); continue
        bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for med in bp["medians"]: med.set_color("black")
        ax.set_xticklabels(box_labels, fontsize=7)
        ax.set_title(m, fontsize=8)
        ax.set_ylim(0.3, 1.05)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("ROC AUC (per fold)")
    fig.suptitle("AUC distribution — V1 vs V2, tous modèles", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "AUC" / "AUC_boxplots_V1_V2.pdf", dpi=150)
    plt.close(fig)
    print("AUC boxplots saved → post_processing/AUC/")

    # ── Jaccard heatmap ──
    _plot_jaccard(feat_sets, out, palette, runs, all_models)

    # ── λ_min (V2 only) ──
    _plot_lambda_min(params, out / "lambda_min")

    print("\nPost-processing complete.")


def _plot_jaccard(feat_sets, out, palette, runs, all_models):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    def jac(a, b):
        return len(a & b) / len(a | b) if (a or b) else 1.0

    labels = list(feat_sets.keys())
    n = len(labels)
    M = np.array([[jac(feat_sets[l1], feat_sets[l2]) for l2 in labels] for l1 in labels])
    pd.DataFrame(M, index=labels, columns=labels).to_csv(out / "jaccard" / "jaccard_similarity.csv")

    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5)
    plt.colorbar(im, ax=ax, label="Jaccard")
    ax.set_title("Feature similarity — V1 vs V2")
    fig.tight_layout()
    fig.savefig(out / "jaccard" / "jaccard_heatmap.pdf", dpi=150)
    plt.close(fig)
    print("Jaccard heatmap saved → post_processing/jaccard/")


def _plot_lambda_min(params, out_dir):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    records = []
    for run in params["runs"]:
        if run.get("stabl_version") != "v2":
            continue
        p = Path(run["save_path"]) / "v2_theory" / "lambda_min.json"
        if not p.exists(): continue
        for layer, s in _read_json(p).items():
            records.append({"run": run["name"], "layer": layer, **s})

    if not records:
        print("  [lambda_min] No data found — skipping.")
        return

    df = pd.DataFrame(records)
    layers = df["layer"].unique()
    x = np.arange(len(layers))
    w = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        idx = list(layers).index(row["layer"])
        ax.bar(idx - w/2 + i*0.05, row["lambda_min_empirical"], w*0.8,
               color="tomato", alpha=0.6, label="Σ_emp" if i==0 else "")
        ax.bar(idx + w/2 + i*0.05, row["lambda_min_LW"], w*0.8,
               color="steelblue", alpha=0.8, label="Σ_LW" if i==0 else "")

    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_ylabel("λ_min"); ax.set_title("λ_min(Σ_emp) vs λ_min(Σ_LW) — n < p")
    ax.axhline(0, color="black", lw=0.8)
    handles = [plt.Rectangle((0,0),1,1,fc="tomato",  alpha=0.6),
               plt.Rectangle((0,0),1,1,fc="steelblue",alpha=0.8)]
    ax.legend(handles, ["Σ_empirical (dégénère)", "Σ_LW (non-dégénéré)"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_min.pdf", dpi=150)
    plt.close(fig)
    print("λ_min plot saved → post_processing/lambda_min/")


# ── Mode 2 : theory analysis (variance, FDP+, gaussianity) ────────────────────


# ── Mode 2 : theory analysis (variance, FDP+, gaussianity) — 3 layers ─────────

def run_theory_analysis(params_path: str = "./params.json") -> None:
    """
    Runs on all 3 layers (Density, Function, Neighbor) — même filtrage que run_experiment penalized.
    Pour chaque layer :
      1. Variance V1 vs V2 — B=[50,100,500,1000,5000,10000], K_SEEDS seeds, 3 estimateurs
      2. FDP+ theory — ε_B Hoeffding + Bernstein, B_0, |∂|/D, sigma_max
      3. Gaussianité — KS marginal, condition ε < η_t/2
    CSV unifié global avec colonne layer.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    from itertools import combinations as _comb
    from sklearn.base import clone
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, FunctionTransformer, QuantileTransformer
    from scipy.stats import ks_1samp, norm as sp_norm, probplot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stabl.stabl   import Stabl as StablV1
    from stabl.stablV2 import Stabl as StablV2
    from stabl.preprocessing import LowInfoFilter, CorrelationFilter
    from stabl.adaptive import ALasso
    from xgboost import XGBClassifier as _XGB

    params           = _read_json(params_path)
    data_path        = params["data_path"]
    g                = params["general"]
    RANDOM_STATE     = g.get("random_state", 42)
    N_JOBS           = g.get("n_jobs", -1)
    DENSITY_THRESHOLD = g.get("density_threshold", 10)

    B_VALUES     = [50, 100, 200, 500, 700, 1000, 3000, 5000, 7000, 10000, 30000, 50000]
    K_SEEDS      = 5
    N_LAMBDA     = 10
    DELTA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3]
    def _t_star_opt_constrained(m, scores, eps_B, eps_tv=0.0, t_min=0.0):
        """argmin_{t : ∂+(t)=∅} FDP+(t)  — contrainte Hoeffding.
        ∂+(t) = {j : score(j) ∈ (t, t + 2*(ε_B+ε_tv)]}  ; feasible ⟺ bdry == 0.
        Retourne (t_star, fdp_value, eta_t, exact).
        """
        eps_total = eps_B + eps_tv
        thresh = np.array(m.fdr_threshold_range)
        fdrs   = np.array(m.FDRs_)
        mask   = thresh >= t_min
        if not mask.any():
            return float(thresh[-1]), float("inf"), 0.0, False
        thresh_m = thresh[mask]
        fdrs_m   = fdrs[mask]

        objs     = np.empty(len(thresh_m))
        feasible = np.empty(len(thresh_m), dtype=bool)
        etas     = np.empty(len(thresh_m))
        for i, t in enumerate(thresh_m):
            above       = scores[scores > t]
            etas[i]     = float(above.min() - t) if len(above) > 0 else float(1.0 - t)
            D           = max(1, len(above))
            bdry        = int(np.sum((scores > t) & (scores <= t + 2 * eps_total)))
            objs[i]     = fdrs_m[i] + bdry / D
            feasible[i] = bdry == 0

        if feasible.any():
            sub_objs = np.where(feasible, objs, np.inf)
            best_idx = int(np.where(sub_objs == sub_objs.min())[0][0])
            return float(thresh_m[best_idx]), float(fdrs_m[best_idx]), float(etas[best_idx]), True

        best_idx = int(np.where(objs == objs.min())[0][0])
        return float(thresh_m[best_idx]), float(objs[best_idx]), float(etas[best_idx]), False

    def _bernstein_eps_featurewise(sigma2_vec, B, log_t):
        """Maurer-Pontil empirical Bernstein eps_j par feature (p,).
        Utilise B-1 et 7/3 car sigma2 estimé sur les mêmes bootstraps."""
        B_eff = max(B - 1, 1)
        disc  = (7/3)**2 * log_t**2 + 8 * B_eff * log_t * sigma2_vec
        return ((7/3) * log_t + np.sqrt(np.maximum(disc, 0.0))) / (2 * B_eff)

    def _t_star_opt_bernstein(m, scores, sigma2_vec, B, log_t, sigma2_ko_vec=None, eps_tv=0.0, t_min=0.0):
        """argmin_{t : ∂+(t)=∅} FDP+(t)  — contrainte Bernstein FW feature-wise.
        ∂+(t) = {j : score(j) ∈ (t, t + ε_j + ε_ko_j + ε_tv]}  ; feasible ⟺ bdry == 0.
        Retourne (t_star, fdp_value, min_margin, exact).
        """
        eps_j = _bernstein_eps_featurewise(sigma2_vec, B, log_t)
        if sigma2_ko_vec is None:
            sigma2_ko_vec = sigma2_vec
        eps_ko_j = _bernstein_eps_featurewise(sigma2_ko_vec, B, log_t)
        eps_tot  = eps_j + eps_ko_j + eps_tv

        thresh = np.array(m.fdr_threshold_range)
        fdrs   = np.array(m.FDRs_)
        mask   = thresh >= t_min
        if not mask.any():
            return float(thresh[-1]), float("inf"), 0.0, False
        thresh_m = thresh[mask]
        fdrs_m   = fdrs[mask]

        feasible = np.empty(len(thresh_m), dtype=bool)
        for i, t in enumerate(thresh_m):
            bdry        = int(np.sum((scores > t) & (scores <= t + eps_tot)))
            feasible[i] = bdry == 0

        if feasible.any():
            sub_fdrs = np.where(feasible, fdrs_m, np.inf)
            best_idx = int(np.where(sub_fdrs == sub_fdrs.min())[0][0])
            t_star   = float(thresh_m[best_idx])
            sel      = scores > t_star
            margin   = float(np.min(scores[sel] - t_star - eps_tot[sel])) if sel.any() else 0.0
            return t_star, float(fdrs_m[best_idx]), margin, True

        best_idx = int(np.where(fdrs_m == fdrs_m.min())[0][0])
        t_star   = float(thresh_m[best_idx])
        sel      = scores > t_star
        margin   = float(np.min(scores[sel] - t_star - eps_tot[sel])) if sel.any() else 0.0
        return t_star, float(fdrs_m[best_idx]), margin, False

    def _t_star_unconstrained(m, t_min=0.0):
        """argmin_{t >= t_min} FDP+(t)  — sans contrainte ni terme de barrière."""
        thresh = np.array(m.fdr_threshold_range)
        fdrs   = np.array(m.FDRs_)
        mask   = thresh >= t_min
        if not mask.any():
            return float(thresh[-1]), float(fdrs[-1])
        thresh_m = thresh[mask]
        fdrs_m   = fdrs[mask]
        best_idx = int(np.where(fdrs_m == fdrs_m.min())[0][0])
        return float(thresh_m[best_idx]), float(fdrs_m[best_idx])

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

    out = Path("post_processing")
    for _sub in ["variance", "fdp_theory", "gaussianity", "synthesis"]:
        _d = out / _sub
        if _d.exists():
            shutil.rmtree(_d)
        _d.mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess (même filtrage que run_experiment penalized) ─────────
    X_density_raw, X_function_raw, X_neighbor_raw, pen_matrix = load_data(pd, data_path)
    y = np.array([0 if x.startswith("HG") else 1 for x in X_density_raw.index])
    X_density_f, X_fn_pen, _, X_nb = preprocess(
        X_density_raw, X_function_raw, X_neighbor_raw, pen_matrix, DENSITY_THRESHOLD
    )

    def _make_density_pipe():
        return Pipeline([
            ("to_million", FunctionTransformer(lambda X: X * 1e6, feature_names_out="one-to-one")),
            ("variance",   VarianceThreshold(0.01)),
            ("corr",       CorrelationFilter(threshold="auto")),
            ("lif",        LowInfoFilter()),
            ("impute",     SimpleImputer(strategy="median")),
            ("std",        StandardScaler()),
        ])

    def _make_noisy_pipe():
        return Pipeline([
            ("variance", VarianceThreshold(0.01)),
            ("lif",      LowInfoFilter()),
            ("impute",   SimpleImputer(strategy="median")),
            ("std",      StandardScaler()),
        ])

    layers = {
        "Density":  (X_density_f, _make_density_pipe),
        "Function": (X_fn_pen,    _make_noisy_pipe),
        "Neighbor": (X_nb,        _make_noisy_pipe),
    }

    BASE_ESTIMATORS = {
        "lasso":    LogisticRegression(penalty="l1", solver="liblinear",
                                       class_weight="balanced", max_iter=int(1e6)),
        "alasso":   ALasso(tol=1e-3),
        "en":       ElasticNet(l1_ratio=0.5, tol=1e-3),
        "xgboost":  _XGB(n_jobs=1, eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE),
    }
    # XGBoost ne supporte pas lambda_grid="auto" (pas un modèle pénalisé linéaire)
    LAMBDA_GRIDS = {
        "lasso":   "auto",
        "alasso":  "auto",
        "en":      "auto",
        "xgboost": {"n_estimators": [50, 100, 200], "max_depth": [3, 5]},
    }
    colors_est = {"lasso": "#C41E3A", "alasso": "#001A7B", "en": "#2CA02C", "xgboost": "#FF7F0E"}

    _ls_cycle = ["-", "--", ":", "-.", (0,(3,1,1,1)), (0,(5,2)), (0,(1,1))]
    _mk_cycle = ["o", "s", "^", "D", "v", "P", "X"]
    _c_cycle  = ["#C41E3A","#001A7B","#2CA02C","#FF7F0E","#9467BD","#8C564B","#E377C2"]
    ls_delta  = {d: _ls_cycle[i % len(_ls_cycle)] for i, d in enumerate(DELTA_VALUES)}
    mk_delta  = {d: _mk_cycle[i % len(_mk_cycle)] for i, d in enumerate(DELTA_VALUES)}
    c_delta   = {d: _c_cycle[i  % len(_c_cycle)]  for i, d in enumerate(DELTA_VALUES)}

    def _t_star_constrained(m):
        return m.fdr_min_threshold_

    rng_seeds = np.random.default_rng(RANDOM_STATE).integers(0, 2**31, size=K_SEEDS).tolist()

    all_synthesis = []

    # ── Boucle sur les 3 layers ────────────────────────────────────────────────
    for layer_name, (X_df, pipe_factory) in layers.items():
        print(f"\n{'='*65}")
        print(f"LAYER : {layer_name.upper()}")
        print(f"{'='*65}")

        for sub in ["variance", "fdp_theory", "gaussianity"]:
            (out / sub / layer_name).mkdir(parents=True, exist_ok=True)

        # Preprocessing (pas de gaussianisation)
        X = pipe_factory().fit_transform(X_df.values, y)
        n, p = X.shape
        print(f"  Shape après preprocessing : n={n}, p={p}")

        # ε_tv : 0 si on suppose X Gaussien (hypothèse théorique), KS sinon
        GAUSSIAN = True    # True → cas Gaussien (eps_tv=0), False → cas non-Gaussien (eps_tv via KS)
        if GAUSSIAN:
            eps_tv = 0.0
            print(f"  ε_tv = 0.0 (hypothèse Gaussienne)")
        else:
            from scipy.stats import ks_1samp, norm as sp_norm
            ks_stats = np.array([ks_1samp(X[:, j], sp_norm.cdf).statistic for j in range(p)])
            eps_tv   = float(ks_stats.mean())
            print(f"  ε_tv (mean KS) = {eps_tv:.5f} | ε_tv_max = {ks_stats.max():.5f}")

        # ── 1. Variance V1 vs V2 ──────────────────────────────────────────────
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
                        lg = LAMBDA_GRIDS[est_name]
                        m = StablClass(
                            base_estimator=clone(base_est),
                            lambda_grid=lg, n_lambda=N_LAMBDA if lg == "auto" else None,
                            n_bootstraps=B, artificial_type="knockoff",
                            n_jobs=N_JOBS, random_state=int(seed)
                        )
                        m.fit(X, y)
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
        fig.savefig(out / "variance" / layer_name / "variance_vs_B.pdf", dpi=150)
        plt.close(fig)
        pd.DataFrame(rows_var).to_csv(out / "variance" / layer_name / "variance_vs_B.csv", index=False)

        # Jaccard inter-seeds
        def _jac(a, b): return len(a & b) / len(a | b) if (a or b) else 1.0

        B_fixed = max(B_VALUES)
        EST_REF = next(iter(BASE_ESTIMATORS))
        selections = {}
        for version in ["v1", "v2"]:
            for k, (sc, thr) in enumerate(zip(scores_all[EST_REF][version][B_fixed],
                                              thresholds_all[EST_REF][version][B_fixed])):
                selections[(version, k)] = (np.array(sc) >= thr).astype(int)

        jac_records = []
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        for i, (version, color) in enumerate([("v1","#C41E3A"), ("v2","#001A7B")]):
            sel_sets = [set(np.where(selections[(version, k)])[0]) for k in range(K_SEEDS)]
            jac_vals = [_jac(sel_sets[a], sel_sets[b]) for a, b in _comb(range(K_SEEDS), 2)]
            jm, js   = float(np.mean(jac_vals)), float(np.std(jac_vals))
            print(f"  {version.upper()} Jaccard inter-seeds : {jm:.3f} ± {js:.3f}")
            jac_records.append({"layer": layer_name, "version": version, "B_fixed": B_fixed,
                                 "jaccard_mean": round(jm,4), "jaccard_std": round(js,4)})
            ax2.bar(i, jm, yerr=js, color=color, alpha=0.8, capsize=6, width=0.5, label=version.upper())
            ax2.scatter([i]*len(jac_vals), jac_vals, color=color, s=40, zorder=5, alpha=0.7)
        ax2.set_xticks([0,1]); ax2.set_xticklabels(["V1","V2"])
        ax2.set_ylabel("Jaccard inter-seeds"); ax2.set_ylim(0, 1.1)
        ax2.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax2.set_title(f"Reproductibilité — {layer_name}\nB={B_fixed}, {K_SEEDS} seeds")
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        fig2.savefig(out / "variance" / layer_name / f"jaccard_inter_seeds_B{B_fixed}.pdf", dpi=150)
        plt.close(fig2)
        pd.DataFrame(jac_records).to_csv(
            out / "variance" / layer_name / "jaccard_inter_seeds.csv", index=False)
        print(f"  Variance → {out / 'variance' / layer_name}/")

        # ── 2. FDP+ theory ────────────────────────────────────────────────────
        print(f"\n{'─'*55}")
        print(f"  FDP+ THEORY — fit V2 par B")
        print(f"{'─'*55}")

        fdp_per_est_B = {}
        for est_name, base_est in BASE_ESTIMATORS.items():
            print(f"\n  Estimateur : {est_name}")
            for B in B_VALUES:
                lg = LAMBDA_GRIDS[est_name]
                m = StablV2(
                    base_estimator=clone(base_est),
                    lambda_grid=lg, n_lambda=N_LAMBDA if lg == "auto" else None,
                    n_bootstraps=B, artificial_type="knockoff",
                    n_jobs=N_JOBS, random_state=RANDOM_STATE
                )
                m.fit(X, y)
                scores   = m.get_importances()
                m.fdr_threshold_range = np.sort(np.unique(scores))
                m._compute_FDPplus()
                K_lambda = m.stabl_scores_.shape[1]
                sigma2   = m.score_variance_.max(axis=1)          # (p,)
                p_       = sigma2.shape[0]
                ko_sigma2 = (m.ko_score_variance_.max(axis=1)
                             if m.ko_score_variance_ is not None and m.ko_score_variance_.shape[0] == p_
                             else np.full(p_, 0.25))
                fdp_per_est_B[(est_name, B)] = {
                    "scores": scores, "K_lambda": K_lambda,
                    "sigma2": sigma2, "ko_sigma2": ko_sigma2, "m": m,
                }
                t_diag = _t_star_constrained(m)
                print(f"    B={B:5d} | t*_diag={t_diag:.4f} | D={max(1,int(np.sum(scores>t_diag)))} | K={K_lambda}")

        records_fdp = []
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        for est_name in BASE_ESTIMATORS:
            c = colors_est[est_name]
            for delta in DELTA_VALUES:
                eps_pts      = []
                bern_pts     = []
                bdry_hoeff   = []
                bdry_bern_fw = []
                for B in B_VALUES:
                    d          = fdp_per_est_B[(est_name, B)]
                    log_t      = np.log(4 * p * d["K_lambda"] / delta)
                    eps_B      = float(np.sqrt(log_t / (2 * B)))
                    sigma2_vec = d["sigma2"]

                    # ── Sans contrainte
                    t_star_unc, fdp_unc = _t_star_unconstrained(d["m"])
                    D_unc = max(1, int(np.sum(d["scores"] > t_star_unc)))
                    eps_fw_pre    = _bernstein_eps_featurewise(sigma2_vec, B, log_t)
                    eps_ko_fw_pre = _bernstein_eps_featurewise(d["ko_sigma2"], B, log_t)
                    eps_tot_fw_pre = eps_fw_pre + eps_ko_fw_pre + eps_tv
                    eps_total      = eps_B + eps_tv
                    res_unc_h   = int(np.sum((d["scores"] > t_star_unc) &
                                             (d["scores"] <= t_star_unc + 2*eps_total))) / D_unc
                    res_unc_bfw = int(np.sum((d["scores"] > t_star_unc) &
                                             (d["scores"] <= t_star_unc + eps_tot_fw_pre))) / D_unc

                    # ── Hoeffding contraint
                    t_star_h, fdp_h, eta_t_h, exact_h = _t_star_opt_constrained(
                        d["m"], d["scores"], eps_B, eps_tv=eps_tv)
                    D_h = max(1, int(np.sum(d["scores"] > t_star_h)))

                    # ── Bernstein FW contraint
                    t_star_bfw, fdp_bfw, margin_bfw, exact_bfw = _t_star_opt_bernstein(
                        d["m"], d["scores"], sigma2_vec, B, log_t,
                        sigma2_ko_vec=d["ko_sigma2"], eps_tv=eps_tv)
                    D_bfw = max(1, int(np.sum(d["scores"] > t_star_bfw)))
                    eps_fw     = eps_fw_pre
                    eps_ko_fw  = eps_ko_fw_pre
                    eps_fw_mean = float(eps_fw.mean())

                    # ── Hoeffding composite (sans contrainte) : argmin FDP+(t) + bdry_H/D
                    t_star_h_comp, obj_h_comp = _t_star_composite_hoeff(
                        d["m"], d["scores"], eps_B, eps_tv=eps_tv)
                    D_h_comp = max(1, int(np.sum(d["scores"] > t_star_h_comp)))
                    bdry_h_comp = int(np.sum((d["scores"] > t_star_h_comp) &
                                             (d["scores"] <= t_star_h_comp + 2*(eps_B + eps_tv))))
                    bdry_r_h_comp = bdry_h_comp / D_h_comp

                    # ── Bernstein FW composite (sans contrainte) : argmin FDP+(t) + bdry_BFW/D
                    t_star_bfw_comp, obj_bfw_comp = _t_star_composite_bfw(
                        d["m"], d["scores"], sigma2_vec, B, log_t,
                        sigma2_ko_vec=d["ko_sigma2"], eps_tv=eps_tv)
                    D_bfw_comp = max(1, int(np.sum(d["scores"] > t_star_bfw_comp)))
                    eps_tot_fw = eps_tot_fw_pre
                    bdry_bfw_comp = int(np.sum((d["scores"] > t_star_bfw_comp) &
                                               (d["scores"] <= t_star_bfw_comp + eps_tot_fw)))
                    bdry_r_bfw_comp = bdry_bfw_comp / D_bfw_comp

                    eps_pts.append(eps_B)
                    bern_pts.append(eps_fw_mean)
                    bdry_hoeff.append(res_unc_h)
                    bdry_bern_fw.append(res_unc_bfw)

                    records_fdp.append({
                        "layer": layer_name, "estimateur": est_name,
                        "delta": delta, "B": B,
                        "eps_B_hoeffding": round(eps_B, 5),
                        "eps_fw_mean":     round(eps_fw_mean, 5),
                        "t_star_unc":        round(t_star_unc, 4),
                        "D_unc":             D_unc,
                        "fdp_unc":           round(fdp_unc, 4),
                        "residual_unc_hoeff": round(res_unc_h, 4),
                        "residual_unc_bfw":   round(res_unc_bfw, 4),
                        "t_star_hoeffding":    round(t_star_h, 4),
                        "D_hoeffding":         D_h,
                        "fdp_hoeffding":       round(fdp_h, 4),
                        "exact_hoeffding":     exact_h,
                        "D_loss_hoeff_vs_unc": D_h - D_unc,
                        "t_star_bern_fw":     round(t_star_bfw, 4),
                        "D_bern_fw":          D_bfw,
                        "fdp_bern_fw":        round(fdp_bfw, 4),
                        "exact_bern_fw":      exact_bfw,
                        "min_margin_bern_fw": round(margin_bfw, 5),
                        "D_loss_bern_vs_unc": D_bfw - D_unc,
                        "D_gain_bern_vs_hoeff": D_bfw - D_h,
                        "t_star_hoeff_comp":   round(t_star_h_comp, 4),
                        "D_hoeff_comp":        D_h_comp,
                        "obj_hoeff_comp":      round(obj_h_comp, 4),
                        "bdry_r_hoeff_comp":   round(bdry_r_h_comp, 4),
                        "D_diff_hoeff_comp_vs_constr": D_h_comp - D_h,
                        "t_star_bfw_comp":     round(t_star_bfw_comp, 4),
                        "D_bfw_comp":          D_bfw_comp,
                        "obj_bfw_comp":        round(obj_bfw_comp, 4),
                        "bdry_r_bfw_comp":     round(bdry_r_bfw_comp, 4),
                        "D_diff_bfw_comp_vs_constr": D_bfw_comp - D_bfw,
                    })
                    if delta == DELTA_VALUES[0]:
                        print(f"      δ={delta} | ε_H={eps_B:.5f} | ε_fw̄={eps_fw_mean:.5f}")
                        print(f"      [Unc]       t*={t_star_unc:.4f} | D={D_unc} | FDP+={fdp_unc:.4f} | "
                              f"résidu_H={res_unc_h:.4f} | résidu_BFW={res_unc_bfw:.4f}")
                        print(f"      [Hoeff ✗]   t*={t_star_h:.4f} | D={D_h} | FDP+={fdp_h:.4f} | "
                              f"exact={exact_h} | ΔD={D_h - D_unc:+d}")
                        print(f"      [BernFW ✗]  t*={t_star_bfw:.4f} | D={D_bfw} | FDP+={fdp_bfw:.4f} | "
                              f"exact={exact_bfw} | ΔD vs unc={D_bfw - D_unc:+d} | ΔD vs H={D_bfw - D_h:+d}")
                        print(f"      [H comp]    t*={t_star_h_comp:.4f} | D={D_h_comp} | obj={obj_h_comp:.4f} | "
                              f"bdry_r={bdry_r_h_comp:.4f} | ΔD vs H={D_h_comp - D_h:+d}")
                        print(f"      [BFW comp]  t*={t_star_bfw_comp:.4f} | D={D_bfw_comp} | obj={obj_bfw_comp:.4f} | "
                              f"bdry_r={bdry_r_bfw_comp:.4f} | ΔD vs BFW={D_bfw_comp - D_bfw:+d}")

                axes[0].plot(B_VALUES, eps_pts,  "o-", color=c, lw=1.5, ls=ls_delta[delta],
                             label=f"{est_name} Hoeffding" if delta == DELTA_VALUES[0] else None)
                axes[0].plot(B_VALUES, bern_pts, "s:", color=c, lw=1,
                             label=f"{est_name} Bern FW" if delta == DELTA_VALUES[0] else None)
                axes[1].plot(B_VALUES, bdry_hoeff,   "o-", color=c, lw=1.5, ls=ls_delta[delta],
                             label=f"{est_name} δ={delta}")
                axes[2].plot(B_VALUES, bdry_bern_fw, "s-", color=c, lw=1.5, ls=ls_delta[delta],
                             label=f"{est_name} δ={delta}")

        axes[0].set(xlabel="B", ylabel="ε", title="ε_H Hoeffding vs ε̄_fw Bernstein FW")
        axes[0].legend(fontsize=7); axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
        for ax, title in zip(axes[1:], [
                "résidu Hoeffding au t* unc : |∂+(t*,2ε_H)|/D",
                "résidu Bernstein FW au t* unc : |∂+(t*,ε_j+ε_ko_j)|/D"]):
            ax.axhline(0, color="black", lw=0.5)
            ax.set(xlabel="B", ylabel="résidu |∂+(t*_unc)|/D", title=title)
            ax.legend(fontsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.suptitle(f"Garanties FDP+ — {layer_name} — p={p}, n={n}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fdp_theory" / layer_name / "fdp_theory_curves.pdf", dpi=150)
        plt.close(fig)

        B_ref     = B_VALUES[-1]
        delta_ref = DELTA_VALUES[0]
        fig2, axes2 = plt.subplots(1, len(BASE_ESTIMATORS), figsize=(14, 4))
        axes2 = np.atleast_1d(axes2)
        for ax2, est_name in zip(axes2, BASE_ESTIMATORS):
            d_ref      = fdp_per_est_B[(est_name, B_ref)]
            log_ref    = np.log(4 * p * d_ref["K_lambda"] / delta_ref)
            eps_ref    = float(np.sqrt(log_ref / (2 * B_ref)))
            sigma2_vec = d_ref["sigma2"]
            eps_fw     = _bernstein_eps_featurewise(sigma2_vec, B_ref, log_ref)
            eps_ko_fw  = _bernstein_eps_featurewise(d_ref["ko_sigma2"], B_ref, log_ref)
            eps_tot_j  = eps_fw + eps_ko_fw + eps_tv
            eps_tot_h  = eps_ref + eps_tv

            t_unc, _   = _t_star_unconstrained(d_ref["m"])
            t_h, _, _, exact_h   = _t_star_opt_constrained(d_ref["m"], d_ref["scores"], eps_ref, eps_tv=eps_tv)
            t_bfw, _, _, exact_bfw = _t_star_opt_bernstein(d_ref["m"], d_ref["scores"], sigma2_vec, B_ref, log_ref,
                                                            sigma2_ko_vec=d_ref["ko_sigma2"], eps_tv=eps_tv)
            sc = d_ref["scores"]
            D_unc  = max(1, int(np.sum(sc > t_unc)))
            D_h    = max(1, int(np.sum(sc > t_h)))
            D_bfw  = max(1, int(np.sum(sc > t_bfw)))
            res_unc_h   = int(np.sum((sc > t_unc) & (sc <= t_unc + 2*eps_tot_h))) / D_unc
            res_unc_bfw = int(np.sum((sc > t_unc) & (sc <= t_unc + eps_tot_j))) / D_unc

            ax2.hist(sc, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
            ax2.axvline(t_unc, color="gray",    lw=1.5, ls=":",
                        label=f"t* unc={t_unc:.3f} (D={D_unc}, résidu_H={res_unc_h:.3f}, BFW={res_unc_bfw:.3f})")
            ax2.axvline(t_h,   color="#C41E3A", lw=2, ls="--",
                        label=f"t* Hoeff={t_h:.3f} ({'✓' if exact_h else '✗'}, D={D_h})")
            ax2.axvline(t_bfw, color="#2CA02C", lw=2, ls="-.",
                        label=f"t* BernFW={t_bfw:.3f} ({'✓' if exact_bfw else '✗'}, D={D_bfw})")
            ax2.axvspan(t_h, t_h + 2*eps_tot_h, alpha=0.12, color="#C41E3A",
                        label=f"barrière Hoeff (2ε={2*eps_tot_h:.4f})")
            eps_fw_above = eps_tot_j[sc > t_bfw]
            bfw_zone = float(eps_fw_above.max()) if len(eps_fw_above) > 0 else 0.0
            ax2.axvspan(t_bfw, t_bfw + bfw_zone, alpha=0.10, color="#2CA02C",
                        label=f"barrière BernFW (max={bfw_zone:.4f})")
            ax2.set(xlabel="Max stability score", ylabel="Nb features",
                    title=(f"{est_name} (B={B_ref}, δ={delta_ref})\n"
                           f"unc D={D_unc} | Hoeff D={D_h} (Δ={D_h-D_unc:+d}) | BernFW D={D_bfw} (Δ={D_bfw-D_unc:+d})"))
            ax2.legend(fontsize=7)
            ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        fig2.suptitle(f"Distribution des scores — {layer_name}", fontsize=10)
        fig2.tight_layout()
        fig2.savefig(out / "fdp_theory" / layer_name / "score_distribution.pdf", dpi=150)
        plt.close(fig2)
        pd.DataFrame(records_fdp).to_csv(
            out / "fdp_theory" / layer_name / "fdp_theory_table.csv", index=False)

        # ── Graphique décomposé : FDP+(t) / D(t) / |∂+|/D(t) / objectif vs t ──
        n_est  = len(BASE_ESTIMATORS)
        fig3, axes3 = plt.subplots(4, n_est, figsize=(max(6, 5 * n_est), 14), sharex="col")
        axes3 = np.array(axes3).reshape(4, n_est)
        _ROW_LABELS = ["FDP⁺(t)", "D(t)", "|∂⁺(t,2ε_H)| / D(t)", "FDP⁺(t) + |∂⁺|/D [Hoeff]"]
        _ROW_COLORS = ["#4A90D9", "#2CA02C", "#FF7F0E", "steelblue"]

        for col, est_name in enumerate(BASE_ESTIMATORS):
            d_o      = fdp_per_est_B[(est_name, B_ref)]
            log_o    = np.log(4 * p * d_o["K_lambda"] / delta_ref)
            eps_o    = float(np.sqrt(log_o / (2 * B_ref)))
            eps_tot  = eps_o + eps_tv
            sigma2_o = d_o["sigma2"]
            eps_fw_o    = _bernstein_eps_featurewise(sigma2_o, B_ref, log_o)
            eps_ko_fw_o = _bernstein_eps_featurewise(d_o["ko_sigma2"], B_ref, log_o)
            eps_tot_fw_o = eps_fw_o + eps_ko_fw_o + eps_tv
            thresh   = np.array(d_o["m"].fdr_threshold_range)
            fdrs_arr = np.array(d_o["m"].FDRs_)
            sc       = d_o["scores"]

            D_t    = np.array([int(np.sum(sc > t)) for t in thresh])
            bdry_t = np.array([int(np.sum((sc > t) & (sc <= t + 2*eps_tot))) for t in thresh])
            bdry_D = bdry_t / np.maximum(D_t, 1)
            objs   = fdrs_arr + bdry_D
            feasible     = bdry_t == 0
            feasible_bfw = np.array([
                np.sum((sc > t) & (sc <= t + eps_tot_fw_o)) == 0 for t in thresh
            ])

            # Courbe de barrière BernFW : |∂+(t, ε_j+ε_ko_j)| / D(t)
            bdry_bfw_t = np.array([int(np.sum((sc > t) & (sc <= t + eps_tot_fw_o))) for t in thresh])
            bdry_bfw_D = bdry_bfw_t / np.maximum(D_t, 1)

            t_unc_o, _  = _t_star_unconstrained(d_o["m"])
            t_star_o, _, _, exact_o     = _t_star_opt_constrained(d_o["m"], sc, eps_o, eps_tv=eps_tv)
            t_star_bfw_o, _, _, exact_bfw_o = _t_star_opt_bernstein(
                d_o["m"], sc, sigma2_o, B_ref, log_o,
                sigma2_ko_vec=d_o["ko_sigma2"], eps_tv=eps_tv)
            t_star_h_comp_o, _ = _t_star_composite_hoeff(d_o["m"], sc, eps_o, eps_tv=eps_tv)
            t_star_bfw_comp_o, _ = _t_star_composite_bfw(
                d_o["m"], sc, sigma2_o, B_ref, log_o,
                sigma2_ko_vec=d_o["ko_sigma2"], eps_tv=eps_tv)

            # Valeurs des barrières au t* unconstrained
            matches = np.where(thresh == t_unc_o)[0]
            idx_unc = int(matches[0]) if len(matches) > 0 else int(np.argmin(np.abs(thresh - t_unc_o)))
            res_h_unc   = float(bdry_D[idx_unc])
            res_bfw_unc = float(bdry_bfw_D[idx_unc])

            curves = [fdrs_arr, D_t.astype(float), bdry_D, objs]
            for row, (curve, color, ylabel) in enumerate(zip(curves, _ROW_COLORS, _ROW_LABELS)):
                ax = axes3[row, col]
                ax.step(thresh, curve, where="post", color=color, lw=1.5)
                ylo = curve.min(); yhi = curve.max()
                pad = (yhi - ylo) * 0.12 if yhi > ylo else 0.05
                ax.set_ylim(ylo-pad, yhi+pad)
                _lbl_h   = "faisable Hoeff"   if row == 0 else None
                _lbl_bfw = "faisable BernFW"  if row == 0 else None
                for t_f in thresh[feasible]:
                    ax.axvspan(t_f, t_f + 2 * eps_tot, alpha=0.18, color="#4A90D9",
                               label=_lbl_h, zorder=0)
                    _lbl_h = None
                    ax.axvline(t_f, ymin=0, ymax=0.08, color="#4A90D9", lw=1.2, alpha=0.9)
                bfw_win = float(eps_tot_fw_o.max())
                for t_f in thresh[feasible_bfw]:
                    ax.axvspan(t_f, t_f + bfw_win, alpha=0.18, color="#2CA02C",
                               label=_lbl_bfw, zorder=0)
                    _lbl_bfw = None
                    ax.axvline(t_f, ymin=0, ymax=0.08, color="#2CA02C", lw=1.2, alpha=0.9)
                ax.axvline(t_unc_o,          color="gray",    lw=1,   ls=":",
                           label=f"t* unc={t_unc_o:.3f}" if row == 0 else None)
                ax.axvline(t_star_o,          color="#C41E3A", lw=1.5, ls="--",
                           label=f"t* Hoeff contraint={t_star_o:.3f} ({'✓' if exact_o else '✗'})" if row == 0 else None)
                ax.axvline(t_star_bfw_o,      color="#2CA02C", lw=1.5, ls="-.",
                           label=f"t* BernFW contraint={t_star_bfw_o:.3f} ({'✓' if exact_bfw_o else '✗'})" if row == 0 else None)
                ax.axvline(t_star_h_comp_o,   color="#4A90D9", lw=1.2, ls="--",
                           label=f"t* Hoeff composite={t_star_h_comp_o:.3f}" if row == 0 else None)
                ax.axvline(t_star_bfw_comp_o, color="#85C785", lw=1.2, ls="--",
                           label=f"t* BernFW composite={t_star_bfw_comp_o:.3f}" if row == 0 else None)

                # Panneau 2 : superposer BernFW + annoter les deux barrières au t* unc
                if row == 2:
                    ax.step(thresh, bdry_bfw_D, where="post",
                            color="#2CA02C", lw=1.2, ls="--", alpha=0.85,
                            label="|∂⁺|/D BernFW")
                    ax.plot(t_unc_o, res_h_unc,   "o", color="#4A90D9", ms=7, zorder=5,
                            label=f"Hoeff au t*_unc : {res_h_unc:.3f}")
                    ax.plot(t_unc_o, res_bfw_unc, "s", color="#2CA02C", ms=7, zorder=5,
                            label=f"BernFW au t*_unc : {res_bfw_unc:.3f}")
                    ax.legend(fontsize=6, loc="upper right")

                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                if row == 0:
                    ax.set_title(f"{est_name}\nB={B_ref}, δ={delta_ref}, "
                                 f"ε_H={eps_o:.4f} | ε_fw̄={float(eps_fw_o.mean()):.4f}", fontsize=8)
                    ax.legend(fontsize=7, loc="upper right")
                if col == 0: ax.set_ylabel(ylabel, fontsize=8)
                if row == 3: ax.set_xlabel("t", fontsize=8)

        fig3.suptitle(f"Décomposition FDP⁺ — {layer_name} — ε_tv={eps_tv:.5f}  "
                      f"(bleu clair = faisable Hoeffding, vert = faisable BernFW | "
                      f"rouge-- = Hoeff contraint, vert-. = BernFW contraint, "
                      f"bleu-- = Hoeff composite, vert clair-- = BernFW composite)", fontsize=8)
        fig3.tight_layout()
        fig3.savefig(out / "fdp_theory" / layer_name / "objective_function.pdf", dpi=150)
        plt.close(fig3)
        print(f"  FDP+ theory → {out / 'fdp_theory' / layer_name}/")

        # ── 3. Gaussianité ────────────────────────────────────────────────────
        for est_name in BASE_ESTIMATORS:
            gauss_dir = out / "gaussianity" / layer_name / est_name
            gauss_dir.mkdir(parents=True, exist_ok=True)
            _compute_gaussianity(X, fdp_per_est_B, est_name,
                                 B_VALUES[-1], p, DELTA_VALUES, gauss_dir,
                                 _t_star_opt_constrained, eps_tv)

        # ── Accumuler pour synthesis globale ──────────────────────────────────
        var_lookup = {(r["estimateur"], r["B"], r["version"]): (r["var_mean"], r["var_std"])
                      for r in rows_var}
        for row in records_fdp:
            est, B = row["estimateur"], row["B"]
            v1m, v1s = var_lookup.get((est, B, "v1"), (float("nan"), float("nan")))
            v2m, v2s = var_lookup.get((est, B, "v2"), (float("nan"), float("nan")))
            all_synthesis.append({**row,
                "var_v1_mean": round(v1m,7) if v1m==v1m else float("nan"),
                "var_v1_std":  round(v1s,7) if v1s==v1s else float("nan"),
                "var_v2_mean": round(v2m,7) if v2m==v2m else float("nan"),
                "var_v2_std":  round(v2s,7) if v2s==v2s else float("nan"),
            })

    # ── CSV unifié global (Density + Function + Neighbor) ─────────────────────
    df_unified = pd.DataFrame(all_synthesis)
    df_unified.to_csv(out / "synthesis" / "theory_analysis.csv", index=False)
    print(f"\nCSV unifié → {out / 'synthesis' / 'theory_analysis.csv'}")
    print(df_unified.to_string(index=False))
    print("\nTheory analysis complete.")

def _compute_gaussianity(X, fdp_per_est_B, est_name, B_ref, p, DELTA_VALUES, out_dir,
                         t_star_func=None, eps_tv=0.0):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import ks_1samp, norm as sp_norm, probplot

    n, _ = X.shape
    ks_stats  = np.array([ks_1samp(X[:, j], sp_norm.cdf).statistic for j in range(p)])
    eps_proxy = float(ks_stats.mean())
    eps_max   = float(ks_stats.max())

    d = fdp_per_est_B[(est_name, B_ref)]

    rows_gauss = []
    for delta in DELTA_VALUES:
        log_t     = np.log(4 * p * d["K_lambda"] / delta)
        eps_B     = float(np.sqrt(log_t / (2 * B_ref)))
        eps_total = eps_B + eps_tv
        if t_star_func is not None:
            _, _, eta_t, exact = t_star_func(d["m"], d["scores"], eps_B, eps_tv=eps_tv)
            cond = exact
        else:
            eta_t = d.get("eta_t", 0.0)
            cond  = bool(2 * eps_total < eta_t)
        rows_gauss.append({"delta": delta, "eta_t": round(eta_t, 5),
                            "eps_B": round(eps_B, 5), "eps_tv": round(eps_tv, 5),
                            "eps_total": round(eps_total, 5),
                            "condition (2*(eps_B+eps_tv) < eta_t)": cond,
                            "marge": round(eta_t - 2*eps_total, 5)})

    # pour les prints et le graphique, on utilise delta = DELTA_VALUES[0]
    log_ref   = np.log(4 * p * d["K_lambda"] / DELTA_VALUES[0])
    eps_ref   = float(np.sqrt(log_ref / (2 * B_ref)))
    eps_ref_total = eps_ref + eps_tv
    if t_star_func is not None:
        _, _, eta_t, _ = t_star_func(d["m"], d["scores"], eps_ref, eps_tv=eps_tv)
    else:
        eta_t = d.get("eta_t", 0.0)

    condition_mean = eps_proxy < eta_t / 2
    condition_max  = eps_max   < eta_t / 2
    print(f"\n── Gaussianité ─────────────────────────────────────────────")
    print(f"  η_t={eta_t:.4f} | ε_tv={eps_tv:.5f} | ε_proxy(mean KS)={eps_proxy:.4f} | ε_max={eps_max:.4f}")
    print(f"  ε_proxy < η_t/2 : {condition_mean} | ε_max < η_t/2 : {condition_max}")

    _ls = ["-","--",":","-.",(0,(3,1,1,1)),(0,(5,2)),(0,(1,1))]
    _c  = ["#C41E3A","#001A7B","#2CA02C","#FF7F0E","#9467BD","#8C564B","#E377C2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ks_stats, bins=25, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(eps_proxy, color="#C41E3A", lw=2, ls="--",
                    label=f"ε_proxy={eps_proxy:.4f}")
    axes[0].axvline(eta_t/2,   color="orange",  lw=2, ls="-.",
                    label=f"η_t/2={eta_t/2:.4f}")
    axes[0].set_xlabel("KS statistic"); axes[0].set_ylabel("Nb features")
    axes[0].set_title("Distribution des KS marginaux"); axes[0].legend(fontsize=8)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    worst_idx = np.argsort(ks_stats)[-3:]
    best_idx  = np.argsort(ks_stats)[:3]
    for k, (idx, c) in enumerate(zip(worst_idx, ["#C41E3A","#FF7F7F","#FFAAAA"])):
        (osm, osr), _ = probplot(X[:, idx])
        axes[1].plot(osm, osr, "o", color=c, ms=4, alpha=0.7,
                     label=f"worst KS={ks_stats[idx]:.3f}" if k==0 else "")
    for k, (idx, c) in enumerate(zip(best_idx, ["#001A7B","#6B8CFF","#AAC4FF"])):
        (osm, osr), _ = probplot(X[:, idx])
        axes[1].plot(osm, osr, "o", color=c, ms=4, alpha=0.7,
                     label=f"best KS={ks_stats[idx]:.3f}" if k==0 else "")
    lims = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
            max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
    axes[1].plot(lims, lims, "k--", lw=1, alpha=0.5)
    axes[1].set_xlabel("Quantiles théoriques N(0,1)"); axes[1].set_ylabel("Quantiles empiriques")
    axes[1].set_title("QQ-plot (3 meilleures et 3 pires features)")
    axes[1].legend(fontsize=7)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "gaussianity.pdf", dpi=150)
    plt.close(fig)

    pd.DataFrame(rows_gauss).to_csv(out_dir / "gaussianity_summary.csv", index=False)
    pd.DataFrame([{"n": n, "p": p,
                   "eps_proxy_mean_KS": round(eps_proxy,5),
                   "eps_max_KS": round(eps_max,5),
                   "eta_t": round(eta_t,5),
                   "condition_mean_KS": condition_mean,
                   "condition_max_KS": condition_max}]).to_csv(
        out_dir / "gaussianity_ks.csv", index=False)
    print(f"Gaussianité → {out_dir}/")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",      type=int,            help="0=experiment, 1=post-process, 2=theory, 3=parse-params")
    parser.add_argument("idx",       type=int, nargs="?", default=0)
    parser.add_argument("intensity", type=str, nargs="?", default="h")
    parser.add_argument("--params",  type=str, default="./params.json")
    parser.add_argument("--out",     type=str, default="post_processing")
    args = parser.parse_args()

    if args.mode == 0:
        run_experiment(args.idx, args.intensity)
    elif args.mode == 1:
        post_process(args.params, out_dir=args.out)
    elif args.mode == 2:
        run_theory_analysis(args.params)
    elif args.mode == 3:
        parse_params(args.params)
