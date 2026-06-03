"""
test.py — Run local mini version of the synthetic benchmark.
Paramètres réduits pour tourner en ~5-10 min sur un laptop.
"""

import os; os.environ["PYTHONWARNINGS"] = "ignore"
import warnings; warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json, shutil
from pathlib import Path

# ── Paramètres réduits ─────────────────────────────────────────────────────────

PARAMS = {
    "Experiment_Name": "StablV2_Synthetic_test",
    "datasets": [
        # Signal fort, p petit — le plus rapide et le plus lisible
        {"id": 1, "n": 80, "p": 50, "k": 3, "signal": 5.0, "seed": 1},
    ],
    "versions": ["v1", "v2_unc", "v2_constr"],
    "general": {
        "taskType":     "binary",
        "n_bootstraps": 1000,   # réduit pour local — 10000 sur Sherlock
        "n_splits":     3,
        "n_repeats":    2,      # 6 folds local vs 25 sur Sherlock
        "random_state": 42,
        "n_jobs":       -1,
        "cpusHigh":     4,
        "memHighGB":    8,
        "time":         "01:00:00",
    },
}

TEST_DIR = Path(__file__).parent / "test_results"

# ── Import sendOut helpers ─────────────────────────────────────────────────────

from sendOut import (
    generate_dataset,
    _expand_runs,
    _get_stabl_class,
    _stabl_kwargs,
    _make_estimators,
    _import_ml,
    _write_json,
    _read_json,
)


def setup_run_dirs():
    runs = _expand_runs(PARAMS)
    for idx, run in enumerate(runs):
        run_dir = TEST_DIR / f"run_{idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json({**run, **PARAMS["general"], "idx": idx}, run_dir / "params.json")
    print(f"Created {len(runs)} run dirs in {TEST_DIR}/")
    return runs


def run_experiment(idx):
    (np, pd, ElasticNet, LogisticRegression, GridSearchCV,
     RepeatedStratifiedKFold, StratifiedKFold, clone,
     StablV1, StablV2, ALasso, ALogitLasso, multi_omic_stabl_cv,
     Pipeline, StandardScaler, XGBClassifier) = _import_ml()

    p       = _read_json(TEST_DIR / f"run_{idx}" / "params.json")
    version = p["version"]
    ds      = p["dataset"]
    print(f"\nRun {idx}: dataset_{ds['id']} | version={version} | "
          f"n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']}")

    StablClass = _get_stabl_class(version, StablV1, StablV2)
    stabl_kw   = _stabl_kwargs(version)

    import stabl.multi_omic_pipelines as _mop_mod
    if version != "v1":
        from stabl.stablV2 import save_stabl_results as _ssr
    else:
        from stabl.stabl import save_stabl_results as _ssr
    _mop_mod.save_stabl_results = _ssr

    X_df, y, true_features, _ = generate_dataset(
        ds["n"], ds["p"], ds["k"], ds["signal"], ds["seed"])

    save_path = TEST_DIR / p["save_path"].replace("results/", "")
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

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


def post_process():
    import numpy as np
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc as sk_auc

    runs     = _expand_runs(PARAMS)
    datasets = PARAMS["datasets"]
    versions = PARAMS["versions"]

    stabl_models = ["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]
    base_models  = ["Lasso", "ALasso", "ElasticNet", "XGBoost"]
    all_models   = stabl_models + base_models
    palette      = {"v1": "#C41E3A", "v2_unc": "#2CA02C", "v2_constr": "#001A7B"}

    out = TEST_DIR / "post_processing"
    for sub in ["ROC", "AUC", "feature_recovery"]:
        d = out / sub
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    auc_data  = {}
    pred_data = {}
    feat_data = {}
    gt_data   = {}

    for run in runs:
        ds_id   = run["dataset"]["id"]
        version = run["version"]
        sp      = TEST_DIR / run["save_path"].replace("results/", "")

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

    # ROC
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
                   title=f"ROC — {m}\nds{ds_id} (n={ds['n']}, p={ds['p']}, k={ds['k']}, sig={ds['signal']})")
            ax.legend(loc="lower right", fontsize=8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(out / "ROC" / f"ROC_ds{ds_id}_{m.replace(' ','_')}.pdf", dpi=150)
            plt.close(fig)
    print("ROC → post_processing/ROC/")

    # AUC boxplots
    for m in all_models:
        fig, ax = plt.subplots(figsize=(6, 5))
        box_data, box_labels, box_colors = [], [], []
        ds = datasets[0]
        for version in versions:
            key = (ds["id"], version, m)
            if key in auc_data and auc_data[key]:
                box_data.append(auc_data[key])
                box_labels.append(version)
                box_colors.append(palette.get(version, "gray"))
        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
            for patch, c in zip(bp["boxes"], box_colors):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            for med in bp["medians"]: med.set_color("black")
            ax.set_xticklabels(box_labels, fontsize=9)
            ax.set_ylim(0.3, 1.05)
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
            ax.set_ylabel("ROC AUC")
            ax.set_title(f"AUC — {m}")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "AUC" / f"AUC_{m.replace(' ','_')}.pdf", dpi=150)
        plt.close(fig)
    print("AUC → post_processing/AUC/")

    # Feature recovery
    recovery_records = []
    for ds in datasets:
        ds_id    = ds["id"]
        true_set = set(gt_data.get(ds_id, []))
        if not true_set: continue
        for version in versions:
            for m in stabl_models:
                sel = feat_data.get((ds_id, version, m), set())
                tp  = len(sel & true_set)
                precision = tp / len(sel) if sel else 0.0
                recall    = tp / len(true_set)
                f1 = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
                recovery_records.append({
                    "version": version, "model": m,
                    "n_selected": len(sel),
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "f1": round(f1, 3),
                })
                print(f"  {version} | {m}: prec={precision:.2f} rec={recall:.2f} "
                      f"F1={f1:.2f} ({len(sel)} selected / {len(true_set)} true)")

    if recovery_records:
        import pandas as pd
        pd.DataFrame(recovery_records).to_csv(
            out / "feature_recovery" / "recovery_table.csv", index=False)
    print("Feature recovery → post_processing/feature_recovery/")


def run_theory_analysis(ds_idx=None):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from stabl.stablV2 import Stabl as StablV2
    from stabl.stabl   import Stabl as StablV1
    from stabl.adaptive import ALasso

    # B_VALUES : sweep complet pour les courbes variance/Jaccard vs B (Claim 1).
    # B_ref    : valeur de référence pour FDP+ calibration et feature recovery.
    # K_STAB   : nb de seeds indépendants pour estimer Var[score_j] inter-runs.
    B_VALUES = [200, 500, 1000, 2000, 5000, 10000]
    B_SWEEP  = B_VALUES
    B_ref    = B_VALUES[-1]   # 10000
    K_STAB   = 20
    DELTA    = 0.05

    out_base = TEST_DIR / "post_processing" / "fdp_theory"
    if ds_idx is None and out_base.exists():
        shutil.rmtree(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    def _bfw_eps(sigma2, B, log_t):
        B_eff = max(B - 1, 1)
        disc  = (7/3)**2 * log_t**2 + 8 * B_eff * log_t * sigma2
        return ((7/3) * log_t + np.sqrt(np.maximum(disc, 0.0))) / (2 * B_eff)

    def _t_unc(m):
        fdrs = np.array(m.FDRs_); thresh = np.array(m.fdr_threshold_range)
        idx  = int(np.where(fdrs == fdrs.min())[0][0])
        return float(thresh[idx]), float(fdrs[idx])

    def _t_constr(m, scores, sv, sk, B, log_t):
        eps  = _bfw_eps(sv, B, log_t) + _bfw_eps(sk, B, log_t)
        thresh = np.array(m.fdr_threshold_range); fdrs = np.array(m.FDRs_)
        feas = np.array([np.sum((scores > t) & (scores <= t + eps)) == 0 for t in thresh])
        if feas.any():
            sub = np.where(feas, fdrs, np.inf)
            idx = int(np.where(sub == sub.min())[0][0])
            return float(thresh[idx]), float(fdrs[idx]), True
        idx = int(np.where(fdrs == fdrs.min())[0][0])
        return float(thresh[idx]), float(fdrs[idx]), False

    BASE_ESTIMATORS = {
        "lasso": LogisticRegression(penalty="l1", solver="liblinear",
                                    class_weight="balanced", max_iter=int(1e6)),
    }

    stab_seeds = np.random.default_rng(42).integers(0, 2**31, K_STAB).tolist()
    est_name, base_est = list(BASE_ESTIMATORS.items())[0]

    datasets = PARAMS["datasets"]
    if ds_idx is not None:
        datasets = [datasets[ds_idx]]

    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"DATASET {ds['id']}  n={ds['n']}  p={ds['p']}  k={ds['k']}  signal={ds['signal']}")
        print(f"{'='*50}")

        out = out_base / f"dataset_{ds['id']}"
        out.mkdir(parents=True, exist_ok=True)

        _sweep_path = out / "scores_all_B.npz"

        # ── Phase 1 : score collection uniquement si cache absent ──────────────
        if not _sweep_path.exists():
            print(f"  Pas de cache — sweep score collection uniquement…")
            X_df, y, true_features, Sigma_true = generate_dataset(
                ds["n"], ds["p"], ds["k"], ds["signal"], ds["seed"])
            X = StandardScaler().fit_transform(X_df.values, y)
            versions_sweep = {
                "v1":       (StablV1, {}),
                "v2unc":    (StablV2, {"selection_mode": "unconstrained", "delta": DELTA, "cov_matrix": Sigma_true}),
                "v2constr": (StablV2, {"selection_mode": "constrained",  "delta": DELTA, "cov_matrix": Sigma_true}),
            }
            sweep_data = {
                "B_values"  : np.array(B_VALUES),
                "feat_names": np.array(X_df.columns.tolist()),
                "true_mask" : np.array([f in set(true_features) for f in X_df.columns]),
                "DELTA"     : np.array([DELTA]),
            }
            for B in B_VALUES:
                for v_name, (VCls, vkw) in versions_sweep.items():
                    m_sw = VCls(base_estimator=clone(base_est), lambda_grid="auto",
                                artificial_type="knockoff", n_bootstraps=B,
                                random_state=42, **vkw)
                    try:
                        m_sw.fit(X, y)
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
                            _m_s = VCls(base_estimator=clone(base_est), lambda_grid="auto",
                                        artificial_type="knockoff", n_bootstraps=B,
                                        random_state=int(_s), **vkw)
                            _m_s.fit(X, y)
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
        true_set  = set(true_features)
        X         = StandardScaler().fit_transform(X_df.values, y)
        feat_names = X_df.columns.tolist()
        true_mask  = np.array([f in true_set for f in X_df.columns])

        _d = np.load(_sweep_path, allow_pickle=True)

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

        for est_name, base_est in BASE_ESTIMATORS.items():
            eps_fw_pts, D_unc_pts, D_constr_pts, bdry_pts = [], [], [], []
            prec_unc_pts, prec_constr_pts = [], []
            rec_unc_pts,  rec_constr_pts  = [], []
            f1_unc_pts,   f1_constr_pts   = [], []

            for B in B_VALUES:
                pfx_unc = f"v2unc_B{B}"
                if f"{pfx_unc}_scores" not in _d.files:
                    continue
                m_b   = _make_proxy(_d, pfx_unc)
                sc_b  = np.max(m_b.stabl_scores_, axis=1)
                log_t_b = max(np.log(4 * m_b.n_features_in_ * m_b.stabl_scores_.shape[1] / DELTA), 0.0)
                sv_b  = m_b.score_variance_.max(axis=1)
                sk_b  = m_b.ko_score_variance_.max(axis=1) if m_b.ko_score_variance_ is not None else sv_b
                eps_j = _bfw_eps(sv_b, B, log_t_b) + _bfw_eps(sk_b, B, log_t_b)

                t_u, _    = _t_unc(m_b)
                t_c, _, _ = _t_constr(m_b, sc_b, sv_b, sk_b, B, log_t_b)

                Du = int(np.sum(sc_b > t_u)); Dc = int(np.sum(sc_b > t_c))
                bdry_ratio = int(np.sum((sc_b > t_u) & (sc_b <= t_u + eps_j))) / max(1, Du)

                sel_u = set(f for f, s in zip(feat_names, sc_b) if s > t_u)
                sel_c = set(f for f, s in zip(feat_names, sc_b) if s > t_c)
                k_    = max(1, ds["k"])
                tp_u  = len(sel_u & true_set); tp_c = len(sel_c & true_set)
                p_u   = tp_u / max(1, len(sel_u)); r_u = tp_u / k_
                p_c   = tp_c / max(1, len(sel_c)); r_c = tp_c / k_
                fu    = 2*p_u*r_u/(p_u+r_u) if p_u+r_u > 0 else 0.0
                fc    = 2*p_c*r_c/(p_c+r_c) if p_c+r_c > 0 else 0.0

                eps_fw_pts.append(float(eps_j.mean())); D_unc_pts.append(Du)
                D_constr_pts.append(Dc); bdry_pts.append(bdry_ratio)
                prec_unc_pts.append(p_u); prec_constr_pts.append(p_c)
                rec_unc_pts.append(r_u);  rec_constr_pts.append(r_c)
                f1_unc_pts.append(fu);    f1_constr_pts.append(fc)

            B_arr = np.array(B_VALUES[:len(eps_fw_pts)])
            fig, axes = plt.subplots(2, 3, figsize=(15, 7))
            axes = axes.flatten()

            axes[0].plot(B_arr, eps_fw_pts, "s-", color="#2CA02C", label="ε̄_fw BernFW")
            axes[0].set(xlabel="B", ylabel="ε", title="ε BernFW vs B"); axes[0].set_xscale("log")
            axes[0].legend(fontsize=8)
            axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

            axes[1].plot(B_arr, D_unc_pts, "o-", color="#C41E3A", label="D(t* unc)")
            axes[1].plot(B_arr, D_constr_pts, "s-", color="#001A7B", label="D(t* constr)")
            axes[1].axhline(ds["k"], color="gray", ls="--", lw=0.8, label=f"k={ds['k']}")
            axes[1].set(xlabel="B", ylabel="D(t*)", title="Features sélectionnées vs B"); axes[1].set_xscale("log")
            axes[1].legend(fontsize=8)
            axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

            axes[2].plot(B_arr, bdry_pts, "o-", color="#2CA02C", label="|∂⁺(t*_unc)|/D")
            axes[2].set(xlabel="B", ylabel="|∂⁺|/D", title="Frontière BernFW au t* unc"); axes[2].set_xscale("log")
            axes[2].legend(fontsize=8)
            axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)

            axes[3].plot(B_arr, prec_unc_pts, "o-", color="#C41E3A", label="t* unc")
            axes[3].plot(B_arr, prec_constr_pts, "s-", color="#001A7B", label="t* constr")
            axes[3].set(xlabel="B", ylabel="Précision", title="Précision vs B", ylim=(-0.05, 1.05))
            axes[3].axhline(1.0, color="gray", ls="--", lw=0.5); axes[3].set_xscale("log")
            axes[3].legend(fontsize=8)
            axes[3].spines["top"].set_visible(False); axes[3].spines["right"].set_visible(False)

            axes[4].plot(B_arr, rec_unc_pts, "o-", color="#C41E3A", label="t* unc")
            axes[4].plot(B_arr, rec_constr_pts, "s-", color="#001A7B", label="t* constr")
            axes[4].set(xlabel="B", ylabel="Rappel", title="Rappel vs B", ylim=(-0.05, 1.05))
            axes[4].axhline(1.0, color="gray", ls="--", lw=0.5); axes[4].set_xscale("log")
            axes[4].legend(fontsize=8)
            axes[4].spines["top"].set_visible(False); axes[4].spines["right"].set_visible(False)

            axes[5].plot(B_arr, f1_unc_pts, "o-", color="#C41E3A", label="t* unc")
            axes[5].plot(B_arr, f1_constr_pts, "s-", color="#001A7B", label="t* constr")
            axes[5].set(xlabel="B", ylabel="F1", title="F1 vs B", ylim=(-0.05, 1.05))
            axes[5].axhline(1.0, color="gray", ls="--", lw=0.5); axes[5].set_xscale("log")
            axes[5].legend(fontsize=8)
            axes[5].spines["top"].set_visible(False); axes[5].spines["right"].set_visible(False)

            fig.suptitle(f"Théorie BernFW — ds{ds['id']} (n={ds['n']}, p={ds['p']}, "
                         f"k={ds['k']}, signal={ds['signal']}) — {est_name}", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"theory_curves_{est_name}.pdf", dpi=150)
            plt.close(fig)
            print(f"Theory → {out}/theory_curves_{est_name}.pdf")

        # Modèles B_ref depuis le cache
        _pfx     = f"B{B_ref}"
        m        = _make_proxy(_d, f"v2unc_{_pfx}")
        m_v1     = _make_proxy(_d, f"v1_{_pfx}")
        m_constr = _make_proxy(_d, f"v2constr_{_pfx}")

        scores_v1     = np.max(m_v1.stabl_scores_, axis=1)
        t_v1          = float(m_v1.fdr_min_threshold_)
        scores_constr = np.max(m_constr.stabl_scores_, axis=1)
        scores        = np.max(m.stabl_scores_, axis=1)
        log_t   = max(np.log(4 * m.n_features_in_ * m.stabl_scores_.shape[1] / DELTA), 0.0)
        sv      = m.score_variance_.max(axis=1)
        sk      = m.ko_score_variance_.max(axis=1) if m.ko_score_variance_ is not None else sv
        eps_tot = _bfw_eps(sv, B_ref, log_t) + _bfw_eps(sk, B_ref, log_t)

        thresh   = np.array(m.fdr_threshold_range)
        fdrs_arr = np.array(m.FDRs_)
        D_t      = np.array([int(np.sum(scores > t)) for t in thresh])
        bdry_t   = np.array([int(np.sum((scores > t) & (scores <= t + eps_tot))) for t in thresh])
        bdry_D   = bdry_t / np.maximum(D_t, 1)
        feas_bfw = bdry_t == 0

        t_u, _    = _t_unc(m)
        t_c, _, _ = _t_constr(m, scores, sv, sk, B_ref, log_t)

        _ROW_LABELS = ["FDP⁺(t)", "D(t)", "|∂⁺(t)| absolu", "|∂⁺(t)| / D(t)"]
        _ROW_COLORS = ["#4A90D9", "#C41E3A", "#2CA02C", "#1a7b1a"]
        _curves     = [fdrs_arr, D_t.astype(float), bdry_t.astype(float), bdry_D]

        fig3, axes3 = plt.subplots(4, 1, figsize=(7, 13))
        for row, (curve, color, ylabel) in enumerate(zip(_curves, _ROW_COLORS, _ROW_LABELS)):
            ax = axes3[row]
            ax.step(thresh, curve, where="post", color=color, lw=1.5)
            ylo = curve.min(); yhi = curve.max()
            pad = (yhi - ylo) * 0.12 if yhi > ylo else 0.05
            ax.set_ylim(ylo - pad, yhi + pad)
            _lbl = "faisable BernFW" if row == 0 else None
            for t_f in thresh[feas_bfw]:
                ax.axvspan(t_f, t_f + float(eps_tot.max()), alpha=0.15,
                           color="#2CA02C", label=_lbl, zorder=0)
                _lbl = None
            if row in (2, 3):
                ax.axhline(0, color="black", lw=0.8, ls="--", label="∂⁺=∅" if row == 2 else None)
            ax.axvline(t_u,  color="#C41E3A", lw=1.5, ls="--",
                       label=f"t* unc={t_u:.3f}" if row == 0 else None)
            ax.axvline(t_c,  color="#001A7B", lw=1.5, ls="-.",
                       label=f"t* constr={t_c:.3f}" if row == 0 else None)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if row == 0:
                ax.set_title(f"{est_name} — B={B_ref}", fontsize=9)
                ax.legend(fontsize=7)
            if row == 2:
                ax.legend(fontsize=7)
            if row == 3:
                ax.set_xlabel("t", fontsize=9)

        fig3.suptitle(f"Objective function — ds{ds['id']}\n"
                      f"rouge-- = t* unc | bleu-. = t* constr | vert = zone faisable", fontsize=9)
        fig3.tight_layout()
        fig3.savefig(out / "objective_function.pdf", dpi=150)
        plt.close(fig3)
        print(f"Objective function → {out}/objective_function.pdf")

        # ── FDP+(t) et vrai FDP(t) pour les 3 versions ────────────────────────────
        versions_fdp_t = {
            "V1":        (m_v1,     scores_v1,     "#C41E3A"),
            "V2_unc":    (m,        scores,        "#2CA02C"),
            "V2_constr": (m_constr, scores_constr, "#001A7B"),
        }
        fig_ft, axes_ft = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        for ax, (v_name, (mv, sc_v, color)) in zip(axes_ft, versions_fdp_t.items()):
            th_v   = np.array(mv.fdr_threshold_range)
            fp_v   = np.array(mv.FDRs_)
            tfdp_v = np.array([
                np.sum((sc_v > t) & ~true_mask) / max(1, int(np.sum(sc_v > t)))
                for t in th_v
            ])
            t_star_v = float(mv.fdr_min_threshold_)

            fdp_lbl = "FDP+(t)  (+1 num.)" if v_name == "V1" else "FDP+(t)  (sans +1)"
            ax.step(th_v, fp_v,   where="post", color=color, lw=1.8, label=fdp_lbl)
            ax.step(th_v, tfdp_v, where="post", color=color, lw=1.8, ls="--",
                    alpha=0.7, label="Vrai FDP(t)")

            # Augmented FDP+(t) — V2 uniquement (le +1 est remplacé par |∂⁺|/D via BernFW)
            if v_name in ("V2_unc", "V2_constr"):
                log_t_v = max(np.log(4 * mv.n_features_in_ * mv.stabl_scores_.shape[1] / DELTA), 0.0)
                sv_v    = mv.score_variance_.max(axis=1)
                sk_v    = mv.ko_score_variance_.max(axis=1) if mv.ko_score_variance_ is not None else sv_v
                eps_v   = _bfw_eps(sv_v, B_ref, log_t_v) + _bfw_eps(sk_v, B_ref, log_t_v)
                D_v     = np.array([max(1, int(np.sum(sc_v > t))) for t in th_v])
                bdry_v  = np.array([int(np.sum((sc_v > t) & (sc_v <= t + eps_v))) for t in th_v])
                corrected = np.minimum(fp_v + bdry_v / D_v, 1.0)
                ax.step(th_v, corrected, where="post", color=color, lw=1.4, ls="-.",
                        alpha=0.85, label="FDP+_aug(t) = FDP+(t) + |∂⁺|/D")
            ax.axvline(t_star_v, color="black", lw=1.2, ls=":", alpha=0.8,
                       label=f"t*={t_star_v:.3f}")
            ax.set(xlabel="t", ylabel="FDP", title=v_name, xlim=(0, 1), ylim=(-0.05, 1.1))
            ax.axhline(0.2, color="gray", ls="--", lw=0.6, alpha=0.4, label="FDR=0.2")
            ax.legend(fontsize=8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig_ft.suptitle(
            f"FDP+(t) (solide) vs Vrai FDP(t) (tiret) — B={B_ref}\n"
            f"ds{ds['id']} (n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']})",
            fontsize=10)
        fig_ft.tight_layout()
        fig_ft.savefig(out / "fdp_plus_vs_true_fdp.pdf", dpi=150)
        plt.close(fig_ft)
        print(f"FDP+ vs vrai FDP → {out}/fdp_plus_vs_true_fdp.pdf")

        # ── Précision / Recall vs t — V2_unc vs V2_constr ────────────────────────
        prec_t, rec_t = [], []
        for t in thresh:
            sel = set(f for f, s in zip(feat_names, scores) if s > t)
            tp  = len(sel & true_set)
            prec_t.append(tp / len(sel) if sel else 0.0)
            rec_t.append(tp / len(true_set))
        prec_t = np.array(prec_t)
        rec_t  = np.array(rec_t)

        f1_t   = np.where(prec_t + rec_t > 0, 2 * prec_t * rec_t / (prec_t + rec_t), 0.0)

        def _at_t(arr, t_query):
            idx = max(np.searchsorted(thresh, t_query, side="right") - 1, 0)
            return float(arr[idx])

        prec_u, rec_u, f1_u = _at_t(prec_t, t_u), _at_t(rec_t, t_u), _at_t(f1_t, t_u)
        prec_c, rec_c, f1_c = _at_t(prec_t, t_c), _at_t(rec_t, t_c), _at_t(f1_t, t_c)

        sel_u   = set(f for f, s in zip(feat_names, scores)    if s > t_u)
        sel_c   = set(f for f, s in zip(feat_names, scores)    if s > t_c)
        sel_v1  = set(f for f, s in zip(feat_names, scores_v1) if s > t_v1)
        tp_v1   = len(sel_v1 & true_set)
        prec_v1 = tp_v1 / len(sel_v1) if sel_v1 else 0.0
        rec_v1  = tp_v1 / len(true_set)
        f1_v1   = 2*prec_v1*rec_v1/(prec_v1+rec_v1) if prec_v1+rec_v1 > 0 else 0.0

        print(f"\n  V1        → t*={t_v1:.3f}  n_sel={len(sel_v1):3d}  "
              f"prec={prec_v1:.3f}  rec={rec_v1:.3f}  F1={f1_v1:.3f}")
        print(f"  V2_unc    → t*={t_u:.3f}  n_sel={len(sel_u):3d}  "
              f"prec={prec_u:.3f}  rec={rec_u:.3f}  F1={f1_u:.3f}")
        print(f"  V2_constr → t*={t_c:.3f}  n_sel={len(sel_c):3d}  "
              f"prec={prec_c:.3f}  rec={rec_c:.3f}  F1={f1_c:.3f}")
        print(f"  ∩(V1,V2_unc,V2_constr) : {sorted(sel_v1 & sel_u & sel_c)}")
        print(f"  Vraies features         : {sorted(true_set)}")

        fig_pr, axes_pr = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        for ax, curve, ylabel, yvals in [
            (axes_pr[0], prec_t, "Précision", (prec_u, prec_c)),
            (axes_pr[1], rec_t,  "Recall",    (rec_u,  rec_c)),
            (axes_pr[2], f1_t,   "F1",        (f1_u,   f1_c)),
        ]:
            ax.step(thresh, curve, where="post", color="#555555", lw=1.5)
            ax.axvline(t_u, color="#C41E3A", lw=1.5, ls="--",
                       label=f"t* unc    → {yvals[0]:.2f} ({len(sel_u)} sel)")
            ax.axvline(t_c, color="#001A7B", lw=1.5, ls="-.",
                       label=f"t* constr → {yvals[1]:.2f} ({len(sel_c)} sel)")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_ylim(-0.05, 1.1)
            ax.axhline(1.0, color="gray", ls="--", lw=0.5)
            ax.legend(fontsize=8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        axes_pr[2].set_xlabel("t", fontsize=9)
        fig_pr.suptitle(
            f"Précision, Recall & F1 vs t — V2_unc vs V2_constr (même modèle)\n"
            f"ds{ds['id']} (n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']}) — B={B_ref}",
            fontsize=9)
        fig_pr.tight_layout()
        fig_pr.savefig(out / "precision_recall_vs_t.pdf", dpi=150)
        plt.close(fig_pr)
        print(f"Précision/Recall/F1 vs t → {out}/precision_recall_vs_t.pdf")

        # ── Comparaison sélections V1 / V2_unc / V2_constr ────────────────────────
        all_sel     = sel_v1 | sel_u | sel_c | true_set
        feat_order  = sorted(all_sel, key=lambda f: int(f.split("_")[1]))
        v_labels    = ["V1", "V2_unc", "V2_constr"]
        sel_sets    = [sel_v1, sel_u, sel_c]
        t_stars_cmp = [t_v1, t_u, t_c]
        n_sels_cmp  = [len(sel_v1), len(sel_u), len(sel_c)]
        metrics_cmp = [(prec_v1, rec_v1, f1_v1),
                       (prec_u,  rec_u,  f1_u),
                       (prec_c,  rec_c,  f1_c)]
        pal_cmp_v   = {"V1": "#C41E3A", "V2_unc": "#2CA02C", "V2_constr": "#001A7B"}

        n_feat = len(feat_order)
        fig_cmp, axes_cmp = plt.subplots(
            1, 3, figsize=(17, max(5, n_feat * 0.35 + 2)),
            gridspec_kw={"width_ratios": [1.2, 1.2, 2]})

        # Panel 1 : threshold t* (barre) + n_selected (barre transparente, axe droit)
        ax = axes_cmp[0]
        x = np.arange(3); w = 0.35
        ax.bar(x - w/2, t_stars_cmp, w,
               color=[pal_cmp_v[v] for v in v_labels], alpha=0.85)
        ax_r = ax.twinx()
        ax_r.bar(x + w/2, n_sels_cmp, w,
                 color=[pal_cmp_v[v] for v in v_labels], alpha=0.35)
        ax_r.axhline(ds["k"], color="gray", ls="--", lw=0.8, label=f"k={ds['k']}")
        ax_r.legend(fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(v_labels, fontsize=8)
        ax.set_ylabel("t* (threshold)", fontsize=8)
        ax_r.set_ylabel("n sélectionnées", fontsize=8, color="gray")
        ax.set_title(f"Threshold t* (solide)\n& n_sel (transparent) — B={B_ref}", fontsize=8)
        ax.spines["top"].set_visible(False)

        # Panel 2 : Précision / Recall / F1
        ax = axes_cmp[1]
        metric_names = ["Précision", "Recall", "F1"]
        x = np.arange(3); w = 0.25
        for i, v_label in enumerate(v_labels):
            ax.bar(x + (i - 1) * w, metrics_cmp[i], w,
                   label=v_label, color=pal_cmp_v[v_label], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="gray", ls="--", lw=0.5)
        ax.set_ylabel("Score"); ax.legend(fontsize=7)
        ax.set_title("Précision / Recall / F1", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Panel 3 : heatmap features × versions
        ax = axes_cmp[2]
        mat = np.array([[1.0 if f in sel else 0.0 for sel in sel_sets]
                        for f in feat_order])
        ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(v_labels, fontsize=8)
        ax.set_yticks(range(n_feat)); ax.set_yticklabels(feat_order, fontsize=7)
        for j, f in enumerate(feat_order):
            lbl = ax.get_yticklabels()[j]
            if f in true_set:
                lbl.set_color("#C41E3A"); lbl.set_fontweight("bold")
        ax.set_title("Features sélectionnées\n(rouge gras = vraie feature)", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        fig_cmp.suptitle(
            f"Comparaison sélections — ds{ds['id']} "
            f"(n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']}) — B={B_ref}",
            fontsize=10)
        fig_cmp.tight_layout()
        fig_cmp.savefig(out / "selection_comparison.pdf", dpi=150)
        plt.close(fig_cmp)
        print(f"Selection comparison → {out}/selection_comparison.pdf")

        # ── Calibration FDP+ : depuis le cache (seed unique) ──────────────────────
        print("\n  [FDP+ calibration] V1 vs V2_unc vs V2_constr (cache)…")
        import pandas as pd
        fdp_palette    = {"v1": "#C41E3A", "v2_unc": "#2CA02C", "v2_constr": "#001A7B"}
        cache_fdp_vers = {"v1": (m_v1, scores_v1), "v2_unc": (m, scores),
                          "v2_constr": (m_constr, scores_constr)}
        fdp_records = []
        for v_name, (mv, sc_m) in cache_fdp_vers.items():
            t_sel    = float(mv.fdr_min_threshold_)
            fdp_plus = float(np.array(mv.FDRs_).min())
            sel      = set(f for f, s in zip(feat_names, sc_m) if s > t_sel)
            true_fdp = len(sel - true_set) / len(sel) if sel else 0.0
            tp       = len(sel & true_set)
            prec_m   = tp / len(sel) if sel else 0.0
            rec_m    = tp / len(true_set)
            f1_m     = 2*prec_m*rec_m/(prec_m+rec_m) if prec_m+rec_m > 0 else 0.0
            fdp_records.append({"version": v_name,
                                 "fdp_plus": round(fdp_plus, 4),
                                 "true_fdp": round(true_fdp, 4),
                                 "n_selected": len(sel), "f1": round(f1_m, 3)})
            print(f"    {v_name}: FDP+={fdp_plus:.3f}  FDP_vrai={true_fdp:.3f}  "
                  f"F1={f1_m:.2f}  n_sel={len(sel)}")

        df_fdp = pd.DataFrame(fdp_records)
        df_fdp.to_csv(out / "fdp_calibration.csv", index=False)

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
        ax.set(xlabel="FDP+ estimé", ylabel="Vrai FDP (ground truth)",
               title="Calibration FDP+\n(points sous diagonale = borne valide)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        ax = axes_fdp[1]
        ax.bar(x_fdp, [r["f1"] for r in fdp_records],
               color=[fdp_palette[v] for v in v_names_fdp], alpha=0.85)
        ax.set_xticks(x_fdp); ax.set_xticklabels(v_names_fdp, fontsize=8)
        ax.set_ylim(0, 1.15); ax.axhline(1.0, color="gray", ls="--", lw=0.5)
        ax.set_ylabel("F1"); ax.set_title(f"F1 score (B={B_ref})")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        ax = axes_fdp[2]
        ax.axhline(ds["k"], color="gray", ls="--", lw=0.8, label=f"k={ds['k']} vraies features")
        ax.bar(x_fdp, [r["n_selected"] for r in fdp_records],
               color=[fdp_palette[v] for v in v_names_fdp], alpha=0.85)
        ax.set_xticks(x_fdp); ax.set_xticklabels(v_names_fdp, fontsize=8)
        ax.set_ylabel("n features sélectionnées")
        ax.set_title("Features sélectionnées vs vérité terrain")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        fig_fdp.suptitle(
            f"Calibration FDP+ — ds{ds['id']} (n={ds['n']}, p={ds['p']}, "
            f"k={ds['k']}, signal={ds['signal']}) — B={B_ref}", fontsize=10)
        fig_fdp.tight_layout()
        fig_fdp.savefig(out / "fdp_calibration.pdf", dpi=150)
        plt.close(fig_fdp)
        print(f"  FDP+ calibration → {out}/fdp_calibration.pdf")

        # ── Variance des scores + Jaccard + Précision V1 vs V2 vs B ──────────────
        STAB_THRESH = 0.5
        palette_cmp = {"v1": "#C41E3A", "v2_unc": "#2CA02C", "v2_constr": "#001A7B"}
        versions_cmp = list(palette_cmp.keys())
        _pfx_map    = {"v1": "v1", "v2_unc": "v2unc", "v2_constr": "v2constr"}

        def _jaccard(s1, s2):
            u = len(s1 | s2)
            return len(s1 & s2) / u if u > 0 else 1.0

        var_rec  = {v: [] for v in versions_cmp}
        jac_rec  = {v: [] for v in versions_cmp}
        prec_rec = {v: [] for v in versions_cmp}
        rec_rec  = {v: [] for v in versions_cmp}
        f1_rec   = {v: [] for v in versions_cmp}

        for B in B_SWEEP:
            for v_name in versions_cmp:
                c_pfx = f"{_pfx_map[v_name]}_B{B}"
                # var + Jaccard depuis le cache multi-seeds
                if f"{c_pfx}_seed_sc" in _d.files:
                    seed_sc = _d[f"{c_pfx}_seed_sc"]
                    var_rec[v_name].append(float(seed_sc.var(axis=0).mean())
                                           if seed_sc.shape[0] >= 2 else np.nan)
                    sets = [frozenset(np.where(r > STAB_THRESH)[0]) for r in seed_sc]
                    pairs = [_jaccard(a, b) for i, a in enumerate(sets) for b in sets[i+1:]]
                    jac_rec[v_name].append(np.mean(pairs) if pairs else np.nan)
                else:
                    var_rec[v_name].append(np.nan)
                    jac_rec[v_name].append(np.nan)
                # prec/rec/f1 depuis le cache single-seed
                if f"{c_pfx}_scores" in _d.files:
                    m_c   = _make_proxy(_d, c_pfx)
                    sc_c  = np.max(m_c.stabl_scores_, axis=1)
                    t_c   = m_c.fdr_min_threshold_
                    sel_c = set(f for f, s in zip(feat_names, sc_c) if s > t_c)
                    tp_c  = len(sel_c & true_set)
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
                      title="Variance des scores V1 vs V2 vs B")
        axes_s[0].set_xscale("log"); axes_s[0].legend(fontsize=8)
        axes_s[0].spines["top"].set_visible(False); axes_s[0].spines["right"].set_visible(False)
        axes_s[1].set(xlabel="B", ylabel="Jaccard moyen inter-seeds",
                      title="Stabilité Jaccard vs B", ylim=(0, 1.05))
        axes_s[1].axhline(1.0, color="gray", ls="--", lw=0.5)
        axes_s[1].set_xscale("log"); axes_s[1].legend(fontsize=8)
        axes_s[1].spines["top"].set_visible(False); axes_s[1].spines["right"].set_visible(False)
        for ax_i, (metric_d, ylabel, title) in zip(axes_s[2:], [
            (prec_rec, "Précision",  "Précision vs B"),
            (rec_rec,  "Rappel",     "Rappel vs B"),
            (f1_rec,   "F1",         "F1 vs B"),
        ]):
            for v_name in versions_cmp:
                ax_i.plot(B_sw, metric_d[v_name], "o-", color=palette_cmp[v_name], label=v_name)
            ax_i.set(xlabel="B", ylabel=ylabel, title=title, ylim=(-0.05, 1.1))
            ax_i.axhline(1.0, color="gray", ls="--", lw=0.5)
            ax_i.set_xscale("log"); ax_i.legend(fontsize=8)
            ax_i.spines["top"].set_visible(False); ax_i.spines["right"].set_visible(False)
        fig_s.suptitle(
            f"Stabilité V1 vs V2_unc vs V2_constr — ds{ds['id']} "
            f"(n={ds['n']}, p={ds['p']}, k={ds['k']}, signal={ds['signal']})",
            fontsize=10)
        fig_s.tight_layout()
        out.mkdir(parents=True, exist_ok=True)
        fig_s.savefig(out / "stability_comparison.pdf", dpi=150)
        plt.close(fig_s)
        print(f"Stability comparison → {out}/stability_comparison.pdf")

        # ── ε_{B,W,j} et W_j en fonction de B (v2_constr uniquement) ─────────
        eps_med_sig, eps_q1_sig, eps_q3_sig = [], [], []
        eps_med_nul, eps_q1_nul, eps_q3_nul = [], [], []
        wj_med_sig,  wj_med_nul             = [], []
        B_eps_sw = []

        for B in B_SWEEP:
            c_pfx = f"v2constr_B{B}"
            if f"{c_pfx}_scores" not in _d.files:
                continue
            sc  = _d[f"{c_pfx}_scores"]     # (p, K)
            sv  = _d[f"{c_pfx}_score_var"]  # (p, K)
            kov = _d[f"{c_pfx}_ko_var"]     # (n_inj, K)
            sko = _d[f"{c_pfx}_ko_scores"]  # (n_inj, K)
            p_, K_ = sc.shape[0], sc.shape[1]
            if kov.shape[0] != p_:
                continue  # knockoffs non appariés — skip

            idx_p  = np.arange(p_)
            k_st   = np.argmax(sc, axis=1)                       # argmax_k score(j,k)
            sig2_W = sv[idx_p, k_st] + kov[idx_p, k_st]         # σ̂²_W,j
            w_j    = sc[idx_p, k_st] - sko[idx_p, k_st]         # W_j

            L_W   = max(np.log(2.0 * p_ * K_ / DELTA), 0.0)
            B_eff = max(B - 1, 1)
            disc  = (7/3)**2 * L_W**2 + 8 * B_eff * L_W * sig2_W
            eps_W = ((7/3) * L_W + np.sqrt(np.maximum(disc, 0.0))) / (2 * B_eff)

            B_eps_sw.append(B)
            for mask, me, q1, q3, mw in [
                (true_mask,  eps_med_sig, eps_q1_sig, eps_q3_sig, wj_med_sig),
                (~true_mask, eps_med_nul, eps_q1_nul, eps_q3_nul, wj_med_nul),
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
                              f"\nds{ds['id']} (n={ds['n']}, p={ds['p']},"
                              f" k={ds['k']}, signal={ds['signal']})"))
            ax_eps.legend(fontsize=8)
            ax_eps.spines["top"].set_visible(False)
            ax_eps.spines["right"].set_visible(False)
            fig_eps.tight_layout()
            fig_eps.savefig(out / "eps_W_vs_B.pdf", dpi=150)
            plt.close(fig_eps)
            print(f"  ε_W vs B → {out}/eps_W_vs_B.pdf")



# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int, nargs="?", default=None,
                        help="0=full run, 1=post-process only, 2=theory only")
    parser.add_argument("idx", type=int, nargs="?", default=None,
                        help="index du dataset (mode 2 uniquement)")
    args = parser.parse_args()

    if args.mode == 1:
        print("=== Post-processing ===")
        post_process()
    elif args.mode == 2:
        ds_str = f" dataset index {args.idx}" if args.idx is not None else " tous les datasets"
        print(f"=== Theory analysis —{ds_str} ===")
        run_theory_analysis(ds_idx=args.idx)
    else:
        # Full run (default — mode 0 or no arg)
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)
        print("=== Setup run dirs ===")
        setup_run_dirs()

        runs = _expand_runs(PARAMS)
        print(f"\n=== CV ({len(runs)} runs) ===")
        for idx in range(len(runs)):
            run_experiment(idx)

        print("\n=== Post-processing ===")
        post_process()

        print("\n=== Theory analysis ===")
        run_theory_analysis()

    print(f"\nDone. Résultats dans {TEST_DIR}/")
