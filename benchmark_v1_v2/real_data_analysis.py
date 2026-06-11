"""
real_data_analysis.py
Même analyse que single_dataset_analysis.py, mais sur les VRAIES données du labo
(benchmark_v1_v2/data/), donc SANS vérité terrain sur les features :
  - on retire tout ce qui suppose connaître la vraie distribution des nulles
    (vrai FDP, precision/recall/F1, séparation vraies/nulles dans les plots) ;
  - on garde : n_sel, AUC (Train->Validation si dispo, sinon CV 5-fold), Jaccard inter-run,
    variance des scores, fréquence de sélection des biomarqueurs.

Multi-seed : les données sont FIXES, seule la seed du bootstrap STABL varie d'un run à
l'autre -> mesure la stabilité ALGORITHMIQUE (V1 garde une variance irréductible, V2 non).

Knockoffs : Sigma estimé par Ledoit-Wolf (régime n<p), gaussianisation optionnelle.

Sorties : results_data/<dataset>/ {figures, analyse_complete.pdf, database.csv}
          + results_data/runs_metrics.csv (cumulatif, agrégé).

Usage :
  python real_data_analysis.py --dataset SSI_Proteomics --B 1000 --n-seeds 5
  python real_data_analysis.py --dataset all
"""
import os, sys, argparse, warnings, copy, csv
from datetime import datetime
warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

from stabl.stabl import Stabl as StablV1
from stabl.stablV2 import Stabl as StablV2
from stabl.adaptive import ALogitLasso
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_ROOT = os.path.join(HERE, "results_data")

# ── Config des cohortes : feature file, outcome (col + valeur positive), split val ─
DATASETS = {
    "SSI_Proteomics": dict(feat="Biobank SSI/Proteomics.csv",
                           out="Biobank SSI/outcome.csv", col="model1b", pos="1"),
    "SSI_CyTOF":      dict(feat="Biobank SSI/CyTOF.csv",
                           out="Biobank SSI/outcome.csv", col="model1b", pos="1"),
    "SSI_EarlyFusion": dict(feat=[("Prot", "Biobank SSI/Proteomics.csv"),
                                  ("CyTOF", "Biobank SSI/CyTOF.csv")],
                           out="Biobank SSI/outcome.csv", col="model1b", pos="1"),
    "SSI_LateFusion": dict(late=[("Prot", "Biobank SSI/Proteomics.csv"),
                                 ("CyTOF", "Biobank SSI/CyTOF.csv")],
                           out="Biobank SSI/outcome.csv", col="model1b", pos="1"),
    "CFRNA_Preeclampsia": dict(feat="CFRNA/cfrna_dataFINAL.csv",
                           out="CFRNA/all_outcomes.csv", col="Preeclampsia", pos="True"),
    "COVID_Proteomics": dict(feat="COVID-19/Training/Proteomics.csv",
                           out="COVID-19/Training/Mild&ModVsSevere.csv", col="Mild&ModVsSevere", pos="1",
                           feat_val="COVID-19/Validation/Validation_proteomics.csv",
                           out_val="COVID-19/Validation/Validation_outcome(WHO.0 >= 5).csv", col_val="WHO.0 ≥ 5", pos_val="True"),
    "Dream_Taxonomy": dict(feat="Dream/Taxonomy.csv",
                           out="Dream/Preterm.csv", col="was_preterm", pos="True"),
    "Dream_Phylotype": dict(feat="Dream/Phylotype.csv",
                           out="Dream/Preterm.csv", col="was_preterm", pos="True"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="SSI_Proteomics",
                help="clé de DATASETS, ou 'all'")
ap.add_argument("--B", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-seeds", "--n_seeds", type=int, default=5, dest="n_seeds")
ap.add_argument("--gaussianize", action="store_true",
                help="gaussianiser (rank-normal) avant les knockoffs (conseillé en omique)")
args = ap.parse_args()

B, NSEEDS, GAUSS = args.B, args.n_seeds, args.gaussianize
GRID = np.arange(0., 1., .01)


def _read_outcome(cfg):
    o = pd.read_csv(os.path.join(DATA_DIR, cfg["out"]), index_col=0)
    return (o[cfg["col"]].astype(str).str.strip() == str(cfg["pos"])).astype(int)

def _read_feat(path):
    X = pd.read_csv(os.path.join(DATA_DIR, path), index_col=0).apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.mean())

def load_dataset(cfg):
    """Renvoie Xtr, ytr, feat_names, Xval, yval. feat = str (1 omique) ou
    liste [(label, path), ...] -> EARLY FUSION (concat sur patients communs)."""
    y = _read_outcome(cfg)
    feat = cfg["feat"]
    if isinstance(feat, list):                                   # EARLY FUSION
        loaded = [(lab, _read_feat(p)) for lab, p in feat]
        common = y.index
        for _, X in loaded:
            common = common.intersection(X.index)
        mats, names = [], []
        for lab, X in loaded:
            Xc = X.loc[common]; mats.append(Xc.values)
            names += [f"{lab}:{c}" for c in Xc.columns]
        Xall = StandardScaler().fit_transform(np.hstack(mats))
        return Xall, y.loc[common].values, names, None, None
    # 1 omique (+ split validation éventuel)
    X = _read_feat(feat); idx = X.index.intersection(y.index)
    Xtr_df, ytr = X.loc[idx], y.loc[idx]
    Xval_df = yval = None
    if "feat_val" in cfg:
        yv = (pd.read_csv(os.path.join(DATA_DIR, cfg["out_val"]), index_col=0)[cfg["col_val"]]
              .astype(str).str.strip() == str(cfg["pos_val"])).astype(int)
        Xv = _read_feat(cfg["feat_val"]); iv = Xv.index.intersection(yv.index)
        Xv, yval = Xv.loc[iv], yv.loc[iv]
        common = Xtr_df.columns.intersection(Xv.columns)
        Xtr_df, Xval_df = Xtr_df[common], Xv[common]
    feat_names = list(Xtr_df.columns)
    sc = StandardScaler().fit(Xtr_df.values)
    Xtr = sc.transform(Xtr_df.values)
    Xval = sc.transform(Xval_df.values) if Xval_df is not None else None
    yval = yval.values if yval is not None else None
    return Xtr, ytr.values, feat_names, Xval, yval

def load_omics(cfg):
    """LATE FUSION : renvoie [(label, Xstd, feat_names), ...] sur patients communs + y."""
    y = _read_outcome(cfg)
    loaded = [(lab, _read_feat(p)) for lab, p in cfg["late"]]
    common = y.index
    for _, X in loaded:
        common = common.intersection(X.index)
    omics = [(lab, StandardScaler().fit_transform(X.loc[common].values), list(X.loc[common].columns))
             for lab, X in loaded]
    return omics, y.loc[common].values


def get_support(model, name=None):
    if name == "V2_vidé":
        sr = model.stabl_scores_.max(axis=1); eps = model.eps_B_total_fw_
        return np.where(sr > model.fdr_min_threshold_ + eps)[0]
    if hasattr(model, "get_support"):
        return np.where(model.get_support())[0]
    est = model.best_estimator_
    if hasattr(est, "coef_"):
        return np.where(np.abs(est.coef_[0]) > 1e-8)[0]
    return np.where(est.feature_importances_ > 0)[0]


def second_local_min_threshold(obj, grid, tol=1e-9):
    """a0 : un seul plateau de zéros terminant à 1 -> t du 2e min local ; sinon None."""
    obj = np.asarray(obj, float); n = len(obj); zero = obj <= tol
    regions = []; i = 0
    while i < n:
        if zero[i]:
            j = i
            while j + 1 < n and zero[j + 1]:
                j += 1
            regions.append((i, j)); i = j + 1
        else:
            i += 1
    if len(regions) != 1 or regions[0][1] != n - 1:
        return None
    zidx = set(range(regions[0][0], regions[0][1] + 1)); loc = []
    for i in range(1, n - 1):
        if i in zidx:
            continue
        if obj[i] <= obj[i - 1] and obj[i] <= obj[i + 1] and (obj[i] < obj[i - 1] or obj[i] < obj[i + 1]):
            loc.append((obj[i], i))
    if not loc:
        return None
    loc.sort(key=lambda x: (x[0], x[1]))
    return float(grid[loc[0][1]])


def auc_and_roc(sel, Xtr, ytr, Xval, yval):
    """AUC + (fpr,tpr) : Train->Val si val dispo, sinon CV 5-fold sur train."""
    if len(sel) == 0:
        return 0.5, (np.array([0., 1.]), np.array([0., 1.]))
    clf = LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=10000)
    if Xval is not None:
        clf.fit(Xtr[:, sel], ytr); prob = clf.predict_proba(Xval[:, sel])[:, 1]
        a = roc_auc_score(yval, prob); fpr, tpr, _ = roc_curve(yval, prob)
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        prob = cross_val_predict(clf, Xtr[:, sel], ytr, cv=cv, method="predict_proba")[:, 1]
        a = roc_auc_score(ytr, prob); fpr, tpr, _ = roc_curve(ytr, prob)
    return a, (np.asarray(fpr), np.asarray(tpr))


def run_one(seed, Xtr, ytr, Xval, yval, P, cov):
    def _lr():
        return LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                  max_iter=int(1e6), random_state=seed)

    def stabl_v2(mode, alpha=1.0):
        return StablV2(base_estimator=_lr(), lambda_grid="auto", n_lambda=10, n_bootstraps=B,
                       artificial_type="knockoff", artificial_proportion=1.0, sample_fraction=0.5,
                       replace=False, n_jobs=-1, random_state=seed, selection_mode=mode, delta=0.05,
                       alpha=alpha, knockoff_method="equicorrelated", cov_matrix=cov,
                       gaussianize_knockoffs=GAUSS)

    _CG = {"C": np.logspace(-2, 0, 10)}
    def baseline(est, grid):
        ns = max(2, min(5, int(np.min(np.bincount(ytr)))))
        return GridSearchCV(est, grid, cv=ns, scoring="roc_auc", n_jobs=-1)

    fitted = {}
    v1 = StablV1(base_estimator=_lr(), lambda_grid="auto", n_lambda=10, n_bootstraps=B,
                 artificial_type="knockoff", artificial_proportion=1.0, sample_fraction=0.5,
                 replace=False, n_jobs=-1, random_state=seed, cov_matrix=cov)
    v1.fit(Xtr, ytr); fitted["V1"] = v1
    vc = stabl_v2("constrained"); vc.fit(Xtr, ytr); fitted["V2_constr"] = vc
    ma0 = copy.deepcopy(vc); ma0.alpha = 0.0; ma0._compute_FDPplus(); fitted["V2_constr_a0"] = ma0
    fitted["V2_vidé"] = copy.deepcopy(vc)
    # V2_alpha
    sigma = 2.0 * float(np.diff(np.asarray(vc.fdr_threshold_range)).mean())
    g0 = np.asarray(ma0.fdr_threshold_range); sr0 = ma0.stabl_scores_.max(axis=1); e0 = ma0.eps_B_total_fw_
    D0 = np.array([max(1, int((sr0 > t).sum())) for t in g0])
    fro0 = np.array([((sr0 > t) & (sr0 <= t + e0)).sum() for t in g0]) / D0
    obj0 = np.asarray(ma0.FDRs_) + fro0
    alt = second_local_min_threshold(obj0, g0)
    if alt is not None:
        ma0.fdr_min_threshold_ = alt
    zm = obj0 <= 1e-12; zi = np.where(zm)[0]
    if len(zi):
        last = int(zi[-1]); st = last
        while st - 1 >= 0 and zm[st - 1]:
            st -= 1
        t_drop = float(g0[st])
    else:
        t_drop = float(ma0.fdr_min_threshold_)
    use_bar = not bool((ma0.stabl_scores_.max(axis=1) == 1.0).any())
    def alpha_comb(t, D):
        return (float(np.exp((t - t_drop) / sigma)) if use_bar else 0.0) * D
    mal = copy.deepcopy(vc); mal.alpha = alpha_comb; mal._compute_FDPplus(); fitted["V2_alpha"] = mal
    bl = {"ALasso": baseline(ALogitLasso(solver="liblinear", class_weight="balanced",
                                         tol=1e-4, max_iter=int(1e6)), _CG),
          "Lasso": baseline(LogisticRegression(penalty="l1", solver="liblinear",
                            class_weight="balanced", max_iter=int(1e6)), _CG),
          "ElasticNet": baseline(LogisticRegression(penalty="elasticnet", solver="saga",
                            l1_ratio=0.5, class_weight="balanced", max_iter=int(1e6)), _CG)}
    if HAS_XGB:
        bl["XGBoost"] = baseline(XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1,
                                 verbosity=0), {"n_estimators": [100, 300], "max_depth": [3, 5],
                                                "learning_rate": [0.05, 0.2]})
    for nm, m in bl.items():
        m.fit(Xtr, ytr); fitted[nm] = m

    summary, roc, feat, curves, sel_sets = [], {}, {}, {}, {}
    for name, model in fitted.items():
        sel = get_support(model, name); ns = len(sel)
        a, (fpr, tpr) = auc_and_roc(sel, Xtr, ytr, Xval, yval)
        summary.append(dict(name=name, ns=ns, auc=a))
        roc[name] = (fpr, tpr, a); sel_sets[name] = set(int(j) for j in sel)
        sc = getattr(model, "stabl_scores_", None)
        if sc is not None:
            sr = sc.max(axis=1); sko = model.stabl_scores_artificial_.max(axis=1)
            sm = np.zeros(P, bool); sm[sel] = True
            eps = getattr(model, "eps_B_total_fw_", None)
            feat[name] = {"sr": sr, "sr_ko": sko, "selected": sm, "eps": eps}
            if eps is not None:
                Dg = np.array([max(1, int((sr > t).sum())) for t in GRID])
                frog = np.array([((sr > t) & (sr <= t + eps)).sum() for t in GRID]) / Dg
            else:
                frog = np.zeros(len(GRID))
            fdpp = np.asarray(model.FDRs_) if getattr(model, "FDRs_", None) is not None else np.zeros(len(GRID))
            fdpp = np.interp(GRID, np.asarray(model.fdr_threshold_range), fdpp)
            curves[name] = {"fdp_plus": fdpp, "fro": frog, "tstar": float(model.fdr_min_threshold_)}
    return dict(summary=summary, roc=roc, feat=feat, curves=curves, sel_sets=sel_sets)


def analyze(key):
    cfg = DATASETS[key]
    print(f"\n=== {key} ===")
    Xtr, ytr, FEAT, Xval, yval = load_dataset(cfg)
    P = Xtr.shape[1]
    print(f"  X={Xtr.shape}  classes={np.bincount(ytr)}  "
          f"{'val=' + str(Xval.shape) if Xval is not None else 'pas de val -> CV'}")
    cov = LedoitWolf().fit(Xtr).covariance_                          # Sigma estimé (n<p)
    SEEDS = [args.seed + i for i in range(NSEEDS)]
    RES = []
    for i, s in enumerate(SEEDS, 1):
        print(f"  run {i}/{NSEEDS} (seed={s}) ...", flush=True)
        RES.append(run_one(s, Xtr, ytr, Xval, yval, P, cov))

    OUT = os.path.join(OUT_ROOT, key); os.makedirs(OUT, exist_ok=True)
    MODELS = [d["name"] for d in RES[0]["summary"]]
    STABL  = [m for m in MODELS if m in RES[0]["feat"]]
    NR = len(RES)
    TITLE = f"{key} — p={P}, n={Xtr.shape[0]}, B={B} | moy. {NR} seeds"

    def marr(name, key2):
        return np.array([next(d[key2] for d in r["summary"] if d["name"] == name) for r in RES])
    NSEL = {m: marr(m, "ns") for m in MODELS}
    AUC  = {m: marr(m, "auc") for m in MODELS}

    def jac(name):
        S = [r["sel_sets"][name] for r in RES]; v = []
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                u = len(S[a] | S[b]); v.append(1.0 if u == 0 else len(S[a] & S[b]) / u)
        return float(np.mean(v)) if v else 1.0
    JAC = {m: jac(m) for m in MODELS}

    SR = {m: np.vstack([r["feat"][m]["sr"] for r in RES]) for m in STABL}
    KO = {m: np.vstack([r["feat"][m]["sr_ko"] for r in RES]) for m in STABL}
    SELF = {m: np.vstack([r["feat"][m]["selected"] for r in RES]).mean(0) for m in STABL}
    SRm = {m: SR[m].mean(0) for m in STABL}; SRs = {m: SR[m].std(0) for m in STABL}
    KOm = {m: KO[m].mean(0) for m in STABL}
    EPSm = {m: np.vstack([r["feat"][m]["eps"] for r in RES]).mean(0)
            for m in STABL if RES[0]["feat"][m]["eps"] is not None}
    FDPP = {m: np.vstack([r["curves"][m]["fdp_plus"] for r in RES]) for m in STABL}
    FRO  = {m: np.vstack([r["curves"][m]["fro"] for r in RES]) for m in STABL}
    stcol = lambda n: "#C41E3A" if n.startswith("V") else "#4D4F53"

    # ── Fig : table (n_sel, AUC, Jaccard) ─────────────────────────────────────
    fig_t, ax = plt.subplots(figsize=(9, 0.8 + 0.4 * len(MODELS))); ax.axis("off")
    col = ["Modèle", "# sélec.", "AUC", "Jaccard"]
    ms = lambda a: f"{a.mean():.2f}±{a.std():.2f}"
    cell = [[m, ms(NSEL[m]), ms(AUC[m]), f"{JAC[m]:.2f}"] for m in MODELS]
    tb = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.6)
    for j in range(len(col)):
        c = tb[0, j]; c.set_facecolor("#001A7B"); c.set_text_props(color="white", weight="bold")
    for i, m in enumerate(MODELS, 1):
        if m.startswith("V"):
            for j in range(len(col)):
                tb[i, j].set_facecolor("#F2E6E9")
    ax.set_title(f"Sélection & AUC (moy.±std, {NR} seeds) — {TITLE}", fontsize=11, pad=14)
    fig_t.tight_layout(); fig_t.savefig(os.path.join(OUT, "selection_table.png"), dpi=130)

    # ── Fig : AUC + Jaccard bars ──────────────────────────────────────────────
    fig_b, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xp = np.arange(len(MODELS))
    axes[0].bar(xp, [AUC[m].mean() for m in MODELS], yerr=[AUC[m].std() for m in MODELS],
                capsize=3, color=[stcol(m) for m in MODELS], alpha=0.85)
    axes[0].axhline(0.5, ls=":", color="gray"); axes[0].set_ylim(0, 1.02)
    axes[0].set_xticks(xp); axes[0].set_xticklabels(MODELS, rotation=25, ha="right")
    axes[0].set_ylabel("AUC"); axes[0].set_title("AUC (Train→Val ou CV) moy.±std"); axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(xp, [JAC[m] for m in MODELS], color=[stcol(m) for m in MODELS], alpha=0.85)
    axes[1].set_ylim(0, 1.02); axes[1].set_xticks(xp); axes[1].set_xticklabels(MODELS, rotation=25, ha="right")
    axes[1].set_ylabel("Jaccard moyen inter-run"); axes[1].set_title("Stabilité des sélections"); axes[1].grid(axis="y", alpha=0.3)
    fig_b.suptitle(TITLE); fig_b.tight_layout()
    fig_b.savefig(os.path.join(OUT, "auc_jaccard.png"), dpi=130)

    # ── Fig : ROC moyenne ─────────────────────────────────────────────────────
    fig_r, ax = plt.subplots(figsize=(7, 6)); fg = np.linspace(0, 1, 101)
    for m in MODELS:
        tprs = []
        for r in RES:
            fpr, tpr, _ = r["roc"][m]; t = np.interp(fg, fpr, tpr); t[0] = 0.; tprs.append(t)
        ax.plot(fg, np.mean(tprs, 0), lw=1.8, label=f"{m} (AUC={AUC[m].mean():.3f}±{AUC[m].std():.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="gray"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC moyenne — {TITLE}"); ax.legend(loc="lower right", fontsize=7); ax.grid(alpha=0.3)
    fig_r.tight_layout(); fig_r.savefig(os.path.join(OUT, "roc_curves.png"), dpi=130)

    # ── Fig : distributions de scores (réel vs knockoff, SANS vraies/nulles) ──
    def score_fig(m, fname):
        srm, kom, srs = SRm[m], KOm[m], SRs[m]; sr_all, ko_all = SR[m], KO[m]
        sf = SELF[m]
        fig, axe = plt.subplots(1, 2, figsize=(13, 5))
        bins = np.linspace(0, max(sr_all.max(), ko_all.max()) + 0.02, 30)
        axe[0].hist(sr_all.ravel(), bins=bins, alpha=0.5, color="#4D4F53", label="features réelles")
        axe[0].hist(ko_all.ravel(), bins=bins, alpha=0.5, color="#1f77b4", label="knockoffs")
        axe[0].set_xlabel("Score de stabilité max_λ"); axe[0].set_ylabel("Nombre (poolé)")
        axe[0].set_title(f"Distribution des scores ({m})"); axe[0].legend(fontsize=8)
        sc = axe[1].scatter(kom, srm, c=sf, cmap="viridis", s=18, alpha=0.8)
        lim = max(srm.max(), kom.max()) + 0.05
        axe[1].plot([0, lim], [0, lim], ls=":", color="gray", label="score = score_ko")
        axe[1].set_xlabel("score knockoff moyen"); axe[1].set_ylabel("score réel moyen")
        axe[1].set_title("Réel vs knockoff par feature (couleur = freq. sélection)"); axe[1].legend(fontsize=8)
        plt.colorbar(sc, ax=axe[1], label="fréq. sélection")
        fig.suptitle(f"Scores {m} — {TITLE}"); fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=130); return fig
    fig_sc = score_fig("V2_constr", "score_distributions.png")
    fig_sc1 = score_fig("V1", "score_distributions_v1.png") if "V1" in STABL else None

    # ── Fig : objectif FDP+ + frontière (SANS vrai FDP) par run + MOYENNE ─────
    def obj_grid_fig(m, fname, title):
        obj = FDPP[m] + FRO[m]; tst = [r["curves"][m]["tstar"] for r in RES]
        pan = [(FDPP[m][i], obj[i], tst[i], f"run {i+1}", False) for i in range(NR)]
        pan.append((FDPP[m].mean(0), obj.mean(0), float(np.mean(tst)), "MOYENNE", True))
        nc = 3; nr = (len(pan) + nc - 1) // nc
        fig, axe = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.2 * nr), squeeze=False); axe = axe.ravel()
        for a in axe[len(pan):]:
            a.axis("off")
        for a, (fp, oc, ts, ttl, im) in zip(axe, pan):
            if im: a.set_facecolor("#F4F4F4")
            a.plot(GRID, fp, color="#999999", lw=1.0, label="FDP+ seul")
            a.fill_between(GRID, fp, oc, step="post", alpha=0.3, color="#2CA02C", label="frontière")
            a.plot(GRID, oc, color="#1f77b4", lw=2.0, label="objectif")
            a.axvline(ts, color="black", ls=":", lw=1.3, label=f"t*={ts:.2f}")
            a.set_ylim(0, 1.05); a.set_title(ttl, fontsize=9, weight="bold" if im else "normal")
            a.legend(fontsize=6, loc="upper right"); a.grid(alpha=0.25); a.tick_params(labelsize=7)
        fig.suptitle(title, fontsize=10); fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(OUT, fname), dpi=140); return fig
    fig_o = obj_grid_fig("V2_constr_a0", "v2constr_a0_objective.png",
                         f"V2_constr_a0 — objectif FDP+ (sans +1) + frontière par run — {TITLE}")

    # ── Fig : barrière ∂⁺(t*) (SANS étoiles vraies) par run + MOYENNE ─────────
    fig_bd = None
    if "V2_constr_a0" in EPSm:
        Mb = "V2_constr_a0"; SRr, KOr = SR[Mb], KO[Mb]
        eps = [r["feat"][Mb]["eps"] for r in RES]; tst = [r["curves"][Mb]["tstar"] for r in RES]
        lim = max(SRr.max(), KOr.max(), SRm[Mb].max(), KOm[Mb].max()) + 0.04
        pan = [(SRr[i], KOr[i], eps[i], tst[i], f"run {i+1}", False) for i in range(NR)]
        pan.append((SRm[Mb], KOm[Mb], EPSm[Mb], float(np.mean(tst)), "MOYENNE", True))
        nc = 3; nr = (len(pan) + nc - 1) // nc
        fig_bd, axe = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.0 * nr), squeeze=False); axe = axe.ravel()
        for a in axe[len(pan):]:
            a.axis("off")
        for a, (sr, sko, ep, ts, ttl, im) in zip(axe, pan):
            if im: a.set_facecolor("#F4F4F4")
            selm = sr > ts; bnd = selm & (sr <= ts + ep); sf = selm & ~bnd
            for j in np.where(selm)[0]:
                c = "#ff7f0e" if bnd[j] else "#2CA02C"
                a.plot([sko[j], sko[j]], [sr[j] - ep[j], sr[j]], color=c, alpha=0.5, lw=0.9, zorder=2)
            a.scatter(sko[~selm], sr[~selm], s=7, color="#888888", alpha=0.6, zorder=3)
            a.scatter(sko[sf], sr[sf], s=16, color="#2CA02C", alpha=0.85, zorder=4)
            a.scatter(sko[bnd], sr[bnd], s=22, color="#ff7f0e", alpha=0.85, zorder=4)
            a.axhline(ts, color="#1f77b4", ls="--", lw=1.3); a.plot([0, lim], [0, lim], ls=":", color="gray", lw=0.7)
            a.set_xlim(0, lim); a.set_ylim(0, lim); a.tick_params(labelsize=6)
            a.set_title(f"{ttl}  t*={ts:.2f}, |∂⁺|={int(bnd.sum())}", fontsize=8, weight="bold" if im else "normal")
        proxy = [Line2D([], [], marker='o', ls='', color="#888888", label="non sél."),
                 Line2D([], [], marker='o', ls='', color="#2CA02C", label="sél. hors ∂⁺"),
                 Line2D([], [], marker='o', ls='', color="#ff7f0e", label="∂⁺(t*)"),
                 Line2D([], [], color="#1f77b4", ls='--', label="t*")]
        fig_bd.legend(handles=proxy, loc="upper center", ncol=4, fontsize=8)
        fig_bd.supxlabel("score knockoff", fontsize=9); fig_bd.supylabel("score réel", fontsize=9)
        fig_bd.suptitle(f"Barrière ∂⁺(t*) de V2_constr_a0 par run — {TITLE}", fontsize=9, y=0.998)
        fig_bd.tight_layout(rect=[0, 0, 1, 0.95]); fig_bd.savefig(os.path.join(OUT, "barriere_dplus_a0.png"), dpi=140)

    # ── Fig : variance des scores par feature (SANS vraies/nulles) ────────────
    fig_v, ax = plt.subplots(figsize=(11, 4.8))
    for m, c in [("V1", "#1f77b4"), ("V2_constr", "#C41E3A")]:
        if m in STABL:
            ax.scatter(np.arange(P), SRs[m], s=8, alpha=0.4, color=c, label=m)
    ax.set_xlabel("feature"); ax.set_ylabel("std du score inter-run")
    ax.set_title(f"Variance (std) des scores par feature ({NR} seeds) — {TITLE}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig_v.tight_layout()
    fig_v.savefig(os.path.join(OUT, "score_variance.png"), dpi=130)

    # ── Fig : top biomarqueurs (fréquence de sélection, V2_constr) ────────────
    fig_bm, ax = plt.subplots(figsize=(10, 6)); sf = SELF["V2_constr"]
    top = np.argsort(sf)[::-1][:25]; top = top[sf[top] > 0]
    if len(top):
        ax.barh(range(len(top)), sf[top][::-1], color="#C41E3A", alpha=0.85)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([FEAT[j] for j in top[::-1]], fontsize=7)
        ax.set_xlabel(f"fréquence de sélection (sur {NR} runs)"); ax.set_xlim(0, 1.02)
    ax.set_title(f"Top biomarqueurs V2_constr (freq. sélection ≥1 run) — {TITLE}", fontsize=10)
    fig_bm.tight_layout(); fig_bm.savefig(os.path.join(OUT, "top_biomarkers.png"), dpi=130)

    # ── PDF ───────────────────────────────────────────────────────────────────
    pages = [p for p in [fig_t, fig_b, fig_r, fig_sc, fig_sc1, fig_o, fig_bd, fig_v, fig_bm] if p is not None]
    with PdfPages(os.path.join(OUT, "analyse_complete.pdf")) as pdf:
        for f in pages:
            pdf.savefig(f)
    for f in pages:
        plt.close(f)

    # ── database.csv local (1 ligne / run × modèle × feature) ─────────────────
    DBC = ["seed", "run", "dataset", "p", "n", "B", "model", "feature", "feature_name",
           "score", "score_ko", "selected", "n_sel", "auc", "auc_mean", "jaccard_mean"]
    with open(os.path.join(OUT, "database.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DBC); w.writeheader()
        for ri, res in enumerate(RES):
            for name in MODELS:
                rm = next(d for d in res["summary"] if d["name"] == name)
                fe = res["feat"].get(name); ss = res["sel_sets"][name]
                base = dict(seed=SEEDS[ri], run=ri + 1, dataset=key, p=P, n=Xtr.shape[0], B=B,
                            model=name, n_sel=rm["ns"], auc=round(rm["auc"], 4),
                            auc_mean=round(AUC[name].mean(), 4), jaccard_mean=round(JAC[name], 4))
                for j in range(P):
                    row = dict(base, feature=j, feature_name=FEAT[j], selected=int(j in ss))
                    if fe is not None:
                        row["score"] = round(float(fe["sr"][j]), 6); row["score_ko"] = round(float(fe["sr_ko"][j]), 6)
                    else:
                        row["score"] = ""; row["score_ko"] = ""
                    w.writerow(row)
    print(f"  -> {OUT}  (figures + PDF + database.csv)")

    # ── CSV global cumulatif (agrégé) ─────────────────────────────────────────
    GCSV = os.path.join(OUT_ROOT, "runs_metrics.csv")
    gcols = ["timestamp", "dataset", "p", "n", "B", "n_seeds", "model",
             "n_sel_mean", "n_sel_std", "auc_mean", "auc_std", "jaccard_mean"]
    hdr = not os.path.exists(GCSV)
    with open(GCSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gcols)
        if hdr: w.writeheader()
        ts = datetime.now().isoformat(timespec="seconds")
        for m in MODELS:
            w.writerow(dict(timestamp=ts, dataset=key, p=P, n=Xtr.shape[0], B=B, n_seeds=NR,
                            model=m, n_sel_mean=round(NSEL[m].mean(), 3), n_sel_std=round(NSEL[m].std(), 3),
                            auc_mean=round(AUC[m].mean(), 4), auc_std=round(AUC[m].std(), 4),
                            jaccard_mean=round(JAC[m], 4)))


def analyze_late(key):
    """LATE FUSION : STABL par omique (FDP+ contrôlé par modalité) -> UNION des biomarqueurs
    -> modèle prédictif combiné. AUC/Jaccard sur l'union ; scores par omique."""
    cfg = DATASETS[key]; print(f"\n=== {key} (late fusion) ===")
    omics, y = load_omics(cfg)
    offsets, feat_names, parts, covs = [], [], [], []
    off = 0
    for lab, Xs, names in omics:
        offsets.append(off); off += Xs.shape[1]
        feat_names += [f"{lab}:{n}" for n in names]; parts.append(Xs)
        covs.append(LedoitWolf().fit(Xs).covariance_)
    Xcat = np.hstack(parts); Ptot = Xcat.shape[1]
    print(f"  omiques: {[(l, X.shape[1]) for l, X, _ in omics]}  p_total={Ptot}  classes={np.bincount(y)}")
    SEEDS = [args.seed + i for i in range(NSEEDS)]
    RES, PEROMIC = [], []
    for i, s in enumerate(SEEDS, 1):
        print(f"  run {i}/{NSEEDS} (seed={s}) ...", flush=True)
        po = [run_one(s, Xs, y, None, None, Xs.shape[1], covs[oi])
              for oi, (lab, Xs, names) in enumerate(omics)]
        PEROMIC.append(po)
        mres = {}
        for m in [d["name"] for d in po[0]["summary"]]:
            gsel = sorted({offsets[oi] + j for oi, ro in enumerate(po) for j in ro["sel_sets"][m]})
            a, rc = auc_and_roc(np.array(gsel, int), Xcat, y, None, None)
            mres[m] = dict(sel=set(gsel), ns=len(gsel), auc=a, roc=rc)
        RES.append(mres)
    MODELS = [d["name"] for d in PEROMIC[0][0]["summary"]]
    STABL  = [m for m in MODELS if m in PEROMIC[0][0]["feat"]]
    NR = len(RES)
    OUT = os.path.join(OUT_ROOT, key); os.makedirs(OUT, exist_ok=True)
    TITLE = f"{key} (late fusion) — p={Ptot}, n={len(y)}, B={B} | {NR} seeds"
    NSEL = {m: np.array([RES[r][m]["ns"] for r in range(NR)]) for m in MODELS}
    AUC  = {m: np.array([RES[r][m]["auc"] for r in range(NR)]) for m in MODELS}
    def jac(m):
        S = [RES[r][m]["sel"] for r in range(NR)]; v = []
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                u = len(S[a] | S[b]); v.append(1.0 if u == 0 else len(S[a] & S[b]) / u)
        return float(np.mean(v)) if v else 1.0
    JAC = {m: jac(m) for m in MODELS}
    selfreq = np.zeros(Ptot)
    for r in range(NR):
        for j in RES[r]["V2_constr"]["sel"]:
            selfreq[j] += 1
    selfreq /= NR
    stcol = lambda n: "#C41E3A" if n.startswith("V") else "#4D4F53"

    # table
    fig_t, ax = plt.subplots(figsize=(9, 0.8 + 0.4 * len(MODELS))); ax.axis("off")
    ms = lambda a: f"{a.mean():.2f}±{a.std():.2f}"
    tb = ax.table(cellText=[[m, ms(NSEL[m]), ms(AUC[m]), f"{JAC[m]:.2f}"] for m in MODELS],
                  colLabels=["Modèle", "# sélec. (union)", "AUC", "Jaccard"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.6)
    for j in range(4):
        c = tb[0, j]; c.set_facecolor("#001A7B"); c.set_text_props(color="white", weight="bold")
    for i, m in enumerate(MODELS, 1):
        if m.startswith("V"):
            for j in range(4):
                tb[i, j].set_facecolor("#F2E6E9")
    ax.set_title(f"Late fusion — sélection (union) & AUC — {TITLE}", fontsize=11, pad=14)
    fig_t.tight_layout(); fig_t.savefig(os.path.join(OUT, "selection_table.png"), dpi=130)

    # auc + jaccard
    fig_b, axe = plt.subplots(1, 2, figsize=(13, 4.5)); xp = np.arange(len(MODELS))
    axe[0].bar(xp, [AUC[m].mean() for m in MODELS], yerr=[AUC[m].std() for m in MODELS], capsize=3,
               color=[stcol(m) for m in MODELS], alpha=0.85); axe[0].axhline(0.5, ls=":", color="gray")
    axe[0].set_ylim(0, 1.02); axe[0].set_xticks(xp); axe[0].set_xticklabels(MODELS, rotation=25, ha="right")
    axe[0].set_ylabel("AUC"); axe[0].set_title("AUC (union des biomarqueurs, CV)"); axe[0].grid(axis="y", alpha=0.3)
    axe[1].bar(xp, [JAC[m] for m in MODELS], color=[stcol(m) for m in MODELS], alpha=0.85)
    axe[1].set_ylim(0, 1.02); axe[1].set_xticks(xp); axe[1].set_xticklabels(MODELS, rotation=25, ha="right")
    axe[1].set_ylabel("Jaccard inter-run (union)"); axe[1].set_title("Stabilité"); axe[1].grid(axis="y", alpha=0.3)
    fig_b.suptitle(TITLE); fig_b.tight_layout(); fig_b.savefig(os.path.join(OUT, "auc_jaccard.png"), dpi=130)

    # roc
    fig_r, ax = plt.subplots(figsize=(7, 6)); fg = np.linspace(0, 1, 101)
    for m in MODELS:
        tprs = [np.interp(fg, RES[r][m]["roc"][0], RES[r][m]["roc"][1]) for r in range(NR)]
        for t in tprs:
            t[0] = 0.
        ax.plot(fg, np.mean(tprs, 0), lw=1.8, label=f"{m} (AUC={AUC[m].mean():.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="gray"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC moyenne (union) — {TITLE}"); ax.legend(loc="lower right", fontsize=7); ax.grid(alpha=0.3)
    fig_r.tight_layout(); fig_r.savefig(os.path.join(OUT, "roc_curves.png"), dpi=130)

    # top biomarqueurs (union, taggés par omique)
    fig_bm, ax = plt.subplots(figsize=(10, 6))
    top = np.argsort(selfreq)[::-1][:25]; top = top[selfreq[top] > 0]
    if len(top):
        cols = ["#C41E3A" if feat_names[j].startswith("Prot") else "#1f77b4" for j in top[::-1]]
        ax.barh(range(len(top)), selfreq[top][::-1], color=cols, alpha=0.85)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([feat_names[j] for j in top[::-1]], fontsize=7)
        ax.set_xlabel(f"fréq. sélection ({NR} runs)"); ax.set_xlim(0, 1.02)
    ax.legend(handles=[Line2D([], [], color="#C41E3A", lw=6, label="Prot"),
                       Line2D([], [], color="#1f77b4", lw=6, label="CyTOF")], fontsize=8)
    ax.set_title(f"Top biomarqueurs V2_constr (union late fusion) — {TITLE}", fontsize=10)
    fig_bm.tight_layout(); fig_bm.savefig(os.path.join(OUT, "top_biomarkers.png"), dpi=130)

    # distributions de scores PAR OMIQUE (V2_constr)
    omic_figs = []
    for oi, (lab, Xs, names) in enumerate(omics):
        SRo = np.vstack([PEROMIC[r][oi]["feat"]["V2_constr"]["sr"] for r in range(NR)])
        KOo = np.vstack([PEROMIC[r][oi]["feat"]["V2_constr"]["sr_ko"] for r in range(NR)])
        fig, axx = plt.subplots(figsize=(7, 5))
        bins = np.linspace(0, max(SRo.max(), KOo.max()) + 0.02, 30)
        axx.hist(SRo.ravel(), bins=bins, alpha=0.5, color="#4D4F53", label="réelles")
        axx.hist(KOo.ravel(), bins=bins, alpha=0.5, color="#1f77b4", label="knockoffs")
        axx.set_xlabel("score de stabilité"); axx.set_ylabel("nombre")
        axx.set_title(f"Scores {lab} (V2_constr) — {key}"); axx.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"score_distributions_{lab}.png"), dpi=130)
        omic_figs.append(fig)

    pages = [fig_t, fig_b, fig_r, fig_bm] + omic_figs
    with PdfPages(os.path.join(OUT, "analyse_complete.pdf")) as pdf:
        for f in pages:
            pdf.savefig(f)
    for f in pages:
        plt.close(f)

    # database.csv (1 ligne / run × modèle × feature global)
    DBC = ["seed", "run", "dataset", "p", "n", "B", "model", "feature", "feature_name",
           "omic", "score", "selected", "n_sel", "auc", "auc_mean", "jaccard_mean"]
    with open(os.path.join(OUT, "database.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DBC); w.writeheader()
        for ri in range(NR):
            for m in MODELS:
                sel = RES[ri][m]["sel"]
                base = dict(seed=SEEDS[ri], run=ri + 1, dataset=key, p=Ptot, n=len(y), B=B, model=m,
                            n_sel=RES[ri][m]["ns"], auc=round(RES[ri][m]["auc"], 4),
                            auc_mean=round(AUC[m].mean(), 4), jaccard_mean=round(JAC[m], 4))
                for j in range(Ptot):
                    oi = max(k for k in range(len(offsets)) if offsets[k] <= j)
                    loc = j - offsets[oi]; lab = omics[oi][0]
                    fe = PEROMIC[ri][oi]["feat"].get(m)
                    sc = round(float(fe["sr"][loc]), 6) if fe is not None else ""
                    w.writerow(dict(base, feature=j, feature_name=feat_names[j], omic=lab,
                                    score=sc, selected=int(j in sel)))
    print(f"  -> {OUT}  (figures + PDF + database.csv)")

    GCSV = os.path.join(OUT_ROOT, "runs_metrics.csv")
    gcols = ["timestamp", "dataset", "p", "n", "B", "n_seeds", "model",
             "n_sel_mean", "n_sel_std", "auc_mean", "auc_std", "jaccard_mean"]
    hdr = not os.path.exists(GCSV)
    with open(GCSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gcols)
        if hdr: w.writeheader()
        ts = datetime.now().isoformat(timespec="seconds")
        for m in MODELS:
            w.writerow(dict(timestamp=ts, dataset=key, p=Ptot, n=len(y), B=B, n_seeds=NR, model=m,
                            n_sel_mean=round(NSEL[m].mean(), 3), n_sel_std=round(NSEL[m].std(), 3),
                            auc_mean=round(AUC[m].mean(), 4), auc_std=round(AUC[m].std(), 4),
                            jaccard_mean=round(JAC[m], 4)))


os.makedirs(OUT_ROOT, exist_ok=True)
keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
for k in keys:
    if "late" in DATASETS[k]:
        analyze_late(k)
    else:
        analyze(k)
print("\nTerminé.")
