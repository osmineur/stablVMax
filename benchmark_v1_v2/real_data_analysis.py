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
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_val_predict, ParameterGrid
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from sklearn.base import clone

TASK = "binary"          # "binary" (AUC) ou "regression" (R²) — fixé par dataset dans _call_loader
from joblib import Parallel, delayed

from stabl.stabl import Stabl as StablV1, plot_stabl_path, plot_fdr_graph
from stabl.stablV2 import Stabl as StablV2, fit_bootstrapped_sample
from stabl.adaptive import ALogitLasso, ALasso
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Patch knockpy.calc_mineig : pour p>=1500 il passe par ARPACK (eigsh) qui ne converge
#    pas sur des covariances singulières (n<p), et son fallback référence un chemin scipy
#    SUPPRIMÉ -> crash. On le remplace par un calcul dense robuste (valeur propre min exacte).
import knockpy.utilities as _ku
import stabl.stablV2 as _sv2
def _robust_calc_mineig(M):
    return float(np.linalg.eigvalsh(np.asarray(M, dtype=float))[0])
_ku.calc_mineig = _robust_calc_mineig
_sv2.calc_mineig = _robust_calc_mineig

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_ROOT = os.path.join(HERE, "results_data")

# ── Cohortes : LOADERS OFFICIELS STABL (stabl/data.py) -> préproc IDENTIQUE au papier
#    (log2 cfRNA, négation du label val COVID, sinh Onset CyTOF val, remove_low_info_samples...).
from stabl.data import load_ssi, load_covid_19, load_cfrna, load_dream, load_onset_of_labor_cv
DATASETS = {
    "SSI_Proteomics":     dict(loader=load_ssi,  path="Biobank SSI", omics=["Proteomics"],            fusion="single"),
    "SSI_CyTOF":          dict(loader=load_ssi,  path="Biobank SSI", omics=["CyTOF"],                 fusion="single"),
    "SSI_EarlyFusion":    dict(loader=load_ssi,  path="Biobank SSI", omics=["CyTOF", "Proteomics"],   fusion="early"),
    "SSI_LateFusion":     dict(loader=load_ssi,  path="Biobank SSI", omics=["CyTOF", "Proteomics"],   fusion="late"),
    "COVID_Proteomics":   dict(loader=load_covid_19, path="COVID-19", omics=["Proteomics"],           fusion="single"),
    "CFRNA_Preeclampsia": dict(loader=load_cfrna, path="CFRNA",       omics=["CFRNA"],                fusion="single"),
    "Dream_Taxonomy":     dict(loader=load_dream, path="Dream",       omics=["Taxonomy"],             fusion="single"),
    "Dream_Phylotype":    dict(loader=load_dream, path="Dream",       omics=["Phylotype"],            fusion="single"),
    # Onset : outcome DOS continu -> le loader renvoie task='regression' -> binarisé médiane ici.
    "Onset_Proteomics":   dict(loader=load_onset_of_labor_cv, path="Onset of Labor", omics=["Proteomics"],   fusion="single"),
    "Onset_CyTOF":        dict(loader=load_onset_of_labor_cv, path="Onset of Labor", omics=["CyTOF"],        fusion="single"),
    "Onset_Metabolomics": dict(loader=load_onset_of_labor_cv, path="Onset of Labor", omics=["Metabolomics"], fusion="single"),
    "Onset_EarlyFusion":  dict(loader=load_onset_of_labor_cv, path="Onset of Labor",
                               omics=["CyTOF", "Proteomics", "Metabolomics"], fusion="early"),
    "Onset_LateFusion":   dict(loader=load_onset_of_labor_cv, path="Onset of Labor",
                               omics=["CyTOF", "Proteomics", "Metabolomics"], fusion="late"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="SSI_Proteomics",
                help="clé de DATASETS, ou 'all'")
ap.add_argument("--B", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-seeds", "--n_seeds", type=int, default=5, dest="n_seeds")
ap.add_argument("--gaussianize", action="store_true",
                help="gaussianiser (rank-normal) avant les knockoffs (conseillé en omique)")
ap.add_argument("--base", choices=["all", "lasso", "alasso"], default="all",
                help="estimateur(s) de base de STABL ; 'all' = lasso+alasso (1 modèle chacun)")
ap.add_argument("--delta", type=float, default=0.001,
                help="niveau de risque des bornes de concentration (garantie 1-δ) ; défaut 0.001 (99.9%%)")
ap.add_argument("--only-last", "--only_last", action="store_true", dest="only_last",
                help="ne fitter que Last (pas de knockoffs) — utile quand V1/V2 plantent (p grand, cov singulière)")
ap.add_argument("--prefilter", type=int, default=0,
                help="pré-filtre : garde les top-K features par variance PAR OMIQUE (0 = off) ; rend les knockoffs valides en n<<p")
ap.add_argument("--ko-max-p", "--ko_max_p", type=int, default=2000, dest="ko_max_p",
                help="V2 : knockoffs si p<=seuil, sinon FALLBACK random_permutation (knockoffs trop lourds) ; défaut 2000")
args = ap.parse_args()

B, NSEEDS, GAUSS, BASE, DELTA = args.B, args.n_seeds, args.gaussianize, args.base, args.delta
ONLY_LAST = args.only_last; PREFILTER = args.prefilter; KO_MAX_P = args.ko_max_p
BASES = ["lasso", "alasso"] if BASE == "all" else [BASE]
REF = BASES[0]                          # base de référence pour les figures détaillées
GRID = np.arange(0., 1., .01)


def _topk_var(Xdf):
    """Pré-filtre non-supervisé : top-PREFILTER colonnes par variance (faisabilité knockoff en p≫n)."""
    if PREFILTER and 0 < PREFILTER < Xdf.shape[1]:
        return Xdf[Xdf.var(axis=0).nlargest(PREFILTER).index]
    return Xdf

def _call_loader(cfg):
    """Loader officiel STABL. Fixe TASK ; outcome continu CONSERVÉ pour la régression (Onset)."""
    global TASK
    train_dict, valid_dict, y_train, y_valid, _pid, task = cfg["loader"](os.path.join(DATA_DIR, cfg["path"]))
    TASK = "regression" if task == "regression" else "binary"
    cast = float if TASK == "regression" else int
    y_train = y_train.astype(cast)
    y_valid = y_valid.astype(cast) if y_valid is not None else None
    return train_dict, valid_dict, y_train, y_valid

def _prep(Xdf, idx, cols=None):
    """Lignes idx (+ colonnes cols) -> numérique -> fillna(moyenne)."""
    X = Xdf.loc[idx]
    if cols is not None:
        X = X[cols]
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.mean())

def _common_index(dct, omics, y):
    idx = y.index
    for om in omics:
        idx = idx.intersection(dct[om].index)
    return pd.Index(idx).unique()

def load_dataset(cfg):
    """1 omique ou EARLY FUSION (loaders officiels STABL). Renvoie Xtr, ytr, names, Xval, yval."""
    train_dict, valid_dict, y_train, y_valid = _call_loader(cfg)
    omics = cfg["omics"]; multi = len(omics) > 1
    common = _common_index(train_dict, omics, y_train)
    kept = {om: list(_topk_var(_prep(train_dict[om], common)).columns) for om in omics}
    if valid_dict is not None:                                   # COVID : colonnes train ∩ val
        cval = _common_index(valid_dict, omics, y_valid)
        for om in omics:
            kept[om] = [c for c in kept[om] if c in valid_dict[om].columns]
    def _stack(dct, idx):
        mats, names = [], []
        for om in omics:
            mats.append(_prep(dct[om], idx, kept[om]).values)
            names += ([f"{om}:{c}" for c in kept[om]] if multi else list(kept[om]))
        return np.hstack(mats), names
    Xtr_vals, names = _stack(train_dict, common); ytr = y_train.loc[common].values
    Xval = yval = None
    if valid_dict is not None:
        Xval, _ = _stack(valid_dict, cval); yval = y_valid.loc[cval].values
    return Xtr_vals, ytr, names, Xval, yval               # BRUT -> standardisé dans le fold

def load_omics(cfg):
    """LATE FUSION (loaders officiels) : [(omic, Xstd, names), ...] + y."""
    train_dict, _vd, y_train, _yv = _call_loader(cfg)
    omics = cfg["omics"]
    common = _common_index(train_dict, omics, y_train)
    out = []
    for om in omics:
        Xc = _topk_var(_prep(train_dict[om], common))
        out.append((om, Xc.values, list(Xc.columns)))    # BRUT -> standardisé dans le fold
    return out, y_train.loc[common].values


def get_support(model, name=None):
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
    clf = LogisticRegression(penalty=None, solver="lbfgs", class_weight="balanced",
                             max_iter=int(1e6), random_state=42)   # = STABL officiel (sans pénalité)
    if Xval is not None:
        clf.fit(Xtr[:, sel], ytr); prob = clf.predict_proba(Xval[:, sel])[:, 1]
        a = roc_auc_score(yval, prob); fpr, tpr, _ = roc_curve(yval, prob)
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        prob = cross_val_predict(clf, Xtr[:, sel], ytr, cv=cv, method="predict_proba")[:, 1]
        a = roc_auc_score(ytr, prob); fpr, tpr, _ = roc_curve(ytr, prob)
    return a, (np.asarray(fpr), np.asarray(tpr))


def auc_baseline(est, Xtr, ytr, Xval, yval, seed):
    """AUC du baseline ENTRAÎNÉ (son propre classifieur, toutes features) :
    out-of-fold 5-fold CV (ou train->val si val dispo). Pas de fuite, pas de logistique commune."""
    if Xval is not None:
        est.fit(Xtr, ytr); prob = est.predict_proba(Xval)[:, 1]
        a = roc_auc_score(yval, prob); fpr, tpr, _ = roc_curve(yval, prob)
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        prob = cross_val_predict(est, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        a = roc_auc_score(ytr, prob); fpr, tpr, _ = roc_curve(ytr, prob)
    return a, (np.asarray(fpr), np.asarray(tpr))


def mp_eps(score_variance, P, B):
    """Tolérance feature-wise Maurer-Pontil (empirical Bernstein) sur les scores."""
    s2 = score_variance.max(axis=1); Kl = score_variance.shape[1]
    L = max(np.log(4 * P * Kl / DELTA), 0.0); Beff = max(B - 1, 1)
    return ((7/3)*L + np.sqrt(np.maximum((7/3)**2*L**2 + 8*Beff*L*s2, 0.0))) / (2*Beff)


def perm_null_scores(X, y, base, lambda_grid_dict, B, frac, seed):
    """Référence nulle de Last : permutation de Y FRAÎCHE par bootstrap (B permutations,
    une par sous-échantillon — fidèle au Thm 4bis). Réutilise la sélection exacte de STABL
    (fit_bootstrapped_sample + SelectFromModel). Renvoie score_perm(j)=max_λ freq_perm,
    et la variance par (feature, λ) pour Maurer-Pontil."""
    n, P = X.shape; ss = max(2, int(round(frac * n)))
    ldicts = list(ParameterGrid(lambda_grid_dict)); K = len(ldicts); y = np.asarray(y)

    def _boot(b):
        rng = np.random.default_rng([seed, b])
        idx = rng.choice(n, ss, replace=False)
        for _ in range(50):                                       # garantir 2 classes (comme STABL classic_bootstrap)
            if len(np.unique(y[idx])) >= 2:
                break
            idx = rng.choice(n, ss, replace=False)
        Xb, yb = X[idx], y[idx]; yp = yb[rng.permutation(ss)]      # permutation fraîche par bootstrap
        out = np.empty((K, P))
        for ki, ld in enumerate(ldicts):
            out[ki] = fit_bootstrapped_sample(clone(base), Xb, yp, lambda_val=ld, threshold=None)
        return out
    sel = np.asarray(Parallel(n_jobs=-1)(delayed(_boot)(b) for b in range(B)))   # (B, K, P)
    freq = sel.mean(axis=0)                                   # (K, P)
    return freq.max(axis=0), sel.var(axis=0).T                # score_perm (P,), variance (P, K)

def build_baselines(y):
    """Baselines 'propre modèle' (classif: AUC ; régression: R²). Utilisé partout (fit_all + late)."""
    _XG = {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.2]}
    if TASK == "regression":
        cv = KFold(5, shuffle=True, random_state=0); AG = {"alpha": np.logspace(-3, 1, 10)}
        def gs(est, grid): return GridSearchCV(est, grid, cv=cv, scoring="r2", n_jobs=-1)
        bl = {"ALasso": gs(ALasso(max_iter=int(1e5)), AG),
              "Lasso": gs(Lasso(max_iter=int(1e5)), AG),
              "ElasticNet": gs(ElasticNet(l1_ratio=0.5, max_iter=int(1e5)), AG)}
        if HAS_XGB:
            bl["XGBoost"] = gs(XGBRegressor(random_state=42, n_jobs=1, verbosity=0), _XG)
        return bl
    _CG = {"C": np.logspace(-2, 0, 10)}
    ns = max(2, min(5, int(np.min(np.bincount(y)))))
    def gs(est, grid): return GridSearchCV(est, grid, cv=ns, scoring="roc_auc", n_jobs=-1)
    bl = {"ALasso": gs(ALogitLasso(solver="liblinear", class_weight="balanced", tol=1e-4, max_iter=int(1e6)), _CG),
          "Lasso": gs(LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=int(1e6)), _CG),
          "ElasticNet": gs(LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                           class_weight="balanced", max_iter=int(1e4)), _CG)}
    if HAS_XGB:
        bl["XGBoost"] = gs(XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1, verbosity=0), _XG)
    return bl


def make_base(b, seed):
    if TASK == "regression":                              # Onset : Lasso / ALasso régression
        if b == "alasso":
            return ALasso(max_iter=int(1e5), random_state=seed)
        return Lasso(max_iter=int(1e5), random_state=seed)
    if b == "alasso":
        return ALogitLasso(solver="liblinear", class_weight="balanced", tol=1e-4,
                           max_iter=int(1e6), random_state=seed)
    if b == "elasticnet":
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                  class_weight="balanced", max_iter=int(1e4), random_state=seed)
    return LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                              max_iter=int(1e6), random_state=seed)

def fit_all(Xtr, ytr, P, cov, seed):
    """Fit TOUS les modèles (STABL + baselines) sur (Xtr, ytr).
    Réutilisé en full-data (figures) ET dans chaque fold de la nested CV (AUC honnête)."""
    def stabl_v2(b, mode, alpha=1.0):
        # V2 : knockoffs gaussiens si p<=KO_MAX_P ; sinon FALLBACK random_permutation (cov trop lourde)
        if P <= KO_MAX_P:
            art = dict(artificial_type="knockoff", knockoff_method="equicorrelated",
                       cov_matrix=cov, gaussianize_knockoffs=GAUSS)
        else:
            art = dict(artificial_type="random_permutation")
        return StablV2(base_estimator=make_base(b, seed), lambda_grid="auto", n_lambda=10,
                       n_bootstraps=B, artificial_proportion=1.0, sample_fraction=0.5, replace=False,
                       n_jobs=-1, random_state=seed, selection_mode=mode, delta=DELTA, alpha=alpha, **art)

    fitted = {}
    for b in BASES:                                   # une version × chaque base de --base
        if not ONLY_LAST:                             # --only-last : ne fitter que Last
            # V1 = STABL OFFICIEL : leurres par permutation de colonnes (random_permutation) -> pas de Σ
            v1 = StablV1(base_estimator=make_base(b, seed), lambda_grid="auto", n_lambda=10,
                         n_bootstraps=B, artificial_type="random_permutation", artificial_proportion=1.0,
                         sample_fraction=0.5, replace=False, n_jobs=-1, random_state=seed)
            v1.fit(Xtr, ytr); fitted[f"V1_{b}"] = v1
            # V2 = knockoffs gaussiens (Σ) : échoue proprement en p>>n (cov trop grande) -> V2 absent
            try:
                vc = stabl_v2(b, "constrained"); vc.fit(Xtr, ytr); fitted[f"V2_constr_{b}"] = vc
            except (MemoryError, np.linalg.LinAlgError, ValueError) as e:
                print(f"  [V2_constr_{b}] knockoffs infaisables (p={P}): {type(e).__name__} -> sauté")
                vc = None
            if vc is not None:
                # V2_constr_a0 : MÊMES scores, objectif SANS le +1 (alpha=0) -> seul le seuil change
                srv = vc.stabl_scores_.max(axis=1); skv = vc.stabl_scores_artificial_.max(axis=1)
                gv = np.asarray(vc.fdr_threshold_range); Dv = np.array([max(1, int((srv > t).sum())) for t in gv])
                frov = np.array([((srv > t) & (srv <= t + vc.eps_B_total_fw_)).sum() for t in gv]) / Dv
                fdpp_vc_a0 = np.array([int((skv > t).sum()) for t in gv]) / Dv          # |S_ko(t)|/D(t), sans +1
                vc.OBJ_ = np.asarray(vc.FDRs_) + frov                                   # objectif COMPLET = FDP+ + frontière
                tstar_vc_a0 = float(gv[int(np.argmin(fdpp_vc_a0 + frov))])
                vca0 = copy.deepcopy(vc)
                if int((srv > tstar_vc_a0).sum()) > 0:                                  # a0 sélectionne -> objectif sans +1
                    vca0.fdr_min_threshold_ = vca0.hard_threshold = tstar_vc_a0
                    vca0.FDRs_ = fdpp_vc_a0; vca0.OBJ_ = fdpp_vc_a0 + frov
                # sinon : FALLBACK automatique sur V2_constr (avec +1) -> vca0 reste = copie de vc
                fitted[f"V2_constr_a0_{b}"] = vca0
        # Last : score(j) réel + référence nulle = PERMUTATION DE Y fraîche par bootstrap
        real = StablV2(base_estimator=make_base(b, seed), lambda_grid="auto", n_lambda=10,
                       n_bootstraps=B, artificial_type=None, sample_fraction=0.5, replace=False,
                       n_jobs=-1, random_state=seed, selection_mode="unconstrained", delta=DELTA,
                       hard_threshold=0.5)
        real.fit(Xtr, ytr)
        srn = real.stabl_scores_.max(axis=1); gn = np.asarray(real.fdr_threshold_range)
        Dn = np.array([max(1, int((srn > t).sum())) for t in gn])
        epsn = mp_eps(real.score_variance_, P, B)             # ε_B,j
        last = copy.deepcopy(real)
        sperm, var_perm = perm_null_scores(Xtr, ytr, make_base(b, seed),
                                           real.fitted_lambda_grid_, B, 0.5, seed + 9973)
        epsp = mp_eps(var_perm, P, B)
        fdpp_perm = (np.array([int((sperm > t).sum()) for t in gn]) + 1) / Dn  # (|S_perm(t)|+1)/D(t)
        eps_tot = epsn + epsp                                  # ε_B,j + ε_B,j,perm
        fro_perm = np.array([((srn > t) & (srn <= t + eps_tot)).sum() for t in gn]) / Dn
        tstl = float(gn[int(np.argmin(fdpp_perm + fro_perm))])
        last.hard_threshold = last.fdr_min_threshold_ = tstl
        last.eps_B_total_fw_ = eps_tot; last.FDRs_ = fdpp_perm
        last.OBJ_ = fdpp_perm + fro_perm                                       # objectif COMPLET = (|S_perm|+1)/D + frontière
        last.score_perm_ = sperm
        fitted[f"Last_{b}"] = last
        # Last_a0 : MÊMES scores, objectif SANS le +1 -> seul le seuil change
        fdpp_perm_a0 = np.array([int((sperm > t).sum()) for t in gn]) / Dn      # |S_perm(t)|/D(t), sans +1
        tstl_a0 = float(gn[int(np.argmin(fdpp_perm_a0 + fro_perm))])
        last_a0 = copy.deepcopy(last)
        if int((srn > tstl_a0).sum()) > 0:                                     # a0 sélectionne -> objectif sans +1
            last_a0.fdr_min_threshold_ = last_a0.hard_threshold = tstl_a0
            last_a0.FDRs_ = fdpp_perm_a0; last_a0.OBJ_ = fdpp_perm_a0 + fro_perm
        # sinon : FALLBACK automatique sur Last (avec +1) -> last_a0 reste = copie de last
        fitted[f"Last_a0_{b}"] = last_a0
    # baselines (indépendants de la base STABL), une seule fois — classif (AUC) ou régression (R²)
    bl = build_baselines(ytr)
    for nm, m in bl.items():
        m.fit(Xtr, ytr); fitted[nm] = m
    return fitted


def _predict_oof(name, model, Xtr_f, ytr_f, Xte_f):
    """Prédiction du modèle 'name' sur Xte_f, entraîné sur (Xtr_f, ytr_f).
    Baseline -> son propre classifieur ; STABL -> logistique L2 sur ses features sélectionnées."""
    reg = (TASK == "regression")
    if getattr(model, "stabl_scores_", None) is None:        # baseline (déjà fit dans fit_all)
        return model.predict(Xte_f) if reg else model.predict_proba(Xte_f)[:, 1]
    sel = get_support(model, name)
    if len(sel) == 0:
        return np.full(Xte_f.shape[0], float(np.mean(ytr_f)) if reg else 0.5)
    l2 = name.startswith(("V2_constr", "Last"))              # L2 pour V2/Last/a0 (anti sur-apprentissage) ; V1 -> penalty=None (officiel)
    if reg:
        clf = Ridge(alpha=1.0) if l2 else LinearRegression()
    elif l2:
        clf = LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=int(1e6), random_state=42)
    else:
        clf = LogisticRegression(penalty=None, solver="lbfgs", class_weight="balanced", max_iter=int(1e6), random_state=42)
    clf.fit(Xtr_f[:, sel], ytr_f)
    return clf.predict(Xte_f[:, sel]) if reg else clf.predict_proba(Xte_f[:, sel])[:, 1]


def _splitter(y, seed, K=5):
    """KFold (régression) ou StratifiedKFold (classif, bornée par la classe minoritaire)."""
    if TASK == "regression":
        return KFold(K, shuffle=True, random_state=seed)
    return StratifiedKFold(min(K, int(np.min(np.bincount(y)))), shuffle=True, random_state=seed)

def _score(y, pred):
    """R² (régression) ou AUC + courbe ROC (classif)."""
    if TASK == "regression":
        return (r2_score(y, pred), (np.array([0., 1.]), np.array([0., 1.])))
    fpr, tpr, _ = roc_curve(y, pred)
    return (roc_auc_score(y, pred), (np.asarray(fpr), np.asarray(tpr)))


def nested_cv_auc(X, y, P, seed, K=5):
    """NESTED CV honnête : standardisation + sélection + classifieur refit DANS chaque fold,
    prédiction out-of-fold. Renvoie {name: (score, (fpr,tpr))}."""
    oof, names, fsel = {}, None, {}
    for tr, te in _splitter(y, seed, K).split(X, y):
        sc = StandardScaler().fit(X[tr])                     # standardisation DANS le fold
        Xtr_f, Xte_f, ytr_f = sc.transform(X[tr]), sc.transform(X[te]), y[tr]
        cov_f = LedoitWolf().fit(Xtr_f).covariance_ if (not ONLY_LAST and P <= KO_MAX_P) else None
        fitted = fit_all(Xtr_f, ytr_f, P, cov_f, seed)
        if names is None:
            names = list(fitted); oof = {m: np.full(len(y), np.nan) for m in names}; fsel = {m: [] for m in names}
        for name, model in fitted.items():
            oof[name][te] = _predict_oof(name, model, Xtr_f, ytr_f, Xte_f)
            fsel[name].append(set(int(j) for j in get_support(model, name)))   # sélection (STABL OU baseline) sur ces patients
    return {m: (*_score(y, oof[m]), oof[m], fsel[m]) for m in names}   # (score, roc, préds OOF, sélections/fold)


def nested_cv_auc_late(omics, offsets, Xcat, y, seed, K=5):
    """NESTED CV honnête pour la LATE FUSION : dans chaque fold, on fit STABL PAR OMIQUE sur
    le train, on UNIONne les sélections, logistique L2 sur Xcat ; baselines = leur modèle sur Xcat."""
    reg = (TASK == "regression")
    oof, names = {}, None
    for tr, te in _splitter(y, seed, K).split(Xcat, y):
        ytr_f = y[tr]
        per_omic, cat_tr, cat_te = [], [], []
        for oi, (lab, Xs, _) in enumerate(omics):
            sc = StandardScaler().fit(Xs[tr])                     # std DANS le fold, par omique
            Xs_tr, Xs_te = sc.transform(Xs[tr]), sc.transform(Xs[te])
            cat_tr.append(Xs_tr); cat_te.append(Xs_te)
            cov_f = LedoitWolf().fit(Xs_tr).covariance_ if (not ONLY_LAST and Xs.shape[1] <= KO_MAX_P) else None
            per_omic.append(fit_all(Xs_tr, ytr_f, Xs.shape[1], cov_f, seed))
        Xcat_tr, Xcat_te = np.hstack(cat_tr), np.hstack(cat_te)
        if names is None:
            names = list(per_omic[0]); oof = {m: np.full(len(y), np.nan) for m in names}; fsel = {m: [] for m in names}
        blm = build_baselines(ytr_f)
        for name in names:
            if name in blm:                                       # baseline : son modèle sur Xcat
                blm[name].fit(Xcat_tr, ytr_f)
                fsel[name].append(set(int(j) for j in get_support(blm[name], name)))   # sélection baseline par fold
                oof[name][te] = blm[name].predict(Xcat_te) if reg else blm[name].predict_proba(Xcat_te)[:, 1]
            else:                                                 # STABL : union par-omique + classifieur
                gsel = sorted({offsets[oi] + j for oi, f in enumerate(per_omic)
                               for j in get_support(f[name], name)})
                fsel[name].append(set(gsel))                      # sélection (union) sur ce sous-ensemble de patients
                if not gsel:
                    oof[name][te] = float(np.mean(ytr_f)) if reg else 0.5
                else:
                    l2 = name.startswith(("V2_constr", "Last"))     # L2 pour V2/Last/a0
                    if reg:
                        clf = Ridge(alpha=1.0) if l2 else LinearRegression()
                    elif l2:
                        clf = LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=int(1e6), random_state=42)
                    else:
                        clf = LogisticRegression(penalty=None, solver="lbfgs", class_weight="balanced", max_iter=int(1e6), random_state=42)
                    clf.fit(Xcat_tr[:, gsel], ytr_f)
                    oof[name][te] = clf.predict(Xcat_te[:, gsel]) if reg else clf.predict_proba(Xcat_te[:, gsel])[:, 1]
    return {m: (*_score(y, oof[m]), oof[m], fsel[m]) for m in names}


def holdout_auc(Xtr_raw, ytr, Xval_raw, yval, P, seed):
    """train->val honnête : standardisation sur le TRAIN, fit (la sélection ne voit pas le val),
    prédit le val. Renvoie {name: (score, (fpr,tpr))}."""
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xval = sc.transform(Xtr_raw), sc.transform(Xval_raw)
    cov = LedoitWolf().fit(Xtr).covariance_ if (not ONLY_LAST and P <= KO_MAX_P) else None
    fitted = fit_all(Xtr, ytr, P, cov, seed)
    out = {}
    for name, model in fitted.items():
        pred = _predict_oof(name, model, Xtr, ytr, Xval)
        out[name] = (*_score(yval, pred), pred, [])     # pas de folds -> Jaccard inter-seed (fallback)
    return out


def run_one(seed, Xtr, ytr, Xval, yval, P, cov):
    fitted = fit_all(Xtr, ytr, P, cov, seed)
    summary, roc, feat, curves, sel_sets = [], {}, {}, {}, {}
    for name, model in fitted.items():
        sel = get_support(model, name); ns = len(sel)
        # AUC/ROC calculées séparément en NESTED CV (analyze) -> pas de calcul redondant ici
        summary.append(dict(name=name, ns=ns, auc=0.5))
        sel_sets[name] = set(int(j) for j in sel)
        sc = getattr(model, "stabl_scores_", None)
        if sc is not None:
            sr = sc.max(axis=1)
            sko_a = getattr(model, "stabl_scores_artificial_", None)
            if getattr(model, "score_perm_", None) is not None:        # Last : réf nulle = score_perm
                sko = model.score_perm_
            elif sko_a is not None:
                sko = sko_a.max(axis=1)
            else:
                sko = np.zeros(P)
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
    stabl_models = {k: v for k, v in fitted.items() if getattr(v, "stabl_scores_", None) is not None}
    return dict(summary=summary, roc=roc, feat=feat, curves=curves, sel_sets=sel_sets, models=stabl_models)


# ── Plots additionnels (parité avec STABL officiel) ───────────────────────────
def plot_native_stabl(models_dict, ref, prefix=""):
    """Stability path + FDR graph officiels (stabl.py), RENVOYÉS comme figures (-> combinés au PDF).
    Le FDR graph utilise l'objectif COMPLET minimisé (OBJ_ = FDP+ + frontière) au lieu du seul FDP+."""
    figs = []
    for name, mdl in models_dict.items():
        if not name.endswith(ref):
            continue
        try:
            res = plot_stabl_path(mdl, show_fig=False, export_file=False)
            fp = res[0] if isinstance(res, tuple) else res
            fp.suptitle(f"Stability path — {prefix}{name}", fontsize=9); figs.append(fp)
        except Exception as e:
            print(f"  [path {prefix}{name}] non dispo: {type(e).__name__}")
        try:
            m2 = copy.deepcopy(mdl)
            obj = np.asarray(getattr(mdl, "OBJ_", mdl.FDRs_))    # objectif complet si dispo (V2/Last), sinon FDRs_ (V1)
            m2.FDRs_ = obj; m2.min_fdr_ = float(obj.min())
            ff, fax = plot_fdr_graph(m2, show_fig=False, export_file=False)
            fax.set_title(f"Objectif minimisé (FDP+ + frontière) — {prefix}{name}", fontsize=9); figs.append(ff)
        except Exception as e:
            print(f"  [fdr {prefix}{name}] non dispo: {type(e).__name__}")
    return figs

def fig_prc(PRED, y, models, out, title, stcol):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in models:
        pr, rc, _ = precision_recall_curve(y, PRED[m]); ap = average_precision_score(y, PRED[m])
        ax.plot(rc, pr, lw=1.6, color=stcol(m), label=f"{m} (AP={ap:.3f})")
    ax.axhline(y.mean(), ls=":", color="gray"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"Precision-Recall — {title}")
    ax.legend(loc="upper right", fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "prc_curves.png"), dpi=130); return fig

def fig_reg_scatter(PRED, y, models, out, title):
    from scipy.stats import pearsonr
    n = len(models); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for i, m in enumerate(models):
        a = ax[i // cols][i % cols]; p = PRED[m]
        a.scatter(y, p, s=14, alpha=.6, color="#001A7B")
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        a.plot([lo, hi], [lo, hi], ls="--", color="gray")
        r2 = r2_score(y, p); rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        rho = pearsonr(y, p)[0] if np.std(p) > 1e-9 else 0.0
        a.set_title(f"{m}\nR²={r2:.2f}  RMSE={rmse:.1f}  r={rho:.2f}", fontsize=9)
        a.set_xlabel("observé"); a.set_ylabel("prédit"); a.grid(alpha=.3)
    for j in range(n, rows * cols):
        ax[j // cols][j % cols].axis("off")
    fig.suptitle(f"Prédit vs observé (CV) — {title}"); fig.tight_layout()
    fig.savefig(os.path.join(out, "regression_scatter.png"), dpi=130); return fig

def fig_pred_box(PRED, y, models, out, title):
    fig, ax = plt.subplots(figsize=(1.5 * len(models) + 2, 5))
    data = []
    for m in models:
        data += [PRED[m][y == 0], PRED[m][y == 1]]
    bp = ax.boxplot(data, positions=np.arange(len(data)), widths=0.6, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#4D4F53" if i % 2 == 0 else "#C41E3A"); patch.set_alpha(.55)
    ax.set_xticks([2 * i + 0.5 for i in range(len(models))])
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Prédiction (proba)"); ax.set_title(f"Prédictions par classe (gris=0 / rouge=1) — {title}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "prediction_boxplot.png"), dpi=130); return fig

def fig_features(top_feats, Xdf, y, out, title):
    """Top biomarqueurs : valeur vs outcome (boxplot classif / scatter régression)."""
    if not top_feats:
        return
    reg = (TASK == "regression"); n = len(top_feats); cols = min(5, n); rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows), squeeze=False)
    for i, f in enumerate(top_feats):
        a = ax[i // cols][i % cols]; v = Xdf[f].values
        if reg:
            a.scatter(v, y, s=10, alpha=.5, color="#001A7B")
        else:
            a.boxplot([v[y == 0], v[y == 1]], labels=["0", "1"])
        a.set_title(str(f)[:24], fontsize=8); a.grid(alpha=.3)
    for j in range(n, rows * cols):
        ax[j // cols][j % cols].axis("off")
    fig.suptitle(f"Top biomarqueurs vs outcome — {title}"); fig.tight_layout()
    fig.savefig(os.path.join(out, "features_vs_outcome.png"), dpi=130); return fig


def analyze(key):
    cfg = DATASETS[key]
    print(f"\n=== {key} ===")
    Xtr_raw, ytr, FEAT, Xval_raw, yval = load_dataset(cfg)          # BRUT (std dans le fold)
    P = Xtr_raw.shape[1]
    bal = np.bincount(ytr.astype(int)) if TASK != "regression" else f"DOS [{ytr.min():.0f},{ytr.max():.0f}]"
    v2deco = "knockoff" if P <= KO_MAX_P else "random_permutation (FALLBACK, p>seuil)"
    print(f"  X={Xtr_raw.shape}  task={TASK}  y={bal}  V2_decoy={v2deco}  "
          f"{'val=' + str(Xval_raw.shape) if Xval_raw is not None else 'pas de val -> CV'}")
    # full-data standardisé pour run_one (figures/biomarqueurs sur tout le train)
    Xtr = StandardScaler().fit_transform(Xtr_raw)
    cov = LedoitWolf().fit(Xtr).covariance_ if P <= KO_MAX_P else None   # Σ seulement si knockoffs (p<=seuil)
    SEEDS = [args.seed + i for i in range(NSEEDS)]
    RES, CVRES = [], []
    for i, s in enumerate(SEEDS, 1):
        print(f"  run {i}/{NSEEDS} (seed={s}) — full-data (figures) + score honnête "
              f"({'train->val' if Xval_raw is not None else 'nested CV'}) ...", flush=True)
        RES.append(run_one(s, Xtr, ytr, None, None, P, cov))     # full-data : sélections + scores + figures
        # score honnête : train->val si val séparé (standardisé sur le train), sinon nested CV (std dans le fold)
        if Xval_raw is not None:
            CVRES.append(holdout_auc(Xtr_raw, ytr, Xval_raw, yval, P, s))
        else:
            CVRES.append(nested_cv_auc(Xtr_raw, ytr, P, s))

    OUT = os.path.join(OUT_ROOT, f"{key}_{BASE}_B{B}_d{DELTA:g}"); os.makedirs(OUT, exist_ok=True)
    MODELS = [d["name"] for d in RES[0]["summary"]]
    STABL  = [m for m in MODELS if m in RES[0]["feat"]]
    NR = len(RES)
    TITLE = f"{key} [base={BASE}] — p={P}, n={Xtr.shape[0]}, B={B} | moy. {NR} seeds"
    MET = "R²" if TASK == "regression" else "AUC"

    def marr(name, key2):
        return np.array([next(d[key2] for d in r["summary"] if d["name"] == name) for r in RES])
    NSEL = {m: marr(m, "ns") for m in MODELS}                 # n_sel : full-data (inchangé)
    AUC  = {m: np.array([CVRES[r][m][0] for r in range(NR)]) for m in MODELS}   # AUC par seed (-> std inter-seed)
    _yev = yval if Xval_raw is not None else ytr
    # AGRÉGATION OFFICIELLE : médiane des prédictions OOF sur les seeds -> 1 AUC (point estimate)
    AUCm = {m: float(_score(_yev, np.median([CVRES[r][m][2] for r in range(NR)], axis=0))[0]) for m in MODELS}

    def jac(name):
        # STABL : Jaccard INTER-FOLD (perturbation des patients, nested CV) = vraie fiabilité.
        # Baselines (ou COVID holdout, pas de folds) : fallback inter-seed (full-data).
        folds = [s for r in range(NR) for s in CVRES[r][name][3]]
        S = folds if len(folds) >= 2 else [RES[r]["sel_sets"][name] for r in range(NR)]
        v = []
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
    col = ["Modèle", "# sélec.", MET, "Jaccard"]
    ms = lambda a: f"{a.mean():.2f}±{a.std():.2f}"
    cell = [[m, ms(NSEL[m]), f"{AUCm[m]:.2f}±{AUC[m].std():.2f}", f"{JAC[m]:.2f}"] for m in MODELS]
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
    axes[0].bar(xp, [AUCm[m] for m in MODELS], yerr=[AUC[m].std() for m in MODELS],
                capsize=3, color=[stcol(m) for m in MODELS], alpha=0.85)
    if TASK != "regression":
        axes[0].axhline(0.5, ls=":", color="gray"); axes[0].set_ylim(0, 1.02)
    axes[0].set_xticks(xp); axes[0].set_xticklabels(MODELS, rotation=25, ha="right")
    axes[0].set_ylabel(MET); axes[0].set_title(f"{MET} (Train→Val ou CV) moy.±std"); axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(xp, [JAC[m] for m in MODELS], color=[stcol(m) for m in MODELS], alpha=0.85)
    axes[1].set_ylim(0, 1.02); axes[1].set_xticks(xp); axes[1].set_xticklabels(MODELS, rotation=25, ha="right")
    axes[1].set_ylabel("Jaccard inter-fold (patients)"); axes[1].set_title("Fiabilité : stabilité sous perturbation des patients"); axes[1].grid(axis="y", alpha=0.3)
    fig_b.suptitle(TITLE); fig_b.tight_layout()
    fig_b.savefig(os.path.join(OUT, "auc_jaccard.png"), dpi=130)

    # ── Fig : ROC moyenne (classification uniquement) ─────────────────────────
    fig_r = None
    if TASK != "regression":
        fig_r, ax = plt.subplots(figsize=(7, 6)); fg = np.linspace(0, 1, 101)
        for m in MODELS:
            tprs = []
            for r in range(NR):
                fpr, tpr = CVRES[r][m][1]; t = np.interp(fg, fpr, tpr); t[0] = 0.; tprs.append(t)
            ax.plot(fg, np.mean(tprs, 0), lw=1.8, label=f"{m} (AUC={AUCm[m]:.3f}±{AUC[m].std():.3f})")
        ax.plot([0, 1], [0, 1], ls=":", color="gray"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC moyenne — {TITLE}"); ax.legend(loc="lower right", fontsize=7); ax.grid(alpha=0.3)
        fig_r.tight_layout(); fig_r.savefig(os.path.join(OUT, "roc_curves.png"), dpi=130)

    # ── Plots parité STABL officiel : path/FDR + PRC/scatter/boxplot + features ──
    PRED = {m: np.median([CVRES[r][m][2] for r in range(NR)], axis=0) for m in MODELS}   # médiane (agrégation officielle)
    YEV = yval if Xval_raw is not None else ytr                  # y correspondant aux prédictions OOF/val
    native_figs = plot_native_stabl(RES[0]["models"], REF)      # path + objectif (combinés au PDF)
    eval_figs = []
    if TASK == "regression":
        eval_figs.append(fig_reg_scatter(PRED, YEV, MODELS, OUT, TITLE))
    else:
        eval_figs.append(fig_prc(PRED, YEV, MODELS, OUT, TITLE, stcol))
        eval_figs.append(fig_pred_box(PRED, YEV, MODELS, OUT, TITLE))
    lastref = f"Last_{REF}"
    if lastref in RES[0]["feat"]:                               # top biomarqueurs (Last réf) vs outcome
        sr_mean = np.mean([RES[r]["feat"][lastref]["sr"] for r in range(NR)], axis=0)
        top_idx = np.argsort(sr_mean)[::-1][:10]
        eval_figs.append(fig_features([FEAT[i] for i in top_idx], pd.DataFrame(Xtr, columns=FEAT), ytr, OUT, TITLE))

    # ── Fig : distributions de scores (réel vs knockoff, SANS vraies/nulles) ──
    def score_fig(m, fname):
        if m not in SR: return None
        srm, kom, srs = SRm[m], KOm[m], SRs[m]; sr_all, ko_all = SR[m], KO[m]
        sf = SELF[m]; kolab = "permutations (Y)" if m.startswith("Last") else "knockoffs"
        fig, axe = plt.subplots(1, 2, figsize=(13, 5))
        bins = np.linspace(0, max(sr_all.max(), ko_all.max()) + 0.02, 30)
        axe[0].hist(sr_all.ravel(), bins=bins, alpha=0.5, color="#4D4F53", label="features réelles")
        axe[0].hist(ko_all.ravel(), bins=bins, alpha=0.5, color="#1f77b4", label=kolab)
        axe[0].set_xlabel("Score de stabilité max_λ"); axe[0].set_ylabel("Nombre (poolé)")
        axe[0].set_title(f"Distribution des scores ({m})"); axe[0].legend(fontsize=8)
        sc = axe[1].scatter(kom, srm, c=sf, cmap="viridis", s=18, alpha=0.8)
        lim = max(srm.max(), kom.max()) + 0.05
        axe[1].plot([0, lim], [0, lim], ls=":", color="gray", label="score = score_réf")
        axe[1].set_xlabel(f"score {kolab} moyen"); axe[1].set_ylabel("score réel moyen")
        axe[1].set_title(f"Réel vs {kolab} par feature (couleur = freq. sélection)"); axe[1].legend(fontsize=8)
        plt.colorbar(sc, ax=axe[1], label="fréq. sélection")
        fig.suptitle(f"Scores {m} — {TITLE}"); fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=130); return fig
    fig_sc = score_fig(f"V2_constr_{REF}", "score_distributions.png")
    fig_sc1 = score_fig(f"V1_{REF}", "score_distributions_v1.png")
    fig_scl = score_fig(f"Last_{REF}", "score_distributions_last.png")

    # ── Fig : objectif FDP+ + frontière (SANS vrai FDP) par run + MOYENNE ─────
    def obj_grid_fig(m, fname, title):
        obj = FDPP[m] + FRO[m]; tst = [r["curves"][m]["tstar"] for r in RES]
        nsl = [len(RES[i]["sel_sets"][m]) for i in range(NR)]
        pan = [(FDPP[m][i], obj[i], tst[i], nsl[i], f"run {i+1}", False) for i in range(NR)]
        pan.append((FDPP[m].mean(0), obj.mean(0), float(np.mean(tst)), float(np.mean(nsl)), "MOYENNE", True))
        nc = 3; nr = (len(pan) + nc - 1) // nc
        fig, axe = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.2 * nr), squeeze=False); axe = axe.ravel()
        for a in axe[len(pan):]:
            a.axis("off")
        for a, (fp, oc, ts, ns, ttl, im) in zip(axe, pan):
            if im: a.set_facecolor("#F4F4F4")
            a.plot(GRID, fp, color="#999999", lw=1.0, label="FDP+ seul")
            a.fill_between(GRID, fp, oc, step="post", alpha=0.3, color="#2CA02C", label="frontière")
            a.plot(GRID, oc, color="#1f77b4", lw=2.0, label="objectif")
            a.axvline(ts, color="black", ls=":", lw=1.3, label=f"t*={ts:.2f}")
            a.set_ylim(0, 1.05)
            a.set_title(f"{ttl}  (n={ns:.0f} sél.)", fontsize=9, weight="bold" if im else "normal")
            a.legend(fontsize=6, loc="upper right"); a.grid(alpha=0.25); a.tick_params(labelsize=7)
        fig.suptitle(title, fontsize=10); fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(OUT, fname), dpi=140); return fig
    # fdp_curves : objectif FDP+ + frontière de TOUS les modèles STABL × (runs + moyenne)
    ncf = NR + 1
    fig_fc, axf = plt.subplots(len(STABL), ncf, figsize=(2.5 * ncf, 2.3 * len(STABL)), squeeze=False)
    for ri, name in enumerate(STABL):
        objm = FDPP[name] + FRO[name]; tstm = [r["curves"][name]["tstar"] for r in RES]
        nsl = [len(RES[i]["sel_sets"][name]) for i in range(NR)]
        cols = [(FDPP[name][i], objm[i], tstm[i], nsl[i], f"run {i+1}", False) for i in range(NR)]
        cols.append((FDPP[name].mean(0), objm.mean(0), float(np.mean(tstm)), float(np.mean(nsl)), "MOY.", True))
        for ci, (fp, oc, ts, ns, ttl, im) in enumerate(cols):
            a = axf[ri][ci]
            if im: a.set_facecolor("#F4F4F4")
            a.plot(GRID, fp, color="#999999", lw=0.9)
            a.plot(GRID, oc, color="#1f77b4", lw=1.6 if im else 1.1)
            a.axvline(ts, color="black", ls=":", lw=1.0)
            a.set_ylim(0, 1.05); a.set_xticks([0, .5, 1]); a.tick_params(labelsize=6)
            a.text(0.04, 0.92, f"n={ns:.0f}", transform=a.transAxes, fontsize=6, ha="left", va="top",
                   color="#C41E3A", weight="bold")
            a.text(0.96, 0.92, f"t*={ts:.2f}", transform=a.transAxes, fontsize=6, ha="right", va="top")
            if ri == 0: a.set_title(ttl, fontsize=8, weight="bold" if im else "normal")
            if ci == 0: a.set_ylabel(name, fontsize=8)
    proxy = [Line2D([], [], color="#1f77b4", lw=2, label="objectif FDP+ + frontière"),
             Line2D([], [], color="#999999", lw=1.5, label="FDP+ seul"),
             Line2D([], [], color="black", ls=":", lw=1.2, label="t* du run")]
    fig_fc.legend(handles=proxy, loc="upper center", ncol=3, fontsize=8)
    fig_fc.supxlabel("seuil t", fontsize=9)
    fig_fc.suptitle(f"FDP+ / objectif par modèle et par run — {TITLE}", y=0.998, fontsize=10)
    fig_fc.tight_layout(rect=[0, 0, 1, 0.95]); fig_fc.savefig(os.path.join(OUT, "fdp_curves.png"), dpi=130)

    # objectifs détaillés (par run + moyenne) : V1, V2_constr, Last
    fig_o_v1 = obj_grid_fig(f"V1_{REF}", "v1_objective.png",
                            f"V1_{REF} — objectif FDP+ par run — {TITLE}")
    fig_o_vc = obj_grid_fig(f"V2_constr_{REF}", "v2constr_objective.png",
                            f"V2_constr_{REF} — objectif FDP+ + frontière par run — {TITLE}")
    fig_o = obj_grid_fig(f"Last_{REF}", "last_objective.png",
                         f"Last_{REF} — objectif (|S_perm|+1)/D + frontière par run — {TITLE}")
    fig_o_vca0 = obj_grid_fig(f"V2_constr_a0_{REF}", "v2constr_a0_objective.png",
                              f"V2_constr_a0_{REF} — objectif |S_ko|/D + frontière (SANS +1) — {TITLE}")
    fig_o_la0 = obj_grid_fig(f"Last_a0_{REF}", "last_a0_objective.png",
                             f"Last_a0_{REF} — objectif |S_perm|/D + frontière (SANS +1) — {TITLE}")

    # ── barrière ∂⁺(t*) par run + MOYENNE — fonction réutilisable (V1/V2_constr/Last) ──
    def barrier_fig(m, fname, title):
        if m not in SR: return None
        SRr, KOr = SR[m], KO[m]
        eps = [(r["feat"][m]["eps"] if r["feat"][m].get("eps") is not None else np.zeros(P)) for r in RES]
        tst = [r["curves"][m]["tstar"] for r in RES]; epsmean = EPSm[m] if m in EPSm else np.zeros(P)
        xlab = "score_perm" if m.startswith("Last") else "score knockoff"
        lim = max(SRr.max(), KOr.max(), SRm[m].max(), KOm[m].max()) + 0.04
        pan = [(SRr[i], KOr[i], eps[i], tst[i], f"run {i+1}", False) for i in range(NR)]
        pan.append((SRm[m], KOm[m], epsmean, float(np.mean(tst)), "MOYENNE", True))
        nc = 3; nr = (len(pan) + nc - 1) // nc
        fig, axe = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.0 * nr), squeeze=False); axe = axe.ravel()
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
            a.set_title(f"{ttl}  t*={ts:.2f}, n={int(selm.sum())} sél., |∂⁺|={int(bnd.sum())}",
                        fontsize=8, weight="bold" if im else "normal")
        proxy = [Line2D([], [], marker='o', ls='', color="#888888", label="non sél."),
                 Line2D([], [], marker='o', ls='', color="#2CA02C", label="sél. hors ∂⁺"),
                 Line2D([], [], marker='o', ls='', color="#ff7f0e", label="∂⁺(t*)"),
                 Line2D([], [], color="#1f77b4", ls='--', label="t*")]
        fig.legend(handles=proxy, loc="upper center", ncol=4, fontsize=8)
        fig.supxlabel(xlab, fontsize=9); fig.supylabel("score réel", fontsize=9)
        fig.suptitle(title, fontsize=9, y=0.998)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(OUT, fname), dpi=140); return fig
    fig_bd_v1 = barrier_fig(f"V1_{REF}", "barriere_v1.png", f"Barrière V1 (score vs knockoff) — {TITLE}")
    fig_bd_vc = barrier_fig(f"V2_constr_{REF}", "barriere_v2constr.png", f"Barrière V2_constr (score vs knockoff) — {TITLE}")
    fig_bd = barrier_fig(f"Last_{REF}", "barriere_dplus.png", f"Barrière Last (score vs score_perm) — {TITLE}")
    fig_bd_vca0 = barrier_fig(f"V2_constr_a0_{REF}", "barriere_v2constr_a0.png", f"Barrière V2_constr_a0 (sans +1) — {TITLE}")
    fig_bd_la0 = barrier_fig(f"Last_a0_{REF}", "barriere_last_a0.png", f"Barrière Last_a0 (sans +1) — {TITLE}")

    fig_v = None        # plot de variance/std des scores retiré (peu informatif)

    # ── Fig : top biomarqueurs (freq. sélection) — V2_constr/V1/Last × chaque base ─
    fig_bm, axbm = plt.subplots(len(BASES), 5, figsize=(33, 6 * len(BASES)), squeeze=False)
    for ri, base in enumerate(BASES):
        for ax, (mdl, c) in zip(axbm[ri], [(f"V2_constr_{base}", "#C41E3A"), (f"V2_constr_a0_{base}", "#7A1020"),
                                           (f"V1_{base}", "#1f77b4"),
                                           (f"Last_{base}", "#2CA02C"), (f"Last_a0_{base}", "#145214")]):
            if mdl not in SELF:
                ax.axis("off"); continue
            sf = SELF[mdl]; top = np.argsort(sf)[::-1][:25]; top = top[sf[top] > 0]
            if len(top):
                ax.barh(range(len(top)), sf[top][::-1], color=c, alpha=0.85)
                ax.set_yticks(range(len(top))); ax.set_yticklabels([FEAT[j] for j in top[::-1]], fontsize=7)
                ax.set_xlabel(f"fréquence de sélection (sur {NR} runs)"); ax.set_xlim(0, 1.02)
            ax.set_title(f"Top biomarqueurs {mdl}", fontsize=10)
    fig_bm.suptitle(f"Biomarqueurs sélectionnés (freq. ≥1 run) — {TITLE}", fontsize=11)
    fig_bm.tight_layout(); fig_bm.savefig(os.path.join(OUT, "top_biomarkers.png"), dpi=130)

    # ── Fig : concentration score(j)−score_réf(j) dans ±ε — par run + MOYENNE (Last, V2_constr) ──
    def conc_fig(m, fname, title, reflab):
        if m not in EPSm: return None
        def panel(ax, sc, scp, eps, ttl, im):
            if im: ax.set_facecolor("#F4F4F4")
            diff = sc - scp; inside = np.abs(diff) <= eps
            order = np.argsort(eps); xj = np.arange(len(eps))
            en = eps[order]; dn = diff[order]; ins = inside[order]
            ax.fill_between(xj, -en, en, color="#2CA02C", alpha=0.18)
            ax.plot(xj, en, color="#2CA02C", lw=0.5); ax.plot(xj, -en, color="#2CA02C", lw=0.5)
            ax.scatter(xj[ins], dn[ins], s=4, color="#2CA02C", zorder=3)
            ax.scatter(xj[~ins], dn[~ins], s=9, color="#C41E3A", zorder=4)
            ax.axhline(0, color="gray", ls=":", lw=0.6); ax.set_ylim(-1.05, 1.05)
            ax.set_title(f"{ttl} ({100*inside.mean():.0f}% dans)", fontsize=8, weight="bold" if im else "normal")
            ax.tick_params(labelsize=6)
        pan = [(RES[i]["feat"][m]["sr"], RES[i]["feat"][m]["sr_ko"], RES[i]["feat"][m]["eps"], f"run {i+1}", False)
               for i in range(NR)]
        pan.append((SRm[m], KOm[m], EPSm[m], "MOYENNE", True))
        nc = 3; nr = (len(pan) + nc - 1) // nc
        fig, axe = plt.subplots(nr, nc, figsize=(4.5 * nc, 3.0 * nr), squeeze=False); axe = axe.ravel()
        for a in axe[len(pan):]:
            a.axis("off")
        for a, (sc, scp, eps, ttl, im) in zip(axe, pan):
            panel(a, sc, scp, eps, ttl, im)
        proxy = [Line2D([], [], marker='o', ls='', color="#2CA02C", label="dans la bande ±ε"),
                 Line2D([], [], marker='o', ls='', color="#C41E3A", label="hors bande")]
        fig.legend(handles=proxy, loc="upper center", ncol=2, fontsize=8)
        fig.supxlabel(f"features (triées par ε)  —  réf. nulle : {reflab}", fontsize=9)
        fig.supylabel("score(j) − score_réf(j)", fontsize=9)
        fig.suptitle(title, fontsize=10); fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(OUT, fname), dpi=130); return fig
    fig_lpc = conc_fig(f"Last_{REF}", "last_concentration.png",
                       f"Concentration Last : score−score_perm dans ±(ε_BFW+ε_perm) par run — {TITLE}", "score_perm")
    fig_vcc = conc_fig(f"V2_constr_{REF}", "v2constr_concentration.png",
                       f"Concentration V2_constr : score−score_ko dans ±ε_WJ par run — {TITLE}", "score_ko")

    # ── PDF ───────────────────────────────────────────────────────────────────
    pages = [p for p in [fig_t, fig_b, fig_r, fig_sc, fig_sc1, fig_scl, fig_fc,
                          fig_o_v1, fig_bd_v1, fig_o_vc, fig_bd_vc, fig_o_vca0, fig_bd_vca0, fig_vcc,
                          fig_o, fig_bd, fig_o_la0, fig_bd_la0, fig_lpc,
                          fig_v, fig_bm] if p is not None] + [f for f in eval_figs if f is not None] + native_figs
    with PdfPages(os.path.join(OUT, "analyse_complete.pdf")) as pdf:
        for f in pages:
            pdf.savefig(f)
    for f in pages:
        plt.close(f)

    # ── database.csv local (1 ligne / run × modèle × feature) ─────────────────
    DBC = ["seed", "run", "dataset", "base", "p", "n", "B", "model", "feature", "feature_name",
           "score", "score_ko", "selected", "n_sel", "auc", "auc_mean", "jaccard_mean"]
    with open(os.path.join(OUT, "database.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DBC); w.writeheader()
        for ri, res in enumerate(RES):
            for name in MODELS:
                rm = next(d for d in res["summary"] if d["name"] == name)
                fe = res["feat"].get(name); ss = res["sel_sets"][name]
                brow = dict(seed=SEEDS[ri], run=ri + 1, dataset=key, base=BASE, p=P, n=Xtr.shape[0], B=B,
                            model=name, n_sel=rm["ns"], auc=round(float(CVRES[ri][name][0]), 4),  # nested CV
                            auc_mean=round(AUCm[name], 4), jaccard_mean=round(JAC[name], 4))
                for j in range(P):
                    row = dict(brow, feature=j, feature_name=FEAT[j], selected=int(j in ss))
                    if fe is not None:
                        row["score"] = round(float(fe["sr"][j]), 6); row["score_ko"] = round(float(fe["sr_ko"][j]), 6)
                    else:
                        row["score"] = ""; row["score_ko"] = ""
                    w.writerow(row)
    print(f"  -> {OUT}  (figures + PDF + database.csv)")

    # ── CSV global cumulatif (agrégé) ─────────────────────────────────────────
    GCSV = os.path.join(OUT_ROOT, "runs_metrics.csv")
    gcols = ["timestamp", "dataset", "base", "p", "n", "B", "n_seeds", "model",
             "n_sel_mean", "n_sel_std", "auc_mean", "auc_std", "jaccard_mean"]
    hdr = not os.path.exists(GCSV)
    with open(GCSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gcols)
        if hdr: w.writeheader()
        ts = datetime.now().isoformat(timespec="seconds")
        for m in MODELS:
            w.writerow(dict(timestamp=ts, dataset=key, base=BASE, p=P, n=Xtr.shape[0], B=B, n_seeds=NR,
                            model=m, n_sel_mean=round(NSEL[m].mean(), 3), n_sel_std=round(NSEL[m].std(), 3),
                            auc_mean=round(AUCm[m], 4), auc_std=round(AUC[m].std(), 4),
                            jaccard_mean=round(JAC[m], 4)))


def analyze_late(key):
    """LATE FUSION : STABL par omique (FDP+ contrôlé par modalité) -> UNION des biomarqueurs
    -> modèle prédictif combiné. AUC/Jaccard sur l'union ; scores par omique."""
    cfg = DATASETS[key]; print(f"\n=== {key} (late fusion) ===")
    omics, y = load_omics(cfg)                                # BRUT (std dans le fold pour la nested CV)
    offsets, feat_names, parts_std, covs, omics_std = [], [], [], [], []
    off = 0
    for lab, Xs, names in omics:
        offsets.append(off); off += Xs.shape[1]
        feat_names += [f"{lab}:{n}" for n in names]
        Xs_std = StandardScaler().fit_transform(Xs)           # full-data std (figures/biomarqueurs)
        omics_std.append((lab, Xs_std, names)); parts_std.append(Xs_std)
        covs.append(LedoitWolf().fit(Xs_std).covariance_ if Xs_std.shape[1] <= KO_MAX_P else None)
    Xcat = np.hstack(parts_std); Ptot = Xcat.shape[1]
    bal = np.bincount(y.astype(int)) if TASK != "regression" else f"DOS [{y.min():.0f},{y.max():.0f}]"
    print(f"  omiques: {[(l, X.shape[1]) for l, X, _ in omics]}  p_total={Ptot}  task={TASK}  y={bal}")
    SEEDS = [args.seed + i for i in range(NSEEDS)]
    RES, PEROMIC, CVRES = [], [], []
    for i, s in enumerate(SEEDS, 1):
        print(f"  run {i}/{NSEEDS} (seed={s}) — full-data + nested CV ...", flush=True)
        po = [run_one(s, Xs, y, None, None, Xs.shape[1], covs[oi])
              for oi, (lab, Xs, names) in enumerate(omics_std)]
        PEROMIC.append(po)
        mres = {}                                      # full-data : union des sélections (biomarqueurs, Jaccard)
        for m in [d["name"] for d in po[0]["summary"]]:
            gsel = sorted({offsets[oi] + j for oi, ro in enumerate(po) for j in ro["sel_sets"][m]})
            mres[m] = dict(sel=set(gsel), ns=len(gsel))
        RES.append(mres)
        CVRES.append(nested_cv_auc_late(omics, offsets, Xcat, y, s))   # AUC honnête (nested CV)
    MODELS = [d["name"] for d in PEROMIC[0][0]["summary"]]
    STABL  = [m for m in MODELS if m in PEROMIC[0][0]["feat"]]
    NR = len(RES)
    OUT = os.path.join(OUT_ROOT, f"{key}_{BASE}_B{B}_d{DELTA:g}"); os.makedirs(OUT, exist_ok=True)
    TITLE = f"{key} (late fusion) [base={BASE}] — p={Ptot}, n={len(y)}, B={B} | {NR} seeds"
    NSEL = {m: np.array([RES[r][m]["ns"] for r in range(NR)]) for m in MODELS}     # full-data (union)
    AUC  = {m: np.array([CVRES[r][m][0] for r in range(NR)]) for m in MODELS}        # AUC par seed (std)
    AUCm = {m: float(_score(y, np.median([CVRES[r][m][2] for r in range(NR)], axis=0))[0]) for m in MODELS}  # médiane préds OOF
    def jac(m):
        folds = [s for r in range(NR) for s in CVRES[r][m][3]]      # inter-fold (perturbation patients)
        S = folds if len(folds) >= 2 else [RES[r][m]["sel"] for r in range(NR)]
        v = []
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                u = len(S[a] | S[b]); v.append(1.0 if u == 0 else len(S[a] & S[b]) / u)
        return float(np.mean(v)) if v else 1.0
    JAC = {m: jac(m) for m in MODELS}
    def sel_freq(m):
        s = np.zeros(Ptot)
        for r in range(NR):
            for j in RES[r][m]["sel"]:
                s[j] += 1
        return s / NR
    SELFREQ = {m: sel_freq(m) for base in BASES
               for m in [f"V2_constr_{base}", f"V2_constr_a0_{base}", f"V1_{base}",
                         f"Last_{base}", f"Last_a0_{base}"] if m in MODELS}
    stcol = lambda n: "#C41E3A" if n.startswith("V") else "#4D4F53"
    MET = "R²" if TASK == "regression" else "AUC"

    # table
    fig_t, ax = plt.subplots(figsize=(9, 0.8 + 0.4 * len(MODELS))); ax.axis("off")
    ms = lambda a: f"{a.mean():.2f}±{a.std():.2f}"
    tb = ax.table(cellText=[[m, ms(NSEL[m]), f"{AUCm[m]:.2f}±{AUC[m].std():.2f}", f"{JAC[m]:.2f}"] for m in MODELS],
                  colLabels=["Modèle", "# sélec. (union)", MET, "Jaccard"], loc="center", cellLoc="center")
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
    axe[0].bar(xp, [AUCm[m] for m in MODELS], yerr=[AUC[m].std() for m in MODELS], capsize=3,
               color=[stcol(m) for m in MODELS], alpha=0.85)
    if TASK != "regression":
        axe[0].axhline(0.5, ls=":", color="gray"); axe[0].set_ylim(0, 1.02)
    axe[0].set_xticks(xp); axe[0].set_xticklabels(MODELS, rotation=25, ha="right")
    axe[0].set_ylabel(MET); axe[0].set_title(f"{MET} (union des biomarqueurs, CV)"); axe[0].grid(axis="y", alpha=0.3)
    axe[1].bar(xp, [JAC[m] for m in MODELS], color=[stcol(m) for m in MODELS], alpha=0.85)
    axe[1].set_ylim(0, 1.02); axe[1].set_xticks(xp); axe[1].set_xticklabels(MODELS, rotation=25, ha="right")
    axe[1].set_ylabel("Jaccard inter-fold (union)"); axe[1].set_title("Fiabilité (perturbation patients)"); axe[1].grid(axis="y", alpha=0.3)
    fig_b.suptitle(TITLE); fig_b.tight_layout(); fig_b.savefig(os.path.join(OUT, "auc_jaccard.png"), dpi=130)

    # roc (classification uniquement)
    fig_r = None
    if TASK != "regression":
        fig_r, ax = plt.subplots(figsize=(7, 6)); fg = np.linspace(0, 1, 101)
        for m in MODELS:
            tprs = [np.interp(fg, CVRES[r][m][1][0], CVRES[r][m][1][1]) for r in range(NR)]
            for t in tprs:
                t[0] = 0.
            ax.plot(fg, np.mean(tprs, 0), lw=1.8, label=f"{m} (AUC={AUCm[m]:.3f})")
        ax.plot([0, 1], [0, 1], ls=":", color="gray"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC moyenne (union) — {TITLE}"); ax.legend(loc="lower right", fontsize=7); ax.grid(alpha=0.3)
        fig_r.tight_layout(); fig_r.savefig(os.path.join(OUT, "roc_curves.png"), dpi=130)

    # ── Plots parité STABL officiel (late) : path/FDR par omique + PRC/scatter/boxplot ──
    PRED = {m: np.median([CVRES[r][m][2] for r in range(NR)], axis=0) for m in MODELS}   # médiane (agrégation officielle)
    native_figs = []
    for oi, (lab, _, _) in enumerate(omics):                    # path/objectif par omique (V1/V2/Last réf)
        native_figs += plot_native_stabl(PEROMIC[0][oi]["models"], REF, prefix=f"{lab}_")
    eval_figs = []
    if TASK == "regression":
        eval_figs.append(fig_reg_scatter(PRED, y, MODELS, OUT, TITLE))
    else:
        eval_figs.append(fig_prc(PRED, y, MODELS, OUT, TITLE, stcol))
        eval_figs.append(fig_pred_box(PRED, y, MODELS, OUT, TITLE))
    sr = SELFREQ.get(f"Last_{REF}")
    if sr is not None:
        top_idx = np.argsort(sr)[::-1][:10]
        eval_figs.append(fig_features([feat_names[i] for i in top_idx], pd.DataFrame(Xcat, columns=feat_names), y, OUT, TITLE))

    # top biomarqueurs (union, taggés par omique) — V2_constr/V1/Last × chaque base
    fig_bm, axbm = plt.subplots(len(BASES), 5, figsize=(34, 6 * len(BASES)), squeeze=False)
    for ri, base in enumerate(BASES):
        for ax, mdl in zip(axbm[ri], [f"V2_constr_{base}", f"V2_constr_a0_{base}", f"V1_{base}",
                                      f"Last_{base}", f"Last_a0_{base}"]):
            if mdl not in SELFREQ:
                ax.axis("off"); continue
            sf = SELFREQ[mdl]; top = np.argsort(sf)[::-1][:25]; top = top[sf[top] > 0]
            if len(top):
                cols = ["#C41E3A" if feat_names[j].startswith("Prot") else "#1f77b4" for j in top[::-1]]
                ax.barh(range(len(top)), sf[top][::-1], color=cols, alpha=0.85)
                ax.set_yticks(range(len(top))); ax.set_yticklabels([feat_names[j] for j in top[::-1]], fontsize=7)
                ax.set_xlabel(f"fréq. sélection ({NR} runs)"); ax.set_xlim(0, 1.02)
            ax.legend(handles=[Line2D([], [], color="#C41E3A", lw=6, label="Prot"),
                               Line2D([], [], color="#1f77b4", lw=6, label="CyTOF")], fontsize=8)
            ax.set_title(f"Top biomarqueurs {mdl} (union)", fontsize=10)
    fig_bm.suptitle(f"Biomarqueurs (union late fusion) — {TITLE}", fontsize=11)
    fig_bm.tight_layout(); fig_bm.savefig(os.path.join(OUT, "top_biomarkers.png"), dpi=130)

    # distributions de scores PAR OMIQUE (V2_constr)
    omic_figs = []
    for oi, (lab, Xs, names) in enumerate(omics):
        SRo = np.vstack([PEROMIC[r][oi]["feat"][f"V2_constr_{REF}"]["sr"] for r in range(NR)])
        KOo = np.vstack([PEROMIC[r][oi]["feat"][f"V2_constr_{REF}"]["sr_ko"] for r in range(NR)])
        fig, axx = plt.subplots(figsize=(7, 5))
        bins = np.linspace(0, max(SRo.max(), KOo.max()) + 0.02, 30)
        axx.hist(SRo.ravel(), bins=bins, alpha=0.5, color="#4D4F53", label="réelles")
        axx.hist(KOo.ravel(), bins=bins, alpha=0.5, color="#1f77b4", label="knockoffs")
        axx.set_xlabel("score de stabilité"); axx.set_ylabel("nombre")
        axx.set_title(f"Scores {lab} (V2_constr) — {key}"); axx.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"score_distributions_{lab}.png"), dpi=130)
        omic_figs.append(fig)

    pages = [fig_t, fig_b, fig_r, fig_bm] + [f for f in eval_figs if f is not None] + omic_figs + native_figs
    with PdfPages(os.path.join(OUT, "analyse_complete.pdf")) as pdf:
        for f in pages:
            pdf.savefig(f)
    for f in pages:
        plt.close(f)

    # database.csv (1 ligne / run × modèle × feature global)
    DBC = ["seed", "run", "dataset", "base", "p", "n", "B", "model", "feature", "feature_name",
           "omic", "score", "selected", "n_sel", "auc", "auc_mean", "jaccard_mean"]
    with open(os.path.join(OUT, "database.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DBC); w.writeheader()
        for ri in range(NR):
            for m in MODELS:
                sel = RES[ri][m]["sel"]
                brow = dict(seed=SEEDS[ri], run=ri + 1, dataset=key, base=BASE, p=Ptot, n=len(y), B=B, model=m,
                            n_sel=RES[ri][m]["ns"], auc=round(float(CVRES[ri][m][0]), 4),   # nested CV
                            auc_mean=round(AUCm[m], 4), jaccard_mean=round(JAC[m], 4))
                for j in range(Ptot):
                    oi = max(k for k in range(len(offsets)) if offsets[k] <= j)
                    loc = j - offsets[oi]; lab = omics[oi][0]
                    fe = PEROMIC[ri][oi]["feat"].get(m)
                    sc = round(float(fe["sr"][loc]), 6) if fe is not None else ""
                    w.writerow(dict(brow, feature=j, feature_name=feat_names[j], omic=lab,
                                    score=sc, selected=int(j in sel)))
    print(f"  -> {OUT}  (figures + PDF + database.csv)")

    GCSV = os.path.join(OUT_ROOT, "runs_metrics.csv")
    gcols = ["timestamp", "dataset", "base", "p", "n", "B", "n_seeds", "model",
             "n_sel_mean", "n_sel_std", "auc_mean", "auc_std", "jaccard_mean"]
    hdr = not os.path.exists(GCSV)
    with open(GCSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gcols)
        if hdr: w.writeheader()
        ts = datetime.now().isoformat(timespec="seconds")
        for m in MODELS:
            w.writerow(dict(timestamp=ts, dataset=key, base=BASE, p=Ptot, n=len(y), B=B, n_seeds=NR, model=m,
                            n_sel_mean=round(NSEL[m].mean(), 3), n_sel_std=round(NSEL[m].std(), 3),
                            auc_mean=round(AUCm[m], 4), auc_std=round(AUC[m].std(), 4),
                            jaccard_mean=round(JAC[m], 4)))


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for k in keys:
        if DATASETS[k]["fusion"] == "late":
            analyze_late(k)
        else:
            analyze(k)
    print("\nTerminé.")
