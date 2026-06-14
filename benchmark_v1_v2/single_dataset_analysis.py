"""
single_dataset_analysis.py
Analyse AGRÉGÉE sur N_SEEDS datasets synthétiques (5 par défaut), pour STABL
(V1, V2_constr, Last) × bases (lasso, alasso)
+ baselines (ALasso, Lasso, ElasticNet, XGBoost).

À chaque appel : on lance N_SEEDS runs (seeds différentes -> datasets différents) et
analyse_complete.pdf contient les DONNÉES MOYENNES et COURBES MOYENNES sur ces runs,
plus la VARIANCE des scores par feature et le JACCARD inter-run des sélections.

Variantes (offset du numérateur de FDP+ : num=(1/r)|S_ko|+alpha) :
  V2_constr      alpha=1   (offset Barber-Candès standard, knockoffs)
  Last           PAS de knockoffs ; référence nulle = PERMUTATION DE Y fraîche par bootstrap
                 (B perms, Thm 4bis). Objectif = (|S_perm(t)|+1)/D + frontière(ε_B,j+ε_B,j,perm).

Paramétrable :
  --artificial knockoff|perm  --n --p --k --signal --B --seed --n-seeds

Figures (results_synthetic/<tag>_n<N>_p<P>_k<K>_sig<SIGNAL>_B<B>/) + analyse_complete.pdf :
  selection_table, precision_recall_f1, roc_curves, score_distributions(V2/V1),
  fdp_par_modele, fdp_curves, last_objective, barriere_dplus, score_variance, jaccard,
  last_perm_validation (score−score_perm dans la bande ±(ε_BFW+ε_perm) sur les nulles).
  Tout est moyenné sur les N_SEEDS runs (± écart-type).

Mémoire cumulative (2 fichiers FIXES, append à chaque appel) — agrégés sur les N_SEEDS :
  benchmark_v1_v2/runs_metrics.csv         — 1 ligne / (appel × modèle) : params, n_seeds,
    seeds, puis mean+std de n_sel, n_true, precision, recall, f1, fdp, auc, et jaccard_mean.
  benchmark_v1_v2/runs_feature_scores.csv  — 1 ligne / (appel × modèle STABL × feature) :
    params, model, feature, is_true, score_mean, score_std, score_ko_mean, sel_freq.
En plus, dans CHAQUE dossier de config : database.csv — format long 1 ligne/(run×modèle×
  feature), TOUTES les infos PAR RUN (score, score_ko, selected, is_true) + métriques du run
  (n_sel, precision, recall, f1, fdp, auc) + moyennes (*_mean, jaccard_mean). Base de données
  complète pour faire n'importe quel plot de cette config.
"""
import os, sys, argparse, warnings, copy, csv
from datetime import datetime
warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.base import clone
from joblib import Parallel, delayed

from stabl.stabl import Stabl as StablV1, plot_stabl_path, plot_fdr_graph
from stabl.stablV2 import Stabl as StablV2, fit_bootstrapped_sample
from stabl.adaptive import ALogitLasso
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── arguments ─────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--artificial", choices=["knockoff", "perm"], default="knockoff")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--p", type=int, default=200)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--signal", type=float, default=10.0)
ap.add_argument("--B", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-seeds", "--n_seeds", type=int, default=5, dest="n_seeds",
                help="nombre de runs/datasets à moyenner")
ap.add_argument("--base", choices=["all", "lasso", "alasso"], default="all",
                help="estimateur(s) de base de STABL ; 'all' = lasso+alasso (1 modèle chacun)")
ap.add_argument("--only-last", "--only_last", action="store_true", dest="only_last",
                help="ne fitter que Last (× bases) — pas V1 ni V2_constr")
ap.add_argument("--delta", type=float, default=0.001,
                help="niveau de risque des bornes de concentration (garantie 1-δ) ; défaut 0.001 (99.9%%)")
ap.add_argument("--ko-max-p", "--ko_max_p", type=int, default=2000, dest="ko_max_p",
                help="V2 : knockoffs si p<=seuil, sinon FALLBACK random_permutation ; défaut 2000")
# ── Réalisme « façon labo » ──
ap.add_argument("--rho", type=float, default=0.9,
                help="corrélation intra-bloc MAX (défaut RÉALISTE 0.9 ; 0=indépendant). Chaque bloc tire son rho dans [rho_min, rho].")
ap.add_argument("--rho-min", "--rho_min", type=float, default=0.2, dest="rho_min",
                help="corrélation intra-bloc MIN (défaut 0.2) ; rho_b ~ Uniform[rho_min, rho] -> rhos hétérogènes par groupe")
ap.add_argument("--block-size", "--block_size", type=int, default=50, dest="block_size",
                help="taille des blocs corrélés (ex. 50 = voie/panel)")
ap.add_argument("--distribution", choices=["gaussian", "lognormal", "counts"], default="lognormal",
                help="distribution observée : lognormal (défaut, protéo/métabo) / counts (microbiome/cfRNA) / gaussian")
ap.add_argument("--prevalence", type=float, default=0.3,
                help="fraction visée de positifs (défaut RÉALISTE 0.3 ; déséquilibré comme en clinique)")
ap.add_argument("--nval", type=int, default=2000,
                help="taille du set de validation (held-out, mêmes caractéristiques) pour l'AUC")
args = ap.parse_args()

N, P, K, SIGNAL, B, SEED = args.n, args.p, args.k, args.signal, args.B, args.seed
NSEEDS = args.n_seeds; BASE = args.base; ONLY_LAST = args.only_last; DELTA = args.delta; KO_MAX_P = args.ko_max_p
RHO, RHO_MIN, BLOCK, DIST, PREVALENCE, NVAL = args.rho, args.rho_min, args.block_size, args.distribution, args.prevalence, args.nval
BASES = ["lasso", "alasso"] if BASE == "all" else [BASE]
REF = BASES[0]                          # base de référence pour les figures détaillées
ART = "random_permutation" if args.artificial == "perm" else "knockoff"
TAG = "perm" if args.artificial == "perm" else "ko"
ART_LABEL = "permutation" if args.artificial == "perm" else "knockoff"

CSV_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_metrics.csv")
FEAT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_feature_scores.csv")

# ── Sélection de N_SEEDS seeds neuves (non déjà utilisées pour ces paramètres) ─
def _seeds_used_for_params():
    used = set()
    if not os.path.exists(CSV_PATH):
        return used
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if (row["artificial"] == ART_LABEL and int(row["n"]) == N
                        and int(row["p"]) == P and int(row["k"]) == K
                        and float(row["signal"]) == float(SIGNAL)
                        and int(row["n_bootstraps"]) == B
                        and row.get("base", "lasso") == BASE):
                    for s in str(row.get("seeds", "")).split(";"):
                        if s.strip():
                            used.add(int(s))
            except (KeyError, ValueError):
                continue
    return used

_used = _seeds_used_for_params()
_pick = np.random.default_rng()
SEEDS = []
_cand = SEED
while len(SEEDS) < NSEEDS:
    if _cand not in _used and _cand not in SEEDS:
        SEEDS.append(_cand)
    _cand = int(_pick.integers(0, 2**31 - 1))
print(f"Seeds utilisées ({NSEEDS}) : {SEEDS}")

_realtag = (f"_rho{RHO:g}_{DIST}_prev{PREVALENCE:g}"
            if (RHO > 0 or DIST != "gaussian" or PREVALENCE != 0.5) else "")
FOLDER = f"{TAG}_n{N}_p{P}_k{K}_sig{SIGNAL:g}{_realtag}_B{B}_{BASE}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_synthetic", FOLDER)
os.makedirs(OUT, exist_ok=True)
print(f"Dataset: artificial={ART}, n={N}, p={P}, k={K}, signal={SIGNAL}, B={B}, n_seeds={NSEEDS}")
print(f"  réalisme: rho∈[{RHO_MIN},{RHO}] par bloc (taille {BLOCK}), distribution={DIST}, prévalence={PREVALENCE}, validation={NVAL}")
print(f"Vraies features = indices 0..{K-1}\nDossier de sortie : {OUT}\n")

GRID = np.arange(0., 1., .01)   # grille de seuils commune (fdr_threshold_range)


# ── helpers (utilisent K global) ──────────────────────────────────────────────
def get_support(model, name=None):
    if hasattr(model, "get_support"):
        return np.where(model.get_support())[0]
    est = model.best_estimator_
    if hasattr(est, "coef_"):
        return np.where(np.abs(est.coef_[0]) > 1e-8)[0]
    return np.where(est.feature_importances_ > 0)[0]

def mp_eps(score_variance, P, B):
    """Tolérance feature-wise Maurer-Pontil (empirical Bernstein) sur les scores."""
    s2 = score_variance.max(axis=1); Kl = score_variance.shape[1]
    L = max(np.log(4 * P * Kl / DELTA), 0.0); Beff = max(B - 1, 1)
    return ((7/3)*L + np.sqrt(np.maximum((7/3)**2*L**2 + 8*Beff*L*s2, 0.0))) / (2*Beff)


def perm_null_scores(X, y, base, lambda_grid_dict, B, frac, seed):
    """Référence nulle de Last : permutation de Y FRAÎCHE par bootstrap (B permutations,
    Thm 4bis). Réutilise la sélection exacte de STABL. Renvoie score_perm(j)=max_λ freq_perm
    et la variance par (feature, λ) pour Maurer-Pontil."""
    n, P = X.shape; ss = max(2, int(round(frac * n)))
    ldicts = list(ParameterGrid(lambda_grid_dict)); K = len(ldicts); y = np.asarray(y)

    def _boot(b):
        rng = np.random.default_rng([seed, b])
        idx = rng.choice(n, ss, replace=False)
        Xb, yb = X[idx], y[idx]; yp = yb[rng.permutation(ss)]      # permutation fraîche par bootstrap
        out = np.empty((K, P))
        for ki, ld in enumerate(ldicts):
            out[ki] = fit_bootstrapped_sample(clone(base), Xb, yp, lambda_val=ld, threshold=None)
        return out
    sel = np.asarray(Parallel(n_jobs=-1)(delayed(_boot)(b) for b in range(B)))   # (B, K, P)
    return sel.mean(axis=0).max(axis=0), sel.var(axis=0).T     # score_perm (P,), variance (P, K)

def true_fdp_curve(sr, grid):
    out = []
    for t in grid:
        S = np.where(sr > t)[0]
        out.append(((S >= K).sum() / max(1, len(S))) if len(S) else 0.0)
    return np.array(out)

def second_local_min_threshold(obj, grid, tol=1e-9):
    """Cas particulier a0 : si l'objectif a UN SEUL plateau de zéros qui se termine
    EXACTEMENT à 1 (le plateau atteint le dernier point de la grille), renvoie le t du
    2e minimum local (par valeur : le 1er = le plateau de zéros) ; sinon None."""
    obj = np.asarray(obj, float); n = len(obj)
    zero = obj <= tol
    regions = []; i = 0                              # régions contiguës de zéros
    while i < n:
        if zero[i]:
            j = i
            while j + 1 < n and zero[j + 1]:
                j += 1
            regions.append((i, j)); i = j + 1
        else:
            i += 1
    if len(regions) != 1 or regions[0][1] != n - 1:
        return None                                  # pas 'un seul plateau qui termine à 1'
    zero_idx = set(range(regions[0][0], regions[0][1] + 1))   # le plateau (= 1er minimum) à exclure
    locmins = []                                     # minima locaux discrets HORS plateau (et hors bords)
    for i in range(1, n - 1):
        if i in zero_idx:
            continue
        if obj[i] <= obj[i - 1] and obj[i] <= obj[i + 1] and (obj[i] < obj[i - 1] or obj[i] < obj[i + 1]):
            locmins.append((obj[i], i))
    if not locmins:
        return None
    locmins.sort(key=lambda x: (x[0], x[1]))         # le meilleur creux hors plateau = le 2e minimum
    return float(grid[locmins[0][1]])


# ── 1 run complet sur 1 dataset (seed) -> dict de résultats ───────────────────
def run_one(seed):
    # Génération « façon labo » : features CORRÉLÉES par blocs (RHO), distribution observée (DIST),
    # déséquilibre (PREVALENCE). Vraies features 0..K-1 = signal, noyées dans des blocs corrélés
    # (= biomarqueurs entourés de voisins nuls corrélés). Train (N) + validation held-out (NVAL)
    # tirés du MÊME process aléatoire (covariance knockoff ESTIMÉE par LedoitWolf, comme en réel).
    rng = np.random.default_rng(seed)
    Ntot = N + NVAL
    # 1) features gaussiennes corrélées par blocs (facteur latent partagé -> corr ~ RHO dans le bloc)
    Xg = np.empty((Ntot, P))
    for b0 in range(0, P, BLOCK):
        cols = slice(b0, min(b0 + BLOCK, P)); nc = cols.stop - cols.start
        rho_b = rng.uniform(RHO_MIN, RHO) if RHO > 0 else 0.0        # rho PROPRE à ce bloc (hétérogène)
        factor = rng.standard_normal((Ntot, 1))
        Xg[:, cols] = np.sqrt(rho_b) * factor + np.sqrt(1.0 - rho_b) * rng.standard_normal((Ntot, nc))
    # 2) signal (échelle latente) + déséquilibre via quantile
    beta = np.zeros(P); beta[:K] = SIGNAL / np.sqrt(K)
    yl = Xg[:, :K] @ beta[:K] + rng.standard_normal(Ntot)
    y = (yl > np.quantile(yl, 1.0 - PREVALENCE)).astype(int)
    # 3) distribution observée des features
    if DIST == "lognormal":
        X = np.exp(Xg)                                              # abondances positives skewed (protéo/métabo)
    elif DIST == "counts":
        X = rng.poisson(np.exp(Xg - 1.0)).astype(float)            # counts zéro-inflatés (microbiome/cfRNA)
    else:
        X = Xg
    Xtr, Xte, ytr, yte = X[:N], X[N:], y[:N], y[N:]                 # Xte = set de validation held-out (mêmes carac.)
    cov = LedoitWolf().fit(Xtr).covariance_ if (ART == "knockoff" and P <= KO_MAX_P) else None

    def make_base(b):
        if b == "alasso":
            return ALogitLasso(solver="liblinear", class_weight="balanced", tol=1e-4,
                               max_iter=int(1e6), random_state=seed)
        return LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                  max_iter=int(1e6), random_state=seed)

    def stabl_v1(b):
        # V1 = STABL OFFICIEL : leurres par permutation de colonnes (random_permutation), pas de Σ
        return StablV1(base_estimator=make_base(b), lambda_grid="auto", n_lambda=10, n_bootstraps=B,
                       artificial_type="random_permutation", artificial_proportion=1.0, sample_fraction=0.5,
                       replace=False, n_jobs=-1, random_state=seed)

    def stabl_v2(b, mode, alpha=1.0):
        art = "random_permutation" if (ART == "knockoff" and P > KO_MAX_P) else ART   # fallback si trop de colonnes
        kw = dict(base_estimator=make_base(b), lambda_grid="auto", n_lambda=10, n_bootstraps=B,
                  artificial_type=art, artificial_proportion=1.0, sample_fraction=0.5,
                  replace=False, n_jobs=-1, random_state=seed, selection_mode=mode,
                  delta=DELTA, alpha=alpha)
        if art == "knockoff":
            kw.update(knockoff_method="equicorrelated", cov_matrix=cov)
        return StablV2(**kw)

    def baseline(est, grid):
        n_splits = max(2, min(5, int(np.min(np.bincount(ytr)))))
        return GridSearchCV(est, grid, cv=n_splits, scoring="roc_auc", n_jobs=-1)

    _CG = {"C": np.logspace(-2, 0, 10)}
    fitted = {}
    for b in BASES:                                   # une version × chaque base de --base
        if not ONLY_LAST:                             # --only-last : ne fitter que Last
            v1 = stabl_v1(b); v1.fit(Xtr, ytr); fitted[f"V1_{b}"] = v1
            vc = stabl_v2(b, "constrained"); vc.fit(Xtr, ytr); fitted[f"V2_constr_{b}"] = vc
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
            # sinon : FALLBACK automatique sur V2_constr (avec +1)
            fitted[f"V2_constr_a0_{b}"] = vca0
        # Last : score(j) réel + référence nulle = PERMUTATION DE Y fraîche par bootstrap (B perms)
        real = StablV2(base_estimator=make_base(b), lambda_grid="auto", n_lambda=10, n_bootstraps=B,
                       artificial_type=None, sample_fraction=0.5, replace=False, n_jobs=-1,
                       random_state=seed, selection_mode="unconstrained", delta=DELTA, hard_threshold=0.5)
        real.fit(Xtr, ytr)
        srn = real.stabl_scores_.max(axis=1); gn = np.asarray(real.fdr_threshold_range)
        Dn = np.array([max(1, int((srn > t).sum())) for t in gn])
        epsn = mp_eps(real.score_variance_, P, B)             # ε_B,j
        last = copy.deepcopy(real)
        sperm, var_perm = perm_null_scores(Xtr, ytr, make_base(b),
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
        # sinon : FALLBACK automatique sur Last (avec +1)
        fitted[f"Last_a0_{b}"] = last_a0
    # baselines
    bl = {
        "ALasso":     baseline(ALogitLasso(solver="liblinear", class_weight="balanced",
                                           tol=1e-4, max_iter=int(1e6)), _CG),
        "Lasso":      baseline(LogisticRegression(penalty="l1", solver="liblinear",
                               class_weight="balanced", max_iter=int(1e6)), _CG),
        "ElasticNet": baseline(LogisticRegression(penalty="elasticnet", solver="saga",
                               l1_ratio=0.5, class_weight="balanced", max_iter=int(1e6)), _CG),
    }
    if HAS_XGB:
        bl["XGBoost"] = baseline(XGBClassifier(eval_metric="logloss", random_state=42,
                                               n_jobs=1, verbosity=0),
                                 {"n_estimators": [100, 300], "max_depth": [3, 5],
                                  "learning_rate": [0.05, 0.2]})
    for nm, m in bl.items():
        m.fit(Xtr, ytr); fitted[nm] = m

    # évaluation
    summary, roc, feat, curves, sel_sets = [], {}, {}, {}, {}
    for name, model in fitted.items():
        sel = get_support(model, name); ns = len(sel)
        n_true = int((sel < K).sum()); n_null = ns - n_true
        fdp = n_null / max(1, ns)
        precision = n_true / ns if ns > 0 else 0.0
        recall = n_true / K
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if getattr(model, "stabl_scores_", None) is None:    # baseline -> son propre modèle entraîné
            prob = model.predict_proba(Xte)[:, 1]            # fit sur Xtr (run_one), éval sur test held-out
            auc = roc_auc_score(yte, prob); fpr, tpr, _ = roc_curve(yte, prob)
        elif ns > 0:                                          # STABL -> features + logistique
            l2 = name.startswith(("V2_constr", "Last"))       # L2 pour V2/Last/a0 (anti sur-apprentissage) ; V1 -> penalty=None
            clf = LogisticRegression(penalty=("l2" if l2 else None), solver="lbfgs",
                                     class_weight="balanced", max_iter=int(1e6), random_state=42)
            clf.fit(Xtr[:, sel], ytr)
            prob = clf.predict_proba(Xte[:, sel])[:, 1]
            auc = roc_auc_score(yte, prob); fpr, tpr, _ = roc_curve(yte, prob)
        else:
            prob = np.full(len(yte), 0.5)
            auc = 0.5; fpr, tpr = np.array([0., 1.]), np.array([0., 1.])
        summary.append(dict(name=name, ns=ns, n_true=n_true, n_null=n_null, fdp=fdp,
                            precision=precision, recall=recall, f1=f1, auc=auc))
        roc[name] = (np.asarray(fpr), np.asarray(tpr), auc, np.asarray(prob))   # +prédictions (PRC/boxplot)
        sel_sets[name] = set(int(j) for j in sel)
        scores = getattr(model, "stabl_scores_", None)
        if scores is not None:
            sr = scores.max(axis=1)
            sko_a = getattr(model, "stabl_scores_artificial_", None)
            if getattr(model, "score_perm_", None) is not None:        # Last : réf nulle = score_perm
                sr_ko = model.score_perm_
            elif sko_a is not None:
                sr_ko = sko_a.max(axis=1)
            else:
                sr_ko = np.zeros(P)
            selmask = np.zeros(P, bool); selmask[sel] = True
            g = np.asarray(model.fdr_threshold_range)
            eps = getattr(model, "eps_B_total_fw_", None)   # V1 : pas de frontière feature-wise
            feat[name] = {"sr": sr, "sr_ko": sr_ko, "selected": selmask, "eps": eps}
            if eps is not None:
                Dg = np.array([max(1, int((sr > t).sum())) for t in g])
                frog = np.array([((sr > t) & (sr <= t + eps)).sum() for t in g]) / Dg
            else:
                frog = np.zeros(len(g))
            fdpp = np.asarray(model.FDRs_) if getattr(model, "FDRs_", None) is not None \
                else true_fdp_curve(sr, g)
            curves[name] = {"grid": g, "fdp_plus": fdpp,
                            "fdp_true": true_fdp_curve(sr, g), "fro": frog,
                            "tstar": float(model.fdr_min_threshold_)}
    stabl_models = {k: v for k, v in fitted.items() if getattr(v, "stabl_scores_", None) is not None}
    return dict(summary=summary, roc=roc, feat=feat, curves=curves, sel_sets=sel_sets,
                models=stabl_models, yte=yte, Xte=Xte)


# ── boucle sur les seeds ──────────────────────────────────────────────────────
RESULTS = []
for i, s in enumerate(SEEDS, 1):
    print(f"  run {i}/{NSEEDS}  (seed={s}) ...", flush=True)
    RESULTS.append(run_one(s))

MODELS   = [d["name"] for d in RESULTS[0]["summary"]]
STABL    = [m for m in MODELS if m in RESULTS[0]["feat"]]
NR       = len(RESULTS)
TITLE    = f"{ART_LABEL} — n={N}, p={P}, k={K}, signal={SIGNAL}, B={B} | moy. sur {NR} seeds"
METRICS  = ["ns", "n_true", "n_null", "fdp", "precision", "recall", "f1", "auc"]


# ── agrégations ───────────────────────────────────────────────────────────────
def metric_arr(name, key):
    return np.array([next(d[key] for d in r["summary"] if d["name"] == name) for r in RESULTS])

AGG = {m: {k: metric_arr(m, k) for k in METRICS} for m in MODELS}   # arrays (NR,)

def mean_jaccard(name):
    sets = [r["sel_sets"][name] for r in RESULTS]
    vals = []
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            sa, sb = sets[a], sets[b]
            u = len(sa | sb)
            vals.append(1.0 if u == 0 else len(sa & sb) / u)
    return float(np.mean(vals)) if vals else 1.0

JAC = {m: mean_jaccard(m) for m in MODELS}

# scores empilés (NR, P) par modèle STABL -> moyenne & std par feature
SR_STACK   = {m: np.vstack([r["feat"][m]["sr"]    for r in RESULTS]) for m in STABL}
KO_STACK   = {m: np.vstack([r["feat"][m]["sr_ko"] for r in RESULTS]) for m in STABL}
SEL_FREQ   = {m: np.vstack([r["feat"][m]["selected"] for r in RESULTS]).mean(axis=0) for m in STABL}
SR_MEAN    = {m: SR_STACK[m].mean(axis=0) for m in STABL}
SR_STD     = {m: SR_STACK[m].std(axis=0)  for m in STABL}
KO_MEAN    = {m: KO_STACK[m].mean(axis=0) for m in STABL}
EPS_MEAN   = {m: np.vstack([r["feat"][m]["eps"] for r in RESULTS]).mean(axis=0)
              for m in STABL if RESULTS[0]["feat"][m]["eps"] is not None}   # ε_WJ,j moyen

# courbes FDP empilées (NR, len(GRID)) par modèle STABL
def curve_stack(name, key):
    return np.vstack([np.interp(GRID, r["curves"][name]["grid"], r["curves"][name][key])
                      for r in RESULTS])
FDPP = {m: curve_stack(m, "fdp_plus") for m in STABL}
FDPT = {m: curve_stack(m, "fdp_true") for m in STABL}
FRO  = {m: curve_stack(m, "fro")      for m in STABL}


# ══════════════════════════ FIGURES (moyennes) ════════════════════════════════
def stabl_color(n): return "#C41E3A" if n.startswith("V") else "#4D4F53"

# ── Fig 0 : table mean ± std ──────────────────────────────────────────────────
fig_table, ax = plt.subplots(figsize=(12, 0.9 + 0.42 * len(MODELS)))
ax.axis("off")
col = ["Modèle", "# sélec.", "# vraies", "Precision", "Recall", "F1", "FDP", "AUC", "Jaccard"]
def ms(a): return f"{a.mean():.2f}±{a.std():.2f}"
cell = [[m, ms(AGG[m]["ns"]), f"{AGG[m]['n_true'].mean():.1f}/{K}",
         ms(AGG[m]["precision"]), ms(AGG[m]["recall"]), ms(AGG[m]["f1"]),
         ms(AGG[m]["fdp"]), ms(AGG[m]["auc"]), f"{JAC[m]:.2f}"] for m in MODELS]
tbl = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
for j in range(len(col)):
    c = tbl[0, j]; c.set_facecolor("#001A7B"); c.set_text_props(color="white", weight="bold")
for i, m in enumerate(MODELS, 1):
    if m.startswith("V"):
        for j in range(len(col)):
            tbl[i, j].set_facecolor("#F2E6E9")
ax.set_title(f"Sélection par modèle (moy.±std, {NR} seeds) — {TITLE}\n({K} vraies sur {P})",
             fontsize=11, pad=16)
fig_table.tight_layout(); fig_table.savefig(os.path.join(OUT, "selection_table.png"), dpi=130)

# ── Fig 0bis : PRF mean ± std ─────────────────────────────────────────────────
fig_prf, ax = plt.subplots(figsize=(11, 4.8))
xpos = np.arange(len(MODELS)); w = 0.26
for off, key, c, lab in [(-w, "precision", "#1f77b4", "Precision"),
                         (0, "recall", "#2CA02C", "Recall"),
                         (w, "f1", "#C41E3A", "F1")]:
    ax.bar(xpos + off, [AGG[m][key].mean() for m in MODELS], w,
           yerr=[AGG[m][key].std() for m in MODELS], capsize=2, label=lab, color=c)
ax.set_xticks(xpos); ax.set_xticklabels(MODELS, rotation=25, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"Precision / Recall / F1 (moy.±std) — {TITLE}")
fig_prf.tight_layout(); fig_prf.savefig(os.path.join(OUT, "precision_recall_f1.png"), dpi=130)

# ── Fig 1 : ROC moyenne ───────────────────────────────────────────────────────
fig_roc, ax = plt.subplots(figsize=(7, 6))
fpr_grid = np.linspace(0, 1, 101)
for m in MODELS:
    tprs = []
    for r in RESULTS:
        fpr, tpr = r["roc"][m][0], r["roc"][m][1]
        t = np.interp(fpr_grid, fpr, tpr); t[0] = 0.0
        tprs.append(t)
    mt = np.mean(tprs, axis=0)
    ax.plot(fpr_grid, mt, lw=1.8,
            label=f"{m} (AUC={AGG[m]['auc'].mean():.3f}±{AGG[m]['auc'].std():.3f})")
ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC moyenne — {TITLE}")
ax.legend(loc="lower right", fontsize=7); ax.grid(alpha=0.3)
fig_roc.tight_layout(); fig_roc.savefig(os.path.join(OUT, "roc_curves.png"), dpi=130)

# ── Fig 2 : distributions de scores (moyenne par feature, ± std) — V2 et V1 ───
def make_score_fig(model_name, version, fname):
    if model_name not in SR_MEAN: return None
    sr_mean, sr_std, ko_mean = SR_MEAN[model_name], SR_STD[model_name], KO_MEAN[model_name]
    sr_all = SR_STACK[model_name]; ko_all = KO_STACK[model_name]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    null_pool = sr_all[:, K:].ravel(); ko_pool = ko_all.ravel()
    bins = np.linspace(0, max(sr_all.max(), ko_all.max()) + 0.02, 30)
    ax.hist(null_pool, bins=bins, alpha=0.5, color="#4D4F53", label=f"réelles nulles ({NR} seeds)")
    ax.hist(ko_pool,   bins=bins, alpha=0.5, color="#1f77b4", label=f"{ART_LABEL}s ({NR} seeds)")
    for i in range(K):
        ax.axvline(sr_mean[i], color="#C41E3A", lw=2, label="réelles vraies (moy.)" if i == 0 else None)
    ax.set_xlabel("Score de stabilité max_λ"); ax.set_ylabel("Nombre (poolé)")
    ax.set_title(f"Distribution des scores ({version})"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.errorbar(ko_mean[K:], sr_mean[K:], yerr=sr_std[K:], fmt="o", ms=3, alpha=0.4,
                color="#4D4F53", elinewidth=0.6, label="nulles (±std)")
    ax.errorbar(ko_mean[:K], sr_mean[:K], yerr=sr_std[:K], fmt="*", ms=12, color="#C41E3A",
                elinewidth=0.8, label="vraies (±std)", zorder=5)
    lim = max(sr_mean.max(), ko_mean.max()) + 0.05
    ax.plot([0, lim], [0, lim], ls=":", color="gray", label="score = score_art")
    ax.set_xlabel(f"Score {ART_LABEL} moyen"); ax.set_ylabel("Score réel moyen")
    ax.set_title(f"Réel vs {ART_LABEL} par feature (moy.±std)"); ax.legend(fontsize=8)
    fig.suptitle(f"Scores STABL-{version} — {TITLE}"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    return fig

fig_scores     = make_score_fig(f"V2_constr_{REF}", f"V2_constr_{REF}", "score_distributions.png")
fig_scores_v1  = make_score_fig(f"V1_{REF}", f"V1_{REF}", "score_distributions_v1.png") if f"V1_{REF}" in STABL else None
# ── Concentration score(j)−score_réf(j) dans ±ε — par run + MOYENNE (Last, V2_constr) ──
# (synthétique : ★ = vraies features, attendues HORS bande car score >> score_réf)
def conc_fig(m, fname, title, reflab):
    if m not in EPS_MEAN: return None
    def panel(ax, sc, scp, eps, ttl, im):
        if im: ax.set_facecolor("#F4F4F4")
        diff = sc - scp; inside = np.abs(diff) <= eps
        order = np.argsort(eps); xj = np.arange(len(eps)); pos = np.argsort(order)  # feature -> rang
        en = eps[order]; dn = diff[order]; ins = inside[order]
        ax.fill_between(xj, -en, en, color="#2CA02C", alpha=0.18)
        ax.plot(xj, en, color="#2CA02C", lw=0.5); ax.plot(xj, -en, color="#2CA02C", lw=0.5)
        ax.scatter(xj[ins], dn[ins], s=4, color="#2CA02C", zorder=3)
        ax.scatter(xj[~ins], dn[~ins], s=9, color="#C41E3A", zorder=4)
        ax.scatter(pos[:K], diff[:K], s=70, marker="*", color="#1f77b4", edgecolor="black", lw=0.4, zorder=6)  # vraies
        ax.axhline(0, color="gray", ls=":", lw=0.6); ax.set_ylim(-1.05, 1.05)
        nn = inside[K:]                                  # concentration sur les nulles
        ax.set_title(f"{ttl} ({100*nn.mean():.0f}% nulles dans)", fontsize=8, weight="bold" if im else "normal")
        ax.tick_params(labelsize=6)
    pan = [(r["feat"][m]["sr"], r["feat"][m]["sr_ko"], r["feat"][m]["eps"], f"run {i+1}", False)
           for i, r in enumerate(RESULTS)]
    pan.append((SR_MEAN[m], KO_MEAN[m], EPS_MEAN[m], "MOYENNE", True))
    nc = 3; nr = (len(pan) + nc - 1) // nc
    fig, axe = plt.subplots(nr, nc, figsize=(4.5 * nc, 3.0 * nr), squeeze=False); axe = axe.ravel()
    for a in axe[len(pan):]:
        a.axis("off")
    for a, (sc, scp, eps, ttl, im) in zip(axe, pan):
        panel(a, sc, scp, eps, ttl, im)
    proxy = [Line2D([], [], marker='o', ls='', color="#2CA02C", label="nulle dans bande ±ε"),
             Line2D([], [], marker='o', ls='', color="#C41E3A", label="hors bande"),
             Line2D([], [], marker='*', ls='', color="#1f77b4", label="vraie feature")]
    fig.legend(handles=proxy, loc="upper center", ncol=3, fontsize=8)
    fig.supxlabel(f"features (triées par ε)  —  réf. nulle : {reflab}", fontsize=9)
    fig.supylabel("score(j) − score_réf(j)", fontsize=9)
    fig.suptitle(title, fontsize=10); fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, fname), dpi=130); return fig
fig_perm = conc_fig(f"Last_{REF}", "last_perm_validation.png",
                    f"Concentration Last : score−score_perm dans ±(ε_BFW+ε_perm) par run — {TITLE}", "score_perm")
fig_vcc = conc_fig(f"V2_constr_{REF}", "v2constr_concentration.png",
                   f"Concentration V2_constr : score−score_ko dans ±ε_WJ par run — {TITLE}", "score_ko")

# ── Fig 3 : vrai FDP par modèle (moy.±std) ────────────────────────────────────
fig_bar, ax = plt.subplots(figsize=(9, 4.5))
fdpm = [AGG[m]["fdp"].mean() for m in MODELS]; fdps = [AGG[m]["fdp"].std() for m in MODELS]
ax.bar(range(len(MODELS)), fdpm, yerr=fdps, capsize=3,
       color=[stabl_color(m) for m in MODELS], alpha=0.85)
for i, (mu, sd) in enumerate(zip(fdpm, fdps)):
    ax.text(i, mu + sd + 0.01, f"{mu:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=25, ha="right")
ax.set_ylabel("Vrai FDP (moy.±std)"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"Vrai FDP par modèle — {TITLE}")
fig_bar.tight_layout(); fig_bar.savefig(os.path.join(OUT, "fdp_par_modele.png"), dpi=130)

# ── Fig 4 : FDP — 1 colonne par run + colonne MOYENNE, 1 ligne par modèle STABL ─
ncol = NR + 1
fig_curves, axes = plt.subplots(len(STABL), ncol, figsize=(2.5 * ncol, 2.4 * len(STABL)),
                                squeeze=False)
for ri, name in enumerate(STABL):
    obj = FDPP[name] + FRO[name]; ft = FDPT[name]
    tstars = [r["curves"][name]["tstar"] for r in RESULTS]
    nsl = [len(r["sel_sets"][name]) for r in RESULTS]
    cols = [(obj[i], ft[i], tstars[i], nsl[i], f"run {i+1}", False) for i in range(NR)]
    cols.append((obj.mean(0), ft.mean(0), float(np.mean(tstars)), float(np.mean(nsl)), "MOYENNE", True))
    for ci, (oc, fc, ts, ns, ttl, is_mean) in enumerate(cols):
        ax = axes[ri][ci]
        if is_mean: ax.set_facecolor("#F4F4F4")
        ax.plot(GRID, oc, color="#1f77b4", lw=1.7 if is_mean else 1.2)
        ax.plot(GRID, fc, color="#C41E3A", lw=1.5 if is_mean else 1.0, ls="--")
        ax.axvline(ts, color="black", ls=":", lw=1.1)
        ax.set_ylim(0, 1.05); ax.set_xticks([0, .5, 1]); ax.tick_params(labelsize=6)
        ax.text(0.04, 0.92, f"n={ns:.0f}", transform=ax.transAxes, fontsize=6, ha="left", va="top",
                color="#2CA02C", weight="bold")
        ax.text(0.96, 0.92, f"t*={ts:.2f}", transform=ax.transAxes, fontsize=6, ha="right", va="top")
        if ri == 0: ax.set_title(ttl, fontsize=8, weight="bold" if is_mean else "normal")
        if ci == 0: ax.set_ylabel(name, fontsize=8)
proxy = [Line2D([], [], color="#1f77b4", lw=2, label="objectif FDP+ + frontière"),
         Line2D([], [], color="#C41E3A", lw=2, ls="--", label="vrai FDP"),
         Line2D([], [], color="black", ls=":", lw=1.2, label="t* du run")]
fig_curves.legend(handles=proxy, loc="upper center", ncol=3, fontsize=8)
fig_curves.supxlabel("seuil t", fontsize=9)
fig_curves.suptitle(f"FDP par run (objectif vs vrai FDP) + moyenne — {TITLE}", y=0.998, fontsize=10)
fig_curves.tight_layout(rect=[0, 0, 1, 0.95])
fig_curves.savefig(os.path.join(OUT, "fdp_curves.png"), dpi=130)

# ── Fig 5 : objectif + vrai FDP par run — V1, V2_constr, Last (fonction réutilisable) ─
def obj_fig(_M, fname, title):
    if _M not in FDPP: return None
    obj = FDPP[_M] + FRO[_M]; ft = FDPT[_M]; fp = FDPP[_M]
    tstars = [r["curves"][_M]["tstar"] for r in RESULTS]
    nsl = [len(r["sel_sets"][_M]) for r in RESULTS]
    panels = [(fp[i], obj[i], ft[i], tstars[i], nsl[i], f"run {i+1}", False) for i in range(NR)]
    panels.append((fp.mean(0), obj.mean(0), ft.mean(0), float(np.mean(tstars)), float(np.mean(nsl)), "MOYENNE", True))
    ncolp = 3; nrowp = (len(panels) + ncolp - 1) // ncolp
    fig, axes = plt.subplots(nrowp, ncolp, figsize=(4.3 * ncolp, 3.3 * nrowp), squeeze=False); axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    for ax, (fpc, oc, fc, ts, ns, ttl, is_mean) in zip(axes, panels):
        if is_mean: ax.set_facecolor("#F4F4F4")
        ax.plot(GRID, fpc, color="#999999", lw=1.0, label="terme FDP+ / (|S_perm|+1)/D")
        ax.fill_between(GRID, fpc, oc, step="post", alpha=0.30, color="#2CA02C", label="frontière")
        ax.plot(GRID, oc, color="#1f77b4", lw=2.0, label="objectif")
        ax.plot(GRID, fc, color="#C41E3A", lw=1.8, ls="--", label="vrai FDP")
        ax.axvline(ts, color="black", ls=":", lw=1.3, label=f"t*={ts:.2f}")
        ax.set_ylim(0, 1.05); ax.set_xlabel("t", fontsize=8); ax.set_ylabel("FDP", fontsize=8)
        ax.set_title(f"{ttl}  (n={ns:.0f} sél.)", fontsize=9, weight="bold" if is_mean else "normal")
        ax.legend(fontsize=6, loc="upper right"); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10); fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, fname), dpi=140); return fig
fig_obj_v1 = obj_fig(f"V1_{REF}", "v1_objective.png", f"V1_{REF} — objectif FDP+ + vrai FDP par run — {TITLE}")
fig_obj_vc = obj_fig(f"V2_constr_{REF}", "v2constr_objective.png",
                     f"V2_constr_{REF} — objectif FDP+ + frontière + vrai FDP par run — {TITLE}")
fig_obj = obj_fig(f"Last_{REF}", "last_objective.png",
                  f"Last_{REF} — objectif (|S_perm|+1)/D + frontière + vrai FDP par run — {TITLE}")
fig_obj_vca0 = obj_fig(f"V2_constr_a0_{REF}", "v2constr_a0_objective.png",
                       f"V2_constr_a0_{REF} — objectif |S_ko|/D + frontière (SANS +1) + vrai FDP — {TITLE}")
fig_obj_la0 = obj_fig(f"Last_a0_{REF}", "last_a0_objective.png",
                      f"Last_a0_{REF} — objectif |S_perm|/D + frontière (SANS +1) + vrai FDP — {TITLE}")

# Fig variance/std des scores : retirée (peu informatif)
fig_var = None

# ── Fig 7 : JACCARD inter-run des sélections (par modèle) ──────────────────────
fig_jac, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(range(len(MODELS)), [JAC[m] for m in MODELS],
       color=[stabl_color(m) for m in MODELS], alpha=0.85)
for i, m in enumerate(MODELS):
    ax.text(i, JAC[m] + 0.01, f"{JAC[m]:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=25, ha="right")
ax.set_ylabel("Jaccard moyen inter-run"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"Stabilité des sélections : Jaccard moyen ({NR} seeds, "
             f"{NR*(NR-1)//2} paires) — {TITLE}")
fig_jac.tight_layout(); fig_jac.savefig(os.path.join(OUT, "jaccard.png"), dpi=130)

# ── Fig 8 : barrière ∂⁺(t*) (score réel vs réf nulle) — V1, V2_constr, Last (fonction) ──
def barrier_fig(_Mb, fname, title):
    if _Mb not in SR_STACK: return None
    SRr, KOr = SR_STACK[_Mb], KO_STACK[_Mb]                        # (NR, P)
    epss = [(r["feat"][_Mb]["eps"] if r["feat"][_Mb].get("eps") is not None else np.zeros(P)) for r in RESULTS]
    tstars = [r["curves"][_Mb]["tstar"] for r in RESULTS]
    epsm = EPS_MEAN[_Mb] if _Mb in EPS_MEAN else np.zeros(P)
    xlab = "score_perm" if _Mb.startswith("Last") else "score knockoff"
    lim = max(SRr.max(), KOr.max(), SR_MEAN[_Mb].max(), KO_MEAN[_Mb].max()) + 0.04
    panels = [(SRr[i], KOr[i], epss[i], tstars[i], f"run {i+1}", False) for i in range(NR)]
    panels.append((SR_MEAN[_Mb], KO_MEAN[_Mb], epsm, float(np.mean(tstars)), "MOYENNE", True))
    ncolp = 3; nrowp = (len(panels) + ncolp - 1) // ncolp
    fig, axes = plt.subplots(nrowp, ncolp, figsize=(4.2 * ncolp, 4.0 * nrowp), squeeze=False); axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    ia = np.arange(P)
    for ax, (sr, srko, eps, ts, ttl, is_mean) in zip(axes, panels):
        if is_mean: ax.set_facecolor("#F4F4F4")
        selected = sr > ts; boundary = selected & (sr <= ts + eps); safe = selected & ~boundary
        for j in np.where(selected)[0]:
            c = "#ff7f0e" if boundary[j] else "#2CA02C"
            ax.plot([srko[j], srko[j]], [sr[j] - eps[j], sr[j]], color=c, alpha=0.5, lw=1.0, zorder=2)
        for mask, color, s in [(~selected, "#888888", 9), (safe, "#2CA02C", 18),
                               (boundary, "#ff7f0e", 30)]:
            i = np.where(mask & (ia >= K))[0]
            if len(i): ax.scatter(srko[i], sr[i], s=s, color=color, edgecolor="black",
                                 lw=0.2, alpha=0.85, zorder=3)
        for j in range(K):
            c = "#ff7f0e" if boundary[j] else ("#2CA02C" if selected[j] else "#888888")
            ax.scatter(srko[j], sr[j], s=95, marker="*", color=c, edgecolor="black", lw=0.5, zorder=5)
        ax.axhline(ts, color="#1f77b4", ls="--", lw=1.3)
        ax.plot([0, lim], [0, lim], ls=":", color="gray", lw=0.7)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.tick_params(labelsize=6)
        ax.set_title(f"{ttl}  t*={ts:.2f}, n={int(selected.sum())} sél., |∂⁺|={int(boundary.sum())}",
                     fontsize=8, weight="bold" if is_mean else "normal")
    proxy = [Line2D([], [], marker='o', ls='', color="#888888", label="non sél."),
             Line2D([], [], marker='o', ls='', color="#2CA02C", label="sél. hors ∂⁺"),
             Line2D([], [], marker='o', ls='', color="#ff7f0e", label="∂⁺(t*)"),
             Line2D([], [], marker='*', ls='', color="#C41E3A", label="vraies (★)"),
             Line2D([], [], color="#1f77b4", ls='--', label="t*")]
    fig.legend(handles=proxy, loc="upper center", ncol=5, fontsize=8)
    fig.supxlabel(xlab, fontsize=9); fig.supylabel("score réel", fontsize=9)
    fig.suptitle(title, fontsize=9, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(OUT, fname), dpi=140); return fig
fig_bd_v1 = barrier_fig(f"V1_{REF}", "barriere_v1.png", f"Barrière V1_{REF} (score vs knockoff) par run — {TITLE}")
fig_bd_vc = barrier_fig(f"V2_constr_{REF}", "barriere_v2constr.png", f"Barrière V2_constr_{REF} (score vs knockoff) par run — {TITLE}")
fig_bd = barrier_fig(f"Last_{REF}", "barriere_dplus.png", f"Barrière Last_{REF} (score vs score_perm) par run — {TITLE}")
fig_bd_vca0 = barrier_fig(f"V2_constr_a0_{REF}", "barriere_v2constr_a0.png", f"Barrière V2_constr_a0 (sans +1) — {TITLE}")
fig_bd_la0 = barrier_fig(f"Last_a0_{REF}", "barriere_last_a0.png", f"Barrière Last_a0 (sans +1) — {TITLE}")

# ── Plots parité STABL officiel : stability path + FDR + PRC + boxplot + features ──
_stcol = lambda n: "#C41E3A" if n.startswith("V") else ("#2CA02C" if n.startswith("L") else "#4D4F53")
# stability path + objectif COMPLET (FDP+ + frontière via OBJ_) -> figures combinées au PDF
_native_figs = []
for _nm, _mdl in RESULTS[0]["models"].items():
    if not _nm.endswith(REF):
        continue
    try:
        _res = plot_stabl_path(_mdl, show_fig=False, export_file=False)
        _fp = _res[0] if isinstance(_res, tuple) else _res
        _fp.suptitle(f"Stability path — {_nm}", fontsize=9); _native_figs.append(_fp)
    except Exception as e:
        print(f"  [path {_nm}] non dispo: {type(e).__name__}")
    try:
        _m2 = copy.deepcopy(_mdl); _obj = np.asarray(getattr(_mdl, "OBJ_", _mdl.FDRs_))
        _m2.FDRs_ = _obj; _m2.min_fdr_ = float(_obj.min())
        _ff, _fax = plot_fdr_graph(_m2, show_fig=False, export_file=False)
        _fax.set_title(f"Objectif minimisé (FDP+ + frontière) — {_nm}", fontsize=9); _native_figs.append(_ff)
    except Exception as e:
        print(f"  [fdr {_nm}] non dispo: {type(e).__name__}")

_y0 = RESULTS[0]["yte"]; _PRED = {m: RESULTS[0]["roc"][m][3] for m in MODELS}
from sklearn.metrics import precision_recall_curve, average_precision_score
fig_prc, axp = plt.subplots(figsize=(7, 6))
for m in MODELS:
    pr, rc, _ = precision_recall_curve(_y0, _PRED[m]); ap = average_precision_score(_y0, _PRED[m])
    axp.plot(rc, pr, lw=1.5, color=_stcol(m), label=f"{m} (AP={ap:.3f})")
axp.axhline(_y0.mean(), ls=":", color="gray"); axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
axp.set_ylim(0, 1.02); axp.set_title(f"Precision-Recall (test held-out, seed 0) — {TITLE}")
axp.legend(fontsize=7); axp.grid(alpha=.3); fig_prc.tight_layout()
fig_prc.savefig(os.path.join(OUT, "prc_curves.png"), dpi=130)

fig_pb, axb = plt.subplots(figsize=(1.5 * len(MODELS) + 2, 5)); _data = []
for m in MODELS:
    _data += [_PRED[m][_y0 == 0], _PRED[m][_y0 == 1]]
_bp = axb.boxplot(_data, positions=np.arange(len(_data)), widths=0.6, patch_artist=True)
for i, _pt in enumerate(_bp["boxes"]):
    _pt.set_facecolor("#4D4F53" if i % 2 == 0 else "#C41E3A"); _pt.set_alpha(.55)
axb.set_xticks([2 * i + 0.5 for i in range(len(MODELS))]); axb.set_xticklabels(MODELS, rotation=25, ha="right")
axb.set_ylabel("Prédiction (proba)"); axb.set_title(f"Prédictions par classe (gris=0/rouge=1) — {TITLE}")
fig_pb.tight_layout(); fig_pb.savefig(os.path.join(OUT, "prediction_boxplot.png"), dpi=130)

# features vs classe : top sélectionnés par Last (réf) ; ★ = VRAIE feature (j<K)
_lastref = f"Last_{REF}"
if _lastref in RESULTS[0]["feat"]:
    _srm = np.mean([RESULTS[r]["feat"][_lastref]["sr"] for r in range(NR)], axis=0)
    _top = np.argsort(_srm)[::-1][:10]; _Xte0 = RESULTS[0]["Xte"]
    fig_ft, axf = plt.subplots(2, 5, figsize=(16, 6), squeeze=False); axf = axf.ravel()
    for i, j in enumerate(_top):
        v = _Xte0[:, j]; axf[i].boxplot([v[_y0 == 0], v[_y0 == 1]], labels=["0", "1"])
        axf[i].set_title(f"f{j} {'★' if j < K else ''}", fontsize=9,
                         color=("blue" if j < K else "black")); axf[i].grid(alpha=.3)
    for i in range(len(_top), 10):
        axf[i].axis("off")
    fig_ft.suptitle(f"Top biomarqueurs (Last) vs classe — ★=vraie feature — {TITLE}"); fig_ft.tight_layout()
    fig_ft.savefig(os.path.join(OUT, "features_vs_outcome.png"), dpi=130)
else:
    fig_ft = None

# ── PDF combiné ───────────────────────────────────────────────────────────────
pages = [fig_table, fig_prf, fig_roc, fig_prc, fig_pb, fig_ft, fig_scores, fig_scores_v1, fig_bar, fig_curves,
         fig_obj_v1, fig_bd_v1, fig_obj_vc, fig_bd_vc, fig_obj_vca0, fig_bd_vca0, fig_vcc,
         fig_obj, fig_bd, fig_obj_la0, fig_bd_la0, fig_perm,
         fig_var, fig_jac]
pages = [p for p in pages if p is not None] + _native_figs   # path + objectif combinés
with PdfPages(os.path.join(OUT, "analyse_complete.pdf")) as pdf:
    for f in pages:
        pdf.savefig(f)
for f in pages:
    plt.close(f)


# ══════════════════════════ CSV agrégés ═══════════════════════════════════════
_ts    = datetime.now().isoformat(timespec="seconds")
_seeds = ";".join(str(s) for s in SEEDS)

# runs_metrics.csv : 1 ligne / (appel × modèle), mean+std agrégés sur les seeds
MCOLS = ["timestamp", "artificial", "base", "n", "p", "k", "signal", "n_bootstraps", "n_seeds", "seeds",
         "model", "n_sel_mean", "n_sel_std", "n_true_mean", "precision_mean", "precision_std",
         "recall_mean", "recall_std", "f1_mean", "f1_std", "fdp_mean", "fdp_std",
         "auc_mean", "auc_std", "jaccard_mean"]
_mh = not os.path.exists(CSV_PATH)
with open(CSV_PATH, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=MCOLS)
    if _mh: w.writeheader()
    for m in MODELS:
        a = AGG[m]
        w.writerow({
            "timestamp": _ts, "artificial": ART_LABEL, "base": BASE, "n": N, "p": P, "k": K,
            "signal": SIGNAL, "n_bootstraps": B, "n_seeds": NR, "seeds": _seeds, "model": m,
            "n_sel_mean": round(a["ns"].mean(), 3), "n_sel_std": round(a["ns"].std(), 3),
            "n_true_mean": round(a["n_true"].mean(), 3),
            "precision_mean": round(a["precision"].mean(), 4), "precision_std": round(a["precision"].std(), 4),
            "recall_mean": round(a["recall"].mean(), 4), "recall_std": round(a["recall"].std(), 4),
            "f1_mean": round(a["f1"].mean(), 4), "f1_std": round(a["f1"].std(), 4),
            "fdp_mean": round(a["fdp"].mean(), 4), "fdp_std": round(a["fdp"].std(), 4),
            "auc_mean": round(a["auc"].mean(), 4), "auc_std": round(a["auc"].std(), 4),
            "jaccard_mean": round(JAC[m], 4),
        })
print(f"Métriques agrégées ({len(MODELS)} lignes) -> {CSV_PATH}")

# runs_feature_scores.csv : 1 ligne / (appel × modèle STABL × feature)
FCOLS = ["timestamp", "artificial", "base", "n", "p", "k", "signal", "n_bootstraps", "n_seeds",
         "model", "feature", "is_true", "score_mean", "score_std", "score_ko_mean", "sel_freq"]
_fh = not os.path.exists(FEAT_CSV_PATH); _nf = 0
with open(FEAT_CSV_PATH, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FCOLS)
    if _fh: w.writeheader()
    for m in STABL:
        for j in range(P):
            w.writerow({
                "timestamp": _ts, "artificial": ART_LABEL, "base": BASE, "n": N, "p": P, "k": K,
                "signal": SIGNAL, "n_bootstraps": B, "n_seeds": NR, "model": m,
                "feature": j, "is_true": int(j < K),
                "score_mean": round(float(SR_MEAN[m][j]), 6),
                "score_std": round(float(SR_STD[m][j]), 6),
                "score_ko_mean": round(float(KO_MEAN[m][j]), 6),
                "sel_freq": round(float(SEL_FREQ[m][j]), 4),
            })
            _nf += 1
print(f"Scores par-feature agrégés ({_nf} lignes) -> {FEAT_CSV_PATH}")

# ── Base de données LOCALE (dans le dossier de la config) ─────────────────────
# Format LONG, 1 ligne / (run × modèle × feature) : TOUTES les infos pour faire
# n'importe quel plot. Détail PAR RUN (pas seulement agrégé) :
#   - par feature : score, score_ko, selected, is_true (score vide pour les baselines) ;
#   - métriques du run (répétées par feature) : n_sel, precision, recall, f1, fdp, auc ;
#   - moyennes sur les runs (répétées) : *_mean + jaccard_mean.
DB_PATH = os.path.join(OUT, "database.csv")
DB_COLS = ["seed", "run", "artificial", "base", "n", "p", "k", "signal", "n_bootstraps",
           "model", "feature", "is_true", "score", "score_ko", "selected",
           "n_sel", "precision", "recall", "f1", "fdp", "auc",
           "auc_mean", "fdp_mean", "f1_mean", "precision_mean", "recall_mean", "jaccard_mean"]
_db = 0
with open(DB_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=DB_COLS); w.writeheader()
    for ri, res in enumerate(RESULTS):
        for name in MODELS:
            rm = next(d for d in res["summary"] if d["name"] == name)
            am = AGG[name]; fe = res["feat"].get(name); ss = res["sel_sets"][name]
            brow = {
                "seed": SEEDS[ri], "run": ri + 1, "artificial": ART_LABEL, "base": BASE,
                "n": N, "p": P, "k": K, "signal": SIGNAL, "n_bootstraps": B, "model": name,
                "n_sel": rm["ns"], "precision": round(rm["precision"], 4),
                "recall": round(rm["recall"], 4), "f1": round(rm["f1"], 4),
                "fdp": round(rm["fdp"], 4), "auc": round(rm["auc"], 4),
                "auc_mean": round(am["auc"].mean(), 4), "fdp_mean": round(am["fdp"].mean(), 4),
                "f1_mean": round(am["f1"].mean(), 4), "precision_mean": round(am["precision"].mean(), 4),
                "recall_mean": round(am["recall"].mean(), 4), "jaccard_mean": round(JAC[name], 4),
            }
            for j in range(P):
                row = dict(brow, feature=j, is_true=int(j < K), selected=int(j in ss))
                if fe is not None:
                    row["score"]    = round(float(fe["sr"][j]), 6)
                    row["score_ko"] = round(float(fe["sr_ko"][j]), 6)
                else:
                    row["score"] = ""; row["score_ko"] = ""       # baselines : pas de score
                w.writerow(row); _db += 1
print(f"Base de données locale ({_db} lignes) -> {DB_PATH}")
print(f"\nPDF + figures dans : {OUT}")
