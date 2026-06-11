"""
single_dataset_analysis.py
Analyse AGRÉGÉE sur N_SEEDS datasets synthétiques (5 par défaut), pour STABL
(V1, V2_constr, V2_constr_a0, V2_alpha, V2_vidé) + baselines (ALasso, Lasso,
ElasticNet, XGBoost).

À chaque appel : on lance N_SEEDS runs (seeds différentes -> datasets différents) et
analyse_complete.pdf contient les DONNÉES MOYENNES et COURBES MOYENNES sur ces runs,
plus la VARIANCE des scores par feature et le JACCARD inter-run des sélections.

Variantes alpha (offset du numérateur de FDP+ : num=(1/r)|S_ko|+alpha) :
  V2_constr      alpha=1   (offset Barber-Candès standard)
  V2_constr_a0   alpha=0   (pas d'offset)
  V2_alpha       offset = barrière·D(t) -> contrib FDP+ = exp((t−t_drop)/σ) ; barrière
                 ACTIVE ssi aucun score réel de a0 ne vaut exactement 1.
  V2_vidé        = V2_constr + frontière vidée à la main : Ŝ = {score > t*+εWJ,j}.

Paramétrable :
  --artificial knockoff|perm  --n --p --k --signal --B --seed --n-seeds

Figures (results_synthetic/<tag>_n<N>_p<P>_k<K>_sig<SIGNAL>_B<B>/) + analyse_complete.pdf :
  selection_table, precision_recall_f1, roc_curves, score_distributions(V2/V1),
  fdp_par_modele, fdp_curves, v2constr_objective, score_variance (NEW), jaccard (NEW).
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

from stabl.stabl import Stabl as StablV1
from stabl.stablV2 import Stabl as StablV2
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
args = ap.parse_args()

N, P, K, SIGNAL, B, SEED = args.n, args.p, args.k, args.signal, args.B, args.seed
NSEEDS = args.n_seeds
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
                        and int(row["n_bootstraps"]) == B):
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

FOLDER = f"{TAG}_n{N}_p{P}_k{K}_sig{SIGNAL:g}_B{B}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_synthetic", FOLDER)
os.makedirs(OUT, exist_ok=True)
print(f"Dataset: artificial={ART}, n={N}, p={P}, k={K}, signal={SIGNAL}, B={B}, n_seeds={NSEEDS}")
print(f"Vraies features = indices 0..{K-1}\nDossier de sortie : {OUT}\n")

GRID = np.arange(0., 1., .01)   # grille de seuils commune (fdr_threshold_range)


# ── helpers (utilisent K global) ──────────────────────────────────────────────
def get_support(model, name=None):
    if name == "V2_vidé":
        sr     = model.stabl_scores_.max(axis=1)
        eps_wj = model.eps_B_total_fw_
        return np.where(sr > model.fdr_min_threshold_ + eps_wj)[0]
    if hasattr(model, "get_support"):
        return np.where(model.get_support())[0]
    est = model.best_estimator_
    if hasattr(est, "coef_"):
        return np.where(np.abs(est.coef_[0]) > 1e-8)[0]
    return np.where(est.feature_importances_ > 0)[0]

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
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N + 2000, P))
    beta = np.zeros(P); beta[:K] = SIGNAL / np.sqrt(K)
    yl = X[:, :K] @ beta[:K] + rng.standard_normal(N + 2000)
    y = (yl > 0).astype(int)
    Xtr, Xte, ytr, yte = X[:N], X[N:], y[:N], y[N:]
    cov = np.eye(P) if ART == "knockoff" else None

    def _lr():
        return LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                  max_iter=int(1e6), random_state=seed)

    def stabl_v1():
        return StablV1(base_estimator=_lr(), lambda_grid="auto", n_lambda=10, n_bootstraps=B,
                       artificial_type=ART, artificial_proportion=1.0, sample_fraction=0.5,
                       replace=False, n_jobs=-1, random_state=seed, cov_matrix=cov)

    def stabl_v2(mode, alpha=1.0):
        kw = dict(base_estimator=_lr(), lambda_grid="auto", n_lambda=10, n_bootstraps=B,
                  artificial_type=ART, artificial_proportion=1.0, sample_fraction=0.5,
                  replace=False, n_jobs=-1, random_state=seed, selection_mode=mode,
                  delta=0.05, alpha=alpha)
        if ART == "knockoff":
            kw.update(knockoff_method="equicorrelated", cov_matrix=cov)
        return StablV2(**kw)

    def baseline(est, grid):
        n_splits = max(2, min(5, int(np.min(np.bincount(ytr)))))
        return GridSearchCV(est, grid, cv=n_splits, scoring="roc_auc", n_jobs=-1)

    _CG = {"C": np.logspace(-2, 0, 10)}
    fitted = {}
    # STABL bootstrap complet : V1 + V2_constr
    v1 = stabl_v1(); v1.fit(Xtr, ytr); fitted["V1"] = v1
    vc = stabl_v2("constrained"); vc.fit(Xtr, ytr); fitted["V2_constr"] = vc
    # dérivés (aucun re-bootstrap)
    ma0 = copy.deepcopy(vc); ma0.alpha = 0.0; ma0._compute_FDPplus(); fitted["V2_constr_a0"] = ma0
    fitted["V2_vidé"] = copy.deepcopy(vc)
    # V2_alpha : barrière exp, t_drop = borne inf du dernier plateau de zéros de (FDP+ + frontière) a0
    sigma   = 2.0 * float(np.diff(np.asarray(vc.fdr_threshold_range)).mean())
    grid_a0 = np.asarray(ma0.fdr_threshold_range)
    sr_a0   = ma0.stabl_scores_.max(axis=1); eps_a0 = ma0.eps_B_total_fw_
    D_a0    = np.array([max(1, int((sr_a0 > t).sum())) for t in grid_a0])
    fro_a0  = np.array([((sr_a0 > t) & (sr_a0 <= t + eps_a0)).sum() for t in grid_a0]) / D_a0
    obj_a0  = np.asarray(ma0.FDRs_) + fro_a0
    # Override seuil a0 : un seul plateau de zéros qui termine à 1 -> 2e min local de l'objectif.
    _alt = second_local_min_threshold(obj_a0, grid_a0)
    if _alt is not None:
        ma0.fdr_min_threshold_ = _alt
    zmask   = obj_a0 <= 1e-12; zidx = np.where(zmask)[0]
    if len(zidx):
        last = int(zidx[-1]); start = last
        while start - 1 >= 0 and zmask[start - 1]:
            start -= 1
        t_drop = float(grid_a0[start])
    else:
        t_drop = float(ma0.fdr_min_threshold_)
    use_bar = not bool((ma0.stabl_scores_.max(axis=1) == 1.0).any())

    def alpha_combined(t, D):
        return (float(np.exp((t - t_drop) / sigma)) if use_bar else 0.0) * D

    malpha = copy.deepcopy(vc); malpha.alpha = alpha_combined; malpha._compute_FDPplus()
    fitted["V2_alpha"] = malpha
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
        if ns > 0:
            clf = LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced",
                                     max_iter=10000)
            clf.fit(Xtr[:, sel], ytr)
            prob = clf.predict_proba(Xte[:, sel])[:, 1]
            auc = roc_auc_score(yte, prob); fpr, tpr, _ = roc_curve(yte, prob)
        else:
            auc = 0.5; fpr, tpr = np.array([0., 1.]), np.array([0., 1.])
        summary.append(dict(name=name, ns=ns, n_true=n_true, n_null=n_null, fdp=fdp,
                            precision=precision, recall=recall, f1=f1, auc=auc))
        roc[name] = (np.asarray(fpr), np.asarray(tpr), auc)
        sel_sets[name] = set(int(j) for j in sel)
        scores = getattr(model, "stabl_scores_", None)
        if scores is not None:
            sr = scores.max(axis=1); sr_ko = model.stabl_scores_artificial_.max(axis=1)
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
    return dict(summary=summary, roc=roc, feat=feat, curves=curves, sel_sets=sel_sets,
                t_drop=t_drop, sigma=sigma, use_bar=use_bar)


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
        fpr, tpr, _ = r["roc"][m]
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

fig_scores    = make_score_fig("V2_constr", "V2", "score_distributions.png")
fig_scores_v1 = make_score_fig("V1", "V1", "score_distributions_v1.png") if "V1" in STABL else None

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
    cols = [(obj[i], ft[i], tstars[i], f"run {i+1}", False) for i in range(NR)]
    cols.append((obj.mean(0), ft.mean(0), float(np.mean(tstars)), "MOYENNE", True))
    for ci, (oc, fc, ts, ttl, is_mean) in enumerate(cols):
        ax = axes[ri][ci]
        if is_mean: ax.set_facecolor("#F4F4F4")
        ax.plot(GRID, oc, color="#1f77b4", lw=1.7 if is_mean else 1.2)
        ax.plot(GRID, fc, color="#C41E3A", lw=1.5 if is_mean else 1.0, ls="--")
        ax.axvline(ts, color="black", ls=":", lw=1.1)
        ax.set_ylim(0, 1.05); ax.set_xticks([0, .5, 1]); ax.tick_params(labelsize=6)
        ax.text(0.96, 0.92, f"t*={ts:.2f}", transform=ax.transAxes, fontsize=6,
                ha="right", va="top")
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

# ── Fig 5 : V2_constr_a0 objectif (FDP+ SANS +1) — 1 panneau par run + MOYENNE ─
_M = "V2_constr_a0"
obj = FDPP[_M] + FRO[_M]; ft = FDPT[_M]; fp = FDPP[_M]
tstars = [r["curves"][_M]["tstar"] for r in RESULTS]
panels = [(fp[i], obj[i], ft[i], tstars[i], f"run {i+1}", False) for i in range(NR)]
panels.append((fp.mean(0), obj.mean(0), ft.mean(0), float(np.mean(tstars)), "MOYENNE", True))
ncolp = 3; nrowp = (len(panels) + ncolp - 1) // ncolp
fig_obj, axes = plt.subplots(nrowp, ncolp, figsize=(4.3 * ncolp, 3.3 * nrowp), squeeze=False)
axes = axes.ravel()
for ax in axes[len(panels):]:
    ax.axis("off")
for ax, (fpc, oc, fc, ts, ttl, is_mean) in zip(axes, panels):
    if is_mean: ax.set_facecolor("#F4F4F4")
    ax.plot(GRID, fpc, color="#999999", lw=1.0, label="FDP+ seul (sans +1)")
    ax.fill_between(GRID, fpc, oc, step="post", alpha=0.30, color="#2CA02C", label="frontière |∂⁺|/D")
    ax.plot(GRID, oc, color="#1f77b4", lw=2.0, label="objectif")
    ax.plot(GRID, fc, color="#C41E3A", lw=1.8, ls="--", label="vrai FDP")
    ax.axvline(ts, color="black", ls=":", lw=1.3, label=f"t*={ts:.2f}")
    ax.set_ylim(0, 1.05); ax.set_xlabel("t", fontsize=8); ax.set_ylabel("FDP", fontsize=8)
    ax.set_title(ttl, fontsize=9, weight="bold" if is_mean else "normal")
    ax.legend(fontsize=6, loc="upper right"); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
fig_obj.suptitle(f"V2_constr_a0 — objectif FDP+ (sans +1) + frontière par run + moyenne — {TITLE}",
                 fontsize=10)
fig_obj.tight_layout(rect=[0, 0, 1, 0.97])
fig_obj.savefig(os.path.join(OUT, "v2constr_a0_objective.png"), dpi=140)

# ── Fig 6 : VARIANCE des scores par feature (inter-run) — V1 vs V2_constr ──────
fig_var, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ax = axes[0]
idx = np.arange(P)
for name, c in [("V1", "#1f77b4"), ("V2_constr", "#C41E3A")]:
    if name in STABL:
        ax.scatter(idx[K:], SR_STD[name][K:], s=10, alpha=0.4, color=c, label=f"{name} nulles")
        ax.scatter(idx[:K], SR_STD[name][:K], s=70, marker="*", edgecolor="black",
                   lw=0.5, color=c, label=f"{name} vraies", zorder=5)
ax.set_xlabel("feature"); ax.set_ylabel("std du score inter-run")
ax.set_title(f"Variance (std) des scores par feature ({NR} seeds)"); ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
grp_lab, vals_t, vals_n = [], [], []
for name in [m for m in STABL]:
    grp_lab.append(name)
    vals_t.append(SR_STD[name][:K].mean())
    vals_n.append(SR_STD[name][K:].mean())
xp = np.arange(len(grp_lab)); w = 0.38
ax.bar(xp - w/2, vals_t, w, color="#C41E3A", label="vraies (moy. std)")
ax.bar(xp + w/2, vals_n, w, color="#4D4F53", label="nulles (moy. std)")
ax.set_xticks(xp); ax.set_xticklabels(grp_lab, rotation=20, ha="right")
ax.set_ylabel("std moyenne du score"); ax.set_title("Variance moyenne des scores par modèle")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
fig_var.suptitle(f"Variance inter-run des scores — {TITLE}")
fig_var.tight_layout(); fig_var.savefig(os.path.join(OUT, "score_variance.png"), dpi=130)

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

# ── Fig 8 : barrière ∂⁺(t*) de V2_constr_a0 (score réel vs knockoff) — par run + MOYENNE ──
fig_bd = None
if "V2_constr_a0" in EPS_MEAN:
    _Mb = "V2_constr_a0"
    SRr, KOr = SR_STACK[_Mb], KO_STACK[_Mb]                        # (NR, P)
    epss   = [r["feat"][_Mb]["eps"] for r in RESULTS]
    tstars = [r["curves"][_Mb]["tstar"] for r in RESULTS]
    lim = max(SRr.max(), KOr.max(), SR_MEAN[_Mb].max(), KO_MEAN[_Mb].max()) + 0.04
    panels = [(SRr[i], KOr[i], epss[i], tstars[i], f"run {i+1}", False) for i in range(NR)]
    panels.append((SR_MEAN[_Mb], KO_MEAN[_Mb], EPS_MEAN[_Mb],
                   float(np.mean(tstars)), "MOYENNE", True))
    ncolp = 3; nrowp = (len(panels) + ncolp - 1) // ncolp
    fig_bd, axes = plt.subplots(nrowp, ncolp, figsize=(4.2 * ncolp, 4.0 * nrowp), squeeze=False)
    axes = axes.ravel()
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
        ax.set_title(f"{ttl}  t*={ts:.2f}, |∂⁺|={int(boundary.sum())}", fontsize=8,
                     weight="bold" if is_mean else "normal")
    proxy = [Line2D([], [], marker='o', ls='', color="#888888", label="non sél."),
             Line2D([], [], marker='o', ls='', color="#2CA02C", label="sél. hors ∂⁺"),
             Line2D([], [], marker='o', ls='', color="#ff7f0e", label="∂⁺(t*)"),
             Line2D([], [], marker='*', ls='', color="#C41E3A", label="vraies (★)"),
             Line2D([], [], color="#1f77b4", ls='--', label="t*")]
    fig_bd.legend(handles=proxy, loc="upper center", ncol=5, fontsize=8)
    fig_bd.supxlabel("score knockoff", fontsize=9); fig_bd.supylabel("score réel", fontsize=9)
    fig_bd.suptitle(f"Barrière ∂⁺(t*) de V2_constr_a0 par run + moyenne — {TITLE}", fontsize=9, y=0.998)
    fig_bd.tight_layout(rect=[0, 0, 1, 0.95])
    fig_bd.savefig(os.path.join(OUT, "barriere_dplus_a0.png"), dpi=140)

# ── PDF combiné ───────────────────────────────────────────────────────────────
pages = [fig_table, fig_prf, fig_roc, fig_scores, fig_scores_v1, fig_bar,
         fig_curves, fig_obj, fig_bd, fig_var, fig_jac]
pages = [p for p in pages if p is not None]
with PdfPages(os.path.join(OUT, "analyse_complete.pdf")) as pdf:
    for f in pages:
        pdf.savefig(f)
for f in pages:
    plt.close(f)


# ══════════════════════════ CSV agrégés ═══════════════════════════════════════
_ts    = datetime.now().isoformat(timespec="seconds")
_seeds = ";".join(str(s) for s in SEEDS)

# runs_metrics.csv : 1 ligne / (appel × modèle), mean+std agrégés sur les seeds
MCOLS = ["timestamp", "artificial", "n", "p", "k", "signal", "n_bootstraps", "n_seeds", "seeds",
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
            "timestamp": _ts, "artificial": ART_LABEL, "n": N, "p": P, "k": K,
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
FCOLS = ["timestamp", "artificial", "n", "p", "k", "signal", "n_bootstraps", "n_seeds",
         "model", "feature", "is_true", "score_mean", "score_std", "score_ko_mean", "sel_freq"]
_fh = not os.path.exists(FEAT_CSV_PATH); _nf = 0
with open(FEAT_CSV_PATH, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FCOLS)
    if _fh: w.writeheader()
    for m in STABL:
        for j in range(P):
            w.writerow({
                "timestamp": _ts, "artificial": ART_LABEL, "n": N, "p": P, "k": K,
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
DB_COLS = ["seed", "run", "artificial", "n", "p", "k", "signal", "n_bootstraps",
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
            base = {
                "seed": SEEDS[ri], "run": ri + 1, "artificial": ART_LABEL,
                "n": N, "p": P, "k": K, "signal": SIGNAL, "n_bootstraps": B, "model": name,
                "n_sel": rm["ns"], "precision": round(rm["precision"], 4),
                "recall": round(rm["recall"], 4), "f1": round(rm["f1"], 4),
                "fdp": round(rm["fdp"], 4), "auc": round(rm["auc"], 4),
                "auc_mean": round(am["auc"].mean(), 4), "fdp_mean": round(am["fdp"].mean(), 4),
                "f1_mean": round(am["f1"].mean(), 4), "precision_mean": round(am["precision"].mean(), 4),
                "recall_mean": round(am["recall"].mean(), 4), "jaccard_mean": round(JAC[name], 4),
            }
            for j in range(P):
                row = dict(base, feature=j, is_true=int(j < K), selected=int(j in ss))
                if fe is not None:
                    row["score"]    = round(float(fe["sr"][j]), 6)
                    row["score_ko"] = round(float(fe["sr_ko"][j]), 6)
                else:
                    row["score"] = ""; row["score_ko"] = ""       # baselines : pas de score
                w.writerow(row); _db += 1
print(f"Base de données locale ({_db} lignes) -> {DB_PATH}")
print(f"\nPDF + figures dans : {OUT}")
