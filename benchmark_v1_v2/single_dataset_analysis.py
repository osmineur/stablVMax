"""
single_dataset_analysis.py
Analyse complète sur UN dataset synthétique, pour STABL (V1, V2_unc, V2_constr,
V2_wj) + baselines (ALasso, Lasso, ElasticNet, XGBoost).

Paramétrable :
  --artificial  knockoff | perm     (type de feature artificielle)
  --n --p --k --signal --B --seed   (paramètres du dataset / bootstraps)

Crée un dossier results_synthetic/<tag>_n<N>_p<P>_k<K>_sig<SIGNAL>/ contenant :
  1. roc_curves            — courbes ROC AUC de chaque modèle
  2. score_distributions   — scores réels vs artificiels (histogramme + scatter)
  3. fdp_par_modele        — vrai FDP (vérité terrain) par modèle
  4. fdp_curves            — FDP+(t) vs vrai FDP(t) ; V2_constr en objectif décomposé
  5. v2constr_objective    — figure agrandie de l'objectif FDP+ + frontière (V2_constr)
  + analyse_complete.pdf   — tout à la suite (5 pages)

Exemples :
  python single_dataset_analysis.py --artificial knockoff --n 100 --p 200 --k 5 --signal 10
  python single_dataset_analysis.py --artificial perm     --n 200 --p 500 --k 10 --signal 5 --B 1000
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
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
args = ap.parse_args()

N, P, K, SIGNAL, B, SEED = args.n, args.p, args.k, args.signal, args.B, args.seed
ART = "random_permutation" if args.artificial == "perm" else "knockoff"
TAG = "perm" if args.artificial == "perm" else "ko"
ART_LABEL = "permutation" if args.artificial == "perm" else "knockoff"

FOLDER = f"{TAG}_n{N}_p{P}_k{K}_sig{SIGNAL:g}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_synthetic", FOLDER)
os.makedirs(OUT, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
X = rng.standard_normal((N + 2000, P))
beta = np.zeros(P); beta[:K] = SIGNAL / np.sqrt(K)
yl = X[:, :K] @ beta[:K] + rng.standard_normal(N + 2000)
y = (yl > 0).astype(int)
Xtr, Xte, ytr, yte = X[:N], X[N:], y[:N], y[N:]
cov = LedoitWolf().fit(Xtr).covariance_ if ART == "knockoff" else None
print(f"Dataset: artificial={ART}, n={N}, p={P}, k={K}, signal={SIGNAL}, B={B}, "
      f"balance train={ytr.mean():.2f}")
print(f"Vraies features = indices 0..{K-1}")
print(f"Dossier de sortie : {OUT}\n")

def _lr():
    return LogisticRegression(penalty="l1", solver="liblinear",
                              class_weight="balanced", max_iter=int(1e6))

def stabl_v1():
    return StablV1(base_estimator=_lr(), lambda_grid="auto", n_lambda=10,
                   n_bootstraps=B, artificial_type=ART,
                   artificial_proportion=1.0, sample_fraction=0.5, replace=False,
                   n_jobs=-1, random_state=SEED)

def stabl_v2(mode):
    kw = dict(base_estimator=_lr(), lambda_grid="auto", n_lambda=10,
              n_bootstraps=B, artificial_type=ART, artificial_proportion=1.0,
              sample_fraction=0.5, replace=False, n_jobs=-1, random_state=SEED,
              selection_mode=mode, delta=0.05)
    if ART == "knockoff":
        kw.update(knockoff_method="equicorrelated", cov_matrix=cov)
    return StablV2(**kw)

def baseline(est, grid):
    n_splits = max(2, min(5, int(np.min(np.bincount(ytr)))))
    return GridSearchCV(est, grid, cv=n_splits, scoring="roc_auc", n_jobs=-1)

_CG = {"C": np.logspace(-2, 0, 10)}
MODELS = {
    "V1":         stabl_v1(),
    "V2_unc":     stabl_v2("unconstrained"),
    "V2_constr":  stabl_v2("constrained"),
    "V2_wj":      stabl_v2("wj"),
    "ALasso":     baseline(ALogitLasso(solver="liblinear", class_weight="balanced",
                                       tol=1e-4, max_iter=int(1e6)), _CG),
    "Lasso":      baseline(LogisticRegression(penalty="l1", solver="liblinear",
                           class_weight="balanced", max_iter=int(1e6)), _CG),
    "ElasticNet": baseline(LogisticRegression(penalty="elasticnet", solver="saga",
                           l1_ratio=0.5, class_weight="balanced", max_iter=int(1e6)), _CG),
}
if HAS_XGB:
    MODELS["XGBoost"] = baseline(
        XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1, verbosity=0),
        {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.2]})

def get_support(model):
    if hasattr(model, "get_support"):
        return np.where(model.get_support())[0]
    est = model.best_estimator_
    if hasattr(est, "coef_"):
        return np.where(np.abs(est.coef_[0]) > 1e-8)[0]
    return np.where(est.feature_importances_ > 0)[0]

# ── fit + ROC ─────────────────────────────────────────────────────────────────
roc_data = {}; fitted = {}; summary = []
print(f"{'Modèle':<12} {'#sél':>5} {'#vraies':>8} {'#nulles':>8} {'FDP':>6} {'AUC':>7}")
print("-" * 52)
for name, model in MODELS.items():
    model.fit(Xtr, ytr)
    fitted[name] = model
    sel = get_support(model); ns = len(sel)
    n_true = int((sel < K).sum()); n_null = ns - n_true
    fdp = n_null / max(1, ns)
    precision = n_true / ns if ns > 0 else 0.0
    recall = n_true / K
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    if ns > 0:
        clf = LogisticRegression(penalty="l2", solver="lbfgs",
                                 class_weight="balanced", max_iter=10000)
        clf.fit(Xtr[:, sel], ytr)
        prob = clf.predict_proba(Xte[:, sel])[:, 1]
        auc = roc_auc_score(yte, prob); fpr, tpr, _ = roc_curve(yte, prob)
    else:
        auc = 0.5; fpr, tpr = [0, 1], [0, 1]
    roc_data[name] = (fpr, tpr, auc)
    summary.append(dict(name=name, ns=ns, n_true=n_true, n_null=n_null, fdp=fdp,
                        precision=precision, recall=recall, f1=f1, auc=auc))
    print(f"{name:<12} {ns:>5} {n_true:>8} {n_null:>8} {fdp:>6.2f} "
          f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} AUC={auc:.3f}")

TITLE = f"{ART_LABEL} — n={N}, p={P}, k={K}, signal={SIGNAL}, B={B}"

# ── Figure 0 : tableau récapitulatif de sélection ─────────────────────────────
fig_table, ax = plt.subplots(figsize=(12, 0.9 + 0.42 * len(summary)))
ax.axis("off")
col_labels = ["Modèle", "# sélec.", "# vraies", "# nulles",
              "Precision", "Recall", "F1", "FDP", "AUC"]
cell_text = [[r["name"], str(r["ns"]), f"{r['n_true']}/{K}", str(r["n_null"]),
              f"{r['precision']:.2f}", f"{r['recall']:.2f}", f"{r['f1']:.2f}",
              f"{r['fdp']:.2f}", f"{r['auc']:.3f}"] for r in summary]
tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
for j in range(len(col_labels)):                       # en-tête en gras/coloré
    c = tbl[0, j]; c.set_facecolor("#001A7B"); c.set_text_props(color="white", weight="bold")
for i, r in enumerate(summary, start=1):               # surligne les modèles STABL
    if r["name"].startswith("V"):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor("#F2E6E9")
ax.set_title(f"Sélection par modèle — {TITLE}\n({K} vraies features sur {P})",
             fontsize=11, pad=16)
fig_table.tight_layout()
fig_table.savefig(os.path.join(OUT, "selection_table.png"), dpi=130)

# ── Figure 0bis : comparaison Precision / Recall / F1 ─────────────────────────
fig_prf, ax = plt.subplots(figsize=(11, 4.8))
mnames = [r["name"] for r in summary]
xpos = np.arange(len(mnames)); w = 0.26
ax.bar(xpos - w, [r["precision"] for r in summary], w, label="Precision", color="#1f77b4")
ax.bar(xpos,     [r["recall"]    for r in summary], w, label="Recall",    color="#2CA02C")
ax.bar(xpos + w, [r["f1"]        for r in summary], w, label="F1",        color="#C41E3A")
ax.set_xticks(xpos); ax.set_xticklabels(mnames, rotation=25, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.set_title(f"Precision / Recall / F1 par modèle — {TITLE}")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig_prf.tight_layout()
fig_prf.savefig(os.path.join(OUT, "precision_recall_f1.png"), dpi=130)

# ── Figure 1 : ROC ────────────────────────────────────────────────────────────
fig_roc, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=1)
ax.set_xlabel("Taux de faux positifs (FPR)"); ax.set_ylabel("Taux de vrais positifs (TPR)")
ax.set_title(f"Courbes ROC — {TITLE}")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
fig_roc.tight_layout(); fig_roc.savefig(os.path.join(OUT, "roc_curves.png"), dpi=130)

# ── Figure 2 : distributions de scores (V2) ───────────────────────────────────
v2 = MODELS["V2_wj"]
sc_real = v2.stabl_scores_.max(axis=1); sc_ko = v2.stabl_scores_artificial_.max(axis=1)
sc_true = sc_real[:K]; sc_null = sc_real[K:]
fig_scores, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
bins = np.linspace(0, max(sc_real.max(), sc_ko.max()) + 0.02, 30)
ax.hist(sc_null, bins=bins, alpha=0.5, color="#4D4F53", label=f"réelles nulles (n={P-K})")
ax.hist(sc_ko,   bins=bins, alpha=0.5, color="#1f77b4", label=f"{ART_LABEL}s (n={P})")
for i, s in enumerate(sc_true):
    ax.axvline(s, color="#C41E3A", lw=2, label="réelles vraies" if i == 0 else None)
ax.set_xlabel("Score de stabilité max_λ"); ax.set_ylabel("Nombre de features")
ax.set_title("Distribution des scores (V2)"); ax.legend(fontsize=8)
ax = axes[1]
ax.scatter(sc_ko[K:], sc_null, s=14, alpha=0.5, color="#4D4F53", label="nulles")
ax.scatter(sc_ko[:K], sc_true, s=60, color="#C41E3A", marker="*", label="vraies", zorder=5)
lim = max(sc_real.max(), sc_ko.max()) + 0.02
ax.plot([0, lim], [0, lim], ls=":", color="gray", label=f"score = score_{ART_LABEL[:2]}")
ax.set_xlabel(f"Score {ART_LABEL} (artificiel)"); ax.set_ylabel("Score réel")
ax.set_title(f"Réel vs {ART_LABEL} par feature"); ax.legend(fontsize=8)
fig_scores.suptitle(f"Scores STABL-V2 — {TITLE}"); fig_scores.tight_layout()
fig_scores.savefig(os.path.join(OUT, "score_distributions.png"), dpi=130)

# ── Figure 3 : vrai FDP par modèle ────────────────────────────────────────────
fig_bar, ax = plt.subplots(figsize=(9, 4.5))
names = [r["name"] for r in summary]; fdps = [r["fdp"] for r in summary]
cols = ["#C41E3A" if n.startswith("V") else "#4D4F53" for n in names]
bars = ax.bar(range(len(names)), fdps, color=cols, alpha=0.8)
for b, f in zip(bars, fdps):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{f:.2f}",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, ha="right")
ax.set_ylabel("Vrai FDP (vérité terrain)"); ax.set_title(f"Vrai FDP par modèle — {TITLE}")
ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3); fig_bar.tight_layout()
fig_bar.savefig(os.path.join(OUT, "fdp_par_modele.png"), dpi=130)

# ── Figure 4 : FDP+ vs vrai FDP (V2_constr = objectif décomposé) ──────────────
def true_fdp_curve(sr, grid):
    out = []
    for t in grid:
        S = np.where(sr > t)[0]
        out.append(((S >= K).sum()/max(1, len(S))) if len(S) else 0.0)
    return np.array(out)

stabl_names = [n for n in ["V1", "V2_unc", "V2_constr", "V2_wj"] if n in fitted]
fig_curves, axes = plt.subplots(2, 2, figsize=(13, 9)); axes = axes.ravel()
for ax, name in zip(axes, stabl_names):
    m = fitted[name]; grid = np.asarray(m.fdr_threshold_range)
    sr = m.stabl_scores_.max(axis=1); fdp_plus = np.asarray(m.FDRs_)
    fdp_true = true_fdp_curve(sr, grid); tstar = m.fdr_min_threshold_
    eps_tot = getattr(m, "eps_B_total_fw_", None)
    if eps_tot is not None:
        D = np.array([max(1, int((sr > t).sum())) for t in grid])
        fro = np.array([((sr > t) & (sr <= t + eps_tot)).sum() for t in grid]) / D
        feasible = fro == 0
    else:
        fro = None; feasible = None
    if name == "V2_constr" and fro is not None:
        obj = fdp_plus + fro
        if feasible.any():
            ax.fill_between(grid, 0, 1.05, where=feasible, step="post",
                            alpha=0.10, color="#2CA02C", label="zone faisible (∂⁺=∅)")
        ax.plot(grid, fdp_plus, color="#999999", lw=1.3, label="FDP+(t) seul")
        ax.fill_between(grid, fdp_plus, obj, step="post", alpha=0.35,
                        color="#2CA02C", label="frontière |∂⁺(t)|/D(t)")
        ax.plot(grid, obj, color="#1f77b4", lw=2.4, label="OBJECTIF = FDP+ + frontière")
        ax.plot(grid, fdp_true, color="#C41E3A", lw=2, ls="--", label="vrai FDP(t)")
        imin = int(np.argmin(obj))
        ax.axvline(grid[imin], color="black", ls=":", lw=1.6, label=f"t*={grid[imin]:.2f}")
    else:
        ax.plot(grid, fdp_plus, color="#1f77b4", lw=2, label="FDP+(t) estimé")
        ax.plot(grid, fdp_true, color="#C41E3A", lw=2, ls="--", label="vrai FDP(t)")
        ax.axvline(tstar, color="black", ls=":", lw=1.5, label=f"t*={tstar:.2f}")
        if fro is not None:
            ax2 = ax.twinx()
            ax2.fill_between(grid, 0, fro, step="post", alpha=0.15, color="#2CA02C")
            ax2.set_ylabel("|∂⁺(t)|/D(t)", color="#2CA02C", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="#2CA02C", labelsize=7)
            ax.fill_between(grid, 0, 1.0, where=feasible, step="post",
                            alpha=0.07, color="#2CA02C")
    ax.set_xlabel("Seuil t"); ax.set_ylabel("FDP")
    ax.set_title(f"{name}  (mode={getattr(m,'selection_mode','static')})")
    ax.set_ylim(0, 1.05); ax.legend(loc="upper right", fontsize=6)
fig_curves.suptitle(f"FDP+ estimé vs vrai FDP, frontière & zone faisible BW — {TITLE}", y=1.0)
fig_curves.tight_layout(); fig_curves.savefig(os.path.join(OUT, "fdp_curves.png"), dpi=130)

# ── Figure 5 : V2_constr objectif agrandi ─────────────────────────────────────
mc = fitted["V2_constr"]; grid = np.asarray(mc.fdr_threshold_range)
sr = mc.stabl_scores_.max(axis=1); fdpplus = np.asarray(mc.FDRs_)
eps_tot = mc.eps_B_total_fw_
D = np.array([max(1, int((sr > t).sum())) for t in grid])
fro = np.array([((sr > t) & (sr <= t + eps_tot)).sum() for t in grid]) / D
obj = fdpplus + fro; fdp_true = true_fdp_curve(sr, grid); feasible = fro == 0
fig_obj, ax = plt.subplots(figsize=(11, 6.5))
if feasible.any():
    ax.fill_between(grid, 0, 1.05, where=feasible, step="post", alpha=0.10,
                    color="#2CA02C", label="zone faisible (∂⁺=∅)")
ax.plot(grid, fdpplus, color="#999999", lw=1.5, label="FDP+(t) seul")
ax.fill_between(grid, fdpplus, obj, step="post", alpha=0.35, color="#2CA02C",
                label="frontière |∂⁺(t)|/D(t)")
ax.plot(grid, obj, color="#1f77b4", lw=2.6, label="OBJECTIF = FDP+(t) + |∂⁺(t)|/D(t)")
ax.plot(grid, fdp_true, color="#C41E3A", lw=2, ls="--", label="vrai FDP(t)")
imin = int(np.argmin(obj))
ax.axvline(grid[imin], color="black", ls=":", lw=1.8, label=f"t* = {grid[imin]:.2f}")
ax.scatter([grid[imin]], [obj[imin]], color="#1f77b4", s=90, zorder=6,
           label=f"min objectif (obj={obj[imin]:.2f})")
ax.set_xlabel("Seuil t"); ax.set_ylabel("FDP"); ax.set_ylim(0, 1.05)
ax.set_title(f"V2_constr — objectif FDP+ + frontière minimisé\n{TITLE}")
ax.legend(loc="upper center", fontsize=9, ncol=2); ax.grid(alpha=0.3)
fig_obj.tight_layout(); fig_obj.savefig(os.path.join(OUT, "v2constr_objective.png"), dpi=140)

# ── Figure 6 : barrière ∂⁺(t*) sur le scatter réel vs knockoff (V2_constr) ─────
# ∂⁺(t*) = {j : t* < score(j) ≤ t* + ε_WJ,j}  avec ε_WJ,j = εB,j + εB,j,ko.
# La barrière n'est définie qu'à t* (seuil choisi par V2_constr). On trace pour
# chaque feature sélectionnée une barre d'erreur descendante de longueur ε_WJ,j :
# si elle franchit t*, la feature est dans la barrière (incertaine).
mcb     = fitted["V2_constr"]
srb     = mcb.stabl_scores_.max(axis=1)
srkob   = mcb.stabl_scores_artificial_.max(axis=1)
eps_wj  = mcb.eps_B_total_fw_                      # ε_WJ,j par feature
gridb   = np.asarray(mcb.fdr_threshold_range)
fdppb   = np.asarray(mcb.FDRs_)
Db      = np.array([max(1, int((srb > t).sum())) for t in gridb])
frob    = np.array([((srb > t) & (srb <= t + eps_wj)).sum() for t in gridb]) / Db
tstarb  = gridb[int(np.argmin(fdppb + frob))]      # t* = argmin (FDP+ + frontière)

selected = srb > tstarb
boundary = selected & (srb <= tstarb + eps_wj)     # ∂⁺(t*)
safe     = selected & ~boundary
idx_all  = np.arange(P)

fig_bd, ax = plt.subplots(figsize=(8.5, 7.5))
# barres d'erreur ε_WJ,j (descendantes) pour les features sélectionnées
for j in np.where(selected)[0]:
    col = "#ff7f0e" if boundary[j] else "#2CA02C"
    ax.plot([srkob[j], srkob[j]], [srb[j] - eps_wj[j], srb[j]], color=col, alpha=0.45, lw=1.2, zorder=2)
# nuages de points par catégorie (nulles)
def _scatter(mask, color, label, s=22):
    i = np.where(mask & (idx_all >= K))[0]
    if len(i): ax.scatter(srkob[i], srb[i], s=s, color=color, alpha=0.7, label=label, zorder=3)
_scatter(~selected, "#CCCCCC", "non sélectionnées (nulles)", s=14)
_scatter(safe,     "#2CA02C", "sélectionnées hors barrière (nulles)")
_scatter(boundary, "#ff7f0e", "∂⁺(t*) — dans la barrière (nulles)", s=34)
# vraies features en étoiles, même code couleur
for j in range(K):
    col = "#ff7f0e" if boundary[j] else ("#2CA02C" if selected[j] else "#CCCCCC")
    ax.scatter(srkob[j], srb[j], s=150, marker="*", color=col,
               edgecolor="black", lw=0.6, zorder=6)
# repères : t* et la diagonale
lim = max(srb.max(), srkob.max()) + 0.04
ax.axhline(tstarb, color="#1f77b4", ls="--", lw=1.6, label=f"t* = {tstarb:.2f}")
ax.plot([0, lim], [0, lim], ls=":", color="gray", lw=1, label="score = score_ko")
n_bd = int(boundary.sum())
ax.set_xlabel("score knockoff (score_ko)"); ax.set_ylabel("score réel (score)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_title(f"Barrière ∂⁺(t*) de V2_constr — |∂⁺(t*)|={n_bd}, ★=vraies\n"
             f"barre verticale = ε_WJ,j (= εB,j+εB,j,ko) — {TITLE}", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig_bd.tight_layout(); fig_bd.savefig(os.path.join(OUT, "barriere_dplus.png"), dpi=140)

# ── PDF combiné ───────────────────────────────────────────────────────────────
pages = (fig_table, fig_prf, fig_roc, fig_scores, fig_bar, fig_curves, fig_obj, fig_bd)
with PdfPages(os.path.join(OUT, "analyse_complete.pdf")) as pdf:
    for f in pages:
        pdf.savefig(f)
for f in pages:
    plt.close(f)

print(f"\nScores réels — vraies : {sc_true.round(3)}")
print(f"Scores réels — nulles : mean={sc_null.mean():.3f} max={sc_null.max():.3f}")
print(f"Scores {ART_LABEL:<11}: mean={sc_ko.mean():.3f} max={sc_ko.max():.3f}")
print(f"\nPDF combiné + figures dans : {OUT}")
