"""
whatif_constr_wj.py  (VISUALISATION SEULE — ne modifie pas la pipeline)

Cas : n=100, p=400, k=5, signal=10, knockoff, B=1000.
Question : que se passe-t-il si V2_constr ajoute AUSSI la contrainte WJ (W_j > ε_WJ) ?

On fitte V2 une fois (scores identiques entre modes), puis on combine les masques :
  - V2_constr      : score(j) > t*_constr   (t* = argmin FDP+ + frontière)
  - WJ             : W_j = score(j)-score_ko(j) > ε_WJ,j
  - V2_constr ∩ WJ : les deux à la fois
Sortie : results_synthetic/whatif_constr_wj/
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from stabl.stablV2 import Stabl as StablV2

ap = argparse.ArgumentParser()
ap.add_argument("--artificial", choices=["knockoff", "perm"], default="knockoff")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--p", type=int, default=400)
ap.add_argument("--k", type=int, default=5);   ap.add_argument("--signal", type=float, default=10.0)
ap.add_argument("--B", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
N, P, K, SIGNAL, B, SEED = args.n, args.p, args.k, args.signal, args.B, args.seed
ART = "random_permutation" if args.artificial == "perm" else "knockoff"
TAG = "perm" if args.artificial == "perm" else "ko"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_synthetic",
                   f"whatif_constr_wj_{TAG}_n{N}_p{P}_k{K}_sig{SIGNAL:g}")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(SEED)
X = rng.standard_normal((N + 2000, P))
beta = np.zeros(P); beta[:K] = SIGNAL / np.sqrt(K)
yl = X[:, :K] @ beta[:K] + rng.standard_normal(N + 2000)
y = (yl > 0).astype(int)
Xtr, Xte, ytr, yte = X[:N], X[N:], y[:N], y[N:]
cov = LedoitWolf().fit(Xtr).covariance_ if ART == "knockoff" else None

# fit en mode wj (donne w_paired_, eps_paired_, FDRs_, eps_B_total_fw_)
kw = dict(base_estimator=LogisticRegression(penalty="l1", solver="liblinear",
          class_weight="balanced", max_iter=int(1e6)),
          lambda_grid="auto", n_lambda=10, n_bootstraps=B, artificial_type=ART,
          artificial_proportion=1.0, sample_fraction=0.5, replace=False, n_jobs=-1,
          random_state=SEED, selection_mode="wj", delta=0.05)
if ART == "knockoff":
    kw.update(knockoff_method="equicorrelated", cov_matrix=cov)
m = StablV2(**kw)
m.fit(Xtr, ytr)

sr     = m.stabl_scores_.max(axis=1)
sr_ko  = m.stabl_scores_artificial_.max(axis=1)
grid   = np.asarray(m.fdr_threshold_range)
fdpp   = np.asarray(m.FDRs_)
W      = m.w_paired_          # score - score_ko
eps_wj = m.eps_paired_         # ε_WJ,j = ε_j + ε_ko,j
eps_tot = m.eps_B_total_fw_

# t* de V2_constr = argmin (FDP+ + frontière)
D   = np.array([max(1, int((sr > t).sum())) for t in grid])
fro = np.array([((sr > t) & (sr <= t + eps_tot)).sum() for t in grid]) / D
obj = fdpp + fro
tstar_constr = grid[int(np.argmin(obj))]

# les trois masques
mask_constr = sr > tstar_constr
mask_wj     = W > eps_wj
mask_both   = mask_constr & mask_wj

def metrics(mask, label):
    sel = np.where(mask)[0]; ns = len(sel)
    nt = int((sel < K).sum()); nn = ns - nt
    fdp = nn / max(1, ns)
    auc = 0.5
    if ns > 0:
        clf = LogisticRegression(penalty="l2", solver="lbfgs",
                                 class_weight="balanced", max_iter=10000)
        clf.fit(Xtr[:, sel], ytr)
        auc = roc_auc_score(yte, clf.predict_proba(Xte[:, sel])[:, 1])
    print(f"{label:<22} #sél={ns:3d}  vraies={nt}/{K}  nulles={nn:3d}  FDP={fdp:.2f}  AUC={auc:.3f}")
    return ns, nt, nn, fdp, auc

print(f"t*_constr = {tstar_constr:.3f}\n")
r_constr = metrics(mask_constr, "V2_constr seul")
r_wj     = metrics(mask_wj,     "WJ seul")
r_both   = metrics(mask_both,   "V2_constr ∩ WJ")

# ── figure : scatter score vs score_ko, coloré par appartenance ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# gauche : scatter dans l'espace (score_ko, score)
ax = axes[0]
cat = np.full(P, "aucun", dtype=object)
cat[mask_constr & ~mask_wj] = "constr seul"
cat[mask_wj & ~mask_constr] = "wj seul"
cat[mask_both]              = "les deux"
colors = {"aucun": "#CCCCCC", "constr seul": "#1f77b4",
          "wj seul": "#ff7f0e", "les deux": "#2CA02C"}
for c, col in colors.items():
    idx = np.where((cat == c) & (np.arange(P) >= K))[0]
    if len(idx): ax.scatter(sr_ko[idx], sr[idx], s=16, alpha=0.6, color=col, label=f"{c} (nulle)")
# vraies features en étoiles, colorées pareil
for c, col in colors.items():
    idx = np.where((cat == c) & (np.arange(P) < K))[0]
    if len(idx): ax.scatter(sr_ko[idx], sr[idx], s=130, color=col, marker="*",
                            edgecolor="black", linewidth=0.5, zorder=6)
ax.axhline(tstar_constr, color="#1f77b4", ls="--", lw=1.5, label=f"t*_constr={tstar_constr:.2f}")
lim = max(sr.max(), sr_ko.max()) + 0.03
xs = np.linspace(0, lim, 50)
ax.plot(xs, xs + eps_wj.mean(), color="#ff7f0e", ls="--", lw=1.5,
        label=f"WJ: score=score_ko+ε̄  (ε̄={eps_wj.mean():.2f})")
ax.set_xlabel("score knockoff"); ax.set_ylabel("score réel")
ax.set_title("Qui est sélectionné par quoi\n(★ = vraies features)")
ax.legend(fontsize=7, loc="lower right"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)

# droite : barres comparatives
ax = axes[1]
labels = ["V2_constr\nseul", "WJ\nseul", "V2_constr\n∩ WJ"]
res = [r_constr, r_wj, r_both]
xpos = np.arange(3); wdt = 0.35
ax.bar(xpos - wdt/2, [r[1] for r in res], wdt, color="#2CA02C", label="# vraies")
ax.bar(xpos + wdt/2, [r[2] for r in res], wdt, color="#C41E3A", label="# nulles (FP)")
for i, r in enumerate(res):
    ax.text(i, max(r[1], r[2]) + 0.5, f"FDP={r[3]:.2f}\nAUC={r[4]:.3f}",
            ha="center", fontsize=9)
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel("Nombre de features"); ax.set_title("Vraies vs nulles sélectionnées")
ax.legend(); ax.grid(axis="y", alpha=0.3)

fig.suptitle(f"V2_constr + contrainte WJ — {TAG}, n={N}, p={P}, k={K}, signal={SIGNAL}, B={B}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "constr_plus_wj.png"), dpi=140)
fig.savefig(os.path.join(OUT, "constr_plus_wj.pdf"))
print(f"\nFigure : {OUT}/constr_plus_wj.png|pdf")
