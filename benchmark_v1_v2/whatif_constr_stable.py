"""
whatif_constr_stable.py  (VISUALISATION SEULE — ne modifie pas la pipeline)

Compare sur UN dataset :
  - V2_constr        : S = {score > t*}                (garde la barrière)
  - V2_constr_stable : S' = {score > t* + eps_WJ,j}    (retire ∂⁺(t*))
avec FDP+ calculé correctement : numérateur |S_ko(t*)| ancré à t*, dénominateur |.|.

Defaut : knockoff, n=100, p=300, k=10, signal=7, B=1000.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from stabl.stablV2 import Stabl as StablV2

ap = argparse.ArgumentParser()
ap.add_argument("--artificial", choices=["knockoff", "perm"], default="knockoff")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--p", type=int, default=300)
ap.add_argument("--k", type=int, default=10);  ap.add_argument("--signal", type=float, default=7.0)
ap.add_argument("--B", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
N, P, K, SIGNAL, B, SEED = args.n, args.p, args.k, args.signal, args.B, args.seed
ART = "random_permutation" if args.artificial == "perm" else "knockoff"
TAG = "perm" if args.artificial == "perm" else "ko"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_synthetic",
                   f"whatif_stable_{TAG}_n{N}_p{P}_k{K}_sig{SIGNAL:g}")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(SEED)
X = rng.standard_normal((N + 2000, P))
beta = np.zeros(P); beta[:K] = SIGNAL / np.sqrt(K)
yl = X[:, :K] @ beta[:K] + rng.standard_normal(N + 2000)
y = (yl > 0).astype(int)
Xtr, ytr = X[:N], y[:N]
cov = LedoitWolf().fit(Xtr).covariance_ if ART == "knockoff" else None

kw = dict(base_estimator=LogisticRegression(penalty="l1", solver="liblinear",
          class_weight="balanced", max_iter=int(1e6)), lambda_grid="auto", n_lambda=10,
          n_bootstraps=B, artificial_type=ART, artificial_proportion=1.0,
          sample_fraction=0.5, replace=False, n_jobs=-1, random_state=SEED,
          selection_mode="constrained", delta=0.05)
if ART == "knockoff":
    kw.update(knockoff_method="equicorrelated", cov_matrix=cov)
m = StablV2(**kw); m.fit(Xtr, ytr)

sr     = m.stabl_scores_.max(axis=1)
srko   = m.stabl_scores_artificial_.max(axis=1)
eps_wj = m.eps_B_total_fw_
grid   = np.asarray(m.fdr_threshold_range)
fdpp   = np.asarray(m.FDRs_)
D      = np.array([max(1, int((sr > t).sum())) for t in grid])
fro    = np.array([((sr > t) & (sr <= t + eps_wj)).sum() for t in grid]) / D
tstar  = grid[int(np.argmin(fdpp + fro))]

# knockoffs au-dessus de t* (numérateur ancré, r=1 car proportion=1)
nko_tstar = int((srko > tstar).sum())

def report(mask, label):
    sel = np.where(mask)[0]; ns = len(sel)
    nt = int((sel < K).sum()); nn = ns - nt
    true_fdp = nn / max(1, ns)
    fdp_plus = (nko_tstar + 1) / max(1, ns)   # numérateur |S_ko(t*)| + 1, ancré à t*
    print(f"{label:<22} |S|={ns:3d}  vraies={nt}/{K}  nulles={nn:3d}  "
          f"vrai_FDP={true_fdp:.2f}  FDP+={fdp_plus:.2f}")
    return ns, nt, nn, true_fdp, fdp_plus

S_constr = sr > tstar                       # garde la barrière
S_stable = sr > tstar + eps_wj              # retire ∂⁺(t*)
boundary = S_constr & ~S_stable

print(f"t* = {tstar:.3f}   |   knockoffs au-dessus de t* : |S_ko(t*)| = {nko_tstar}")
print(f"features dans la barriere ∂+(t*) retirees : {int(boundary.sum())}\n")
r_c = report(S_constr, "V2_constr (standard)")
r_s = report(S_stable, "V2_constr_stable")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7.5))
for j in np.where(S_constr)[0]:
    col = "#ff7f0e" if boundary[j] else "#2CA02C"
    ax.plot([srko[j], srko[j]], [sr[j] - eps_wj[j], sr[j]], color=col, alpha=0.45, lw=1.2)
def sc(mask, color, label, s=22, marker="o"):
    i = np.where(mask & (np.arange(P) >= K))[0]
    if len(i): ax.scatter(srko[i], sr[i], s=s, color=color, alpha=0.7, label=label, marker=marker)
sc(~S_constr, "#CCCCCC", "non sélectionnées", s=14)
sc(S_stable,  "#2CA02C", "gardées (stable, hors barrière)")
sc(boundary,  "#ff7f0e", "RETIRÉES (∂⁺(t*), instables)", s=34)
for j in range(K):
    col = "#ff7f0e" if boundary[j] else ("#2CA02C" if S_stable[j] else "#CCCCCC")
    ax.scatter(srko[j], sr[j], s=150, marker="*", color=col, edgecolor="black", lw=0.6, zorder=6)
lim = max(sr.max(), srko.max()) + 0.04
ax.axhline(tstar, color="#1f77b4", ls="--", lw=1.6, label=f"t* = {tstar:.2f}")
ax.plot([0, lim], [0, lim], ls=":", color="gray", lw=1)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("score knockoff"); ax.set_ylabel("score réel")
ax.set_title(f"V2_constr standard vs stable (retrait ∂⁺(t*)) — {TAG} n={N} p={P} k={K} sig={SIGNAL}\n"
             f"standard: |S|={r_c[0]} vrai_FDP={r_c[3]:.2f} FDP+={r_c[4]:.2f}  |  "
             f"stable: |S|={r_s[0]} vrai_FDP={r_s[3]:.2f} FDP+={r_s[4]:.2f}", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "constr_stable.png"), dpi=140)
fig.savefig(os.path.join(OUT, "constr_stable.pdf"))
print(f"\nFigure : {OUT}/constr_stable.png|pdf")
