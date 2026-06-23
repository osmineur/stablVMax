#!/usr/bin/env python3
"""
Spectre — Cross-Reproducibility Spectral Selection, modèle GAZ DE COULOMB / UNFOLDING
=====================================================================================
Opérateur CROISÉ M̂ = (1/2B) Σ_b (u_b w_bᵀ + w_b u_bᵀ)  (split disjoint A,B préservé).
  -> cible M∞ = ppᵀ, diagonale p_j²  => effondrement quadratique (résistance winner's curse).
On lit le spectre comme un GAZ DE COULOMB :
  1. CDF spectrale croissante  N(λ) = #{λ_i ≤ λ}/p_eff.
  2. UNFOLDING par noyau intégré : Ñ_h(λ) = (1/p_eff) Σ_i Φ((λ−λ_i)/h)  (somme de sigmoïdes, croissante).
     ξ_k = Ñ_h(λ_k) ;  espacements renormalisés  s_k = p_eff (ξ_k − ξ_{k+1}).
  3. CALIBRATION de h par AUTO-COHÉRENCE : h* tel que Var_bulk(s_k) = (4−π)/π ≈ 0.273 (Wigner).
     Le bulk (bruit) est déterminé par POINT FIXE bulk/r̂/h.
  4. RANG : r̂ = max{k ≤ rmax : s_k > τ}, τ = quantile Wigner(1−α1) (≈3).
  5. CHARGES ĉ_j = ‖P_V̂ e_j‖² = Σ_{l≤r̂} V̂[j,l]² (somme = r̂, conservation de masse).
  6. FEATURES : loi nulle Beta(r̂/2,(p_eff−r̂)/2) ; sélection par Benjamini–Hochberg (FDR α2) ou seuil τ_c.

Deux décisions, deux lois nulles UNIVERSELLES (Wigner pour le rang, Beta pour les charges),
deux seuils = quantiles de ces lois. Seul paramètre estimé : h* (auto-cohérence).

Sorties (results_spectre/<config>_coulomb/) :
  roc.png, eigen_cdf.png (spectre + CDF + Ñ_h), calibration.png (Var(h), point fixe),
  unfolding.png (ξ_k, espacements s_k + τ + r̂), wigner.png (bulk vs Wigner),
  charges.png (charges + loi Beta nulle + seuil/BH), top_features.png + selected_features.txt,
  analyse_coulomb.pdf + summary.txt (h*, τ, τ_c, Var_bulk, r̂, n_sel, AUC, FDP).
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, beta as beta_dist
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from joblib import Parallel, delayed

WIG_VAR = (4 - np.pi) / np.pi                       # ≈ 0.2732 — variance de la loi de Wigner (β=1)

# ───────────────────────── arguments ─────────────────────────
ap = argparse.ArgumentParser(description="Spectre — gaz de Coulomb / unfolding")
ap.add_argument("--n", type=int, default=300); ap.add_argument("--p", type=int, default=500)
ap.add_argument("--k", type=int, default=5, help="nb de vraies features (0..k-1)")
ap.add_argument("--signal", type=float, default=3.0); ap.add_argument("--rho", type=float, default=0.0)
ap.add_argument("--block-size", "--block_size", type=int, default=5, dest="block_size")
ap.add_argument("--subtypes", type=int, default=0, help="0=off ; R>0 : mixture à R sous-types (K=R·block_size)")
ap.add_argument("--B", type=int, default=500); ap.add_argument("--n-seeds", "--n_seeds", type=int, default=1, dest="n_seeds")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rmax", type=int, default=1000, help="borne sup du rang r̂ (on cherche r̂ dans les rmax premiers espacements)")
ap.add_argument("--alpha1", type=float, default=1e-3, help="niveau Wigner -> seuil τ des espacements (τ=√(−(4/π)ln α1))")
ap.add_argument("--tau", type=float, default=0.0, help="override direct du seuil τ des espacements (0 = calculé depuis α1)")
ap.add_argument("--alpha2", type=float, default=0.10, help="niveau (FDR pour BH / type-I pour le seuil) des charges")
ap.add_argument("--n-h", "--n_h", type=int, default=50, dest="n_h", help="taille de la grille de h")
ap.add_argument("--fp-iter", "--fp_iter", type=int, default=8, dest="fp_iter", help="itérations max du point fixe bulk/r̂/h")
ap.add_argument("--n-C", "--n_C", type=int, default=10, dest="n_C")
ap.add_argument("--C-min", "--C_min", type=float, default=0.01, dest="C_min")
ap.add_argument("--C-max", "--C_max", type=float, default=1.0, dest="C_max")
ap.add_argument("--cv-folds", "--cv_folds", type=int, default=3, dest="cv_folds")
LAB = {
    "covid":             dict(loader="covid", path="COVID-19",    omics=["Proteomics"]),
    "SSI_Proteomics":    dict(loader="ssi",   path="Biobank SSI", omics=["Proteomics"]),
    "SSI_CyTOF":         dict(loader="ssi",   path="Biobank SSI", omics=["CyTOF"]),
    "SSI_EarlyFusion":   dict(loader="ssi",   path="Biobank SSI", omics=["CyTOF", "Proteomics"]),
    "CFRNA":             dict(loader="cfrna", path="CFRNA",       omics=["CFRNA"]),
    "Dream_Taxonomy":    dict(loader="dream", path="Dream",       omics=["Taxonomy"]),
    "Dream_Phylotype":   dict(loader="dream", path="Dream",       omics=["Phylotype"]),
    "Dream_EarlyFusion": dict(loader="dream", path="Dream",       omics=["Phylotype", "Taxonomy"]),
}
ap.add_argument("--data", choices=["synthetic"] + list(LAB), default="synthetic")
ap.add_argument("--data-dir", "--data_dir", dest="data_dir", default="../benchmark_v1_v2/data")
ap.add_argument("--prefilter", type=int, default=0)
args = ap.parse_args()

DATA = args.data
HAS_TRUTH = (DATA == "synthetic")
C_GRID = np.logspace(np.log10(args.C_min), np.log10(args.C_max), args.n_C)
TAU = args.tau if args.tau > 0 else float(np.sqrt(-(4.0/np.pi) * np.log(args.alpha1)))   # τ direct, ou quantile Wigner(1−α1)
_REAL = None; HAS_VAL = True

if DATA == "synthetic":
    P = args.p; FEATURE_NAMES = None
    if args.subtypes > 0:
        KTRUE = args.subtypes * args.block_size
        TAG = f"n{args.n}_p{args.p}_sub{args.subtypes}x{args.block_size}_sig{args.signal:g}_rho{args.rho:g}_B{args.B}_coulomb"
    else:
        KTRUE = args.k
        TAG = f"n{args.n}_p{args.p}_k{args.k}_sig{args.signal:g}_rho{args.rho:g}_B{args.B}_coulomb"
else:
    import sys, pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from stabl.data import load_covid_19, load_ssi, load_cfrna, load_dream
    from sklearn.preprocessing import StandardScaler
    LOADERS = {"covid": load_covid_19, "ssi": load_ssi, "cfrna": load_cfrna, "dream": load_dream}
    cfg = LAB[DATA]; omics = cfg["omics"]; multi = len(omics) > 1
    ddir = args.data_dir if os.path.isabs(args.data_dir) else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_dir)
    tr_d, val_d, ytr_s, yval_s = LOADERS[cfg["loader"]](os.path.join(ddir, cfg["path"]))[:4]
    HAS_VAL = val_d is not None
    def _prep(Xdf, idx, cols=None):
        X = Xdf.loc[idx]
        if cols is not None: X = X[cols]
        return X.apply(pd.to_numeric, errors="coerce").fillna(X.apply(pd.to_numeric, errors="coerce").mean())
    def _topk(Xdf):
        if args.prefilter and 0 < args.prefilter < Xdf.shape[1]:
            return Xdf[Xdf.var(axis=0).nlargest(args.prefilter).index]
        return Xdf
    common = pd.Index(ytr_s.index)
    for om in omics: common = common.intersection(tr_d[om].index)
    common = common.unique()
    kept = {om: list(_topk(_prep(tr_d[om], common)).columns) for om in omics}
    if HAS_VAL:
        cval = pd.Index(yval_s.index)
        for om in omics: cval = cval.intersection(val_d[om].index)
        cval = cval.unique()
        for om in omics: kept[om] = [c for c in kept[om] if c in val_d[om].columns]
    def _stack(dct, idx):
        mats, names = [], []
        for om in omics:
            mats.append(_prep(dct[om], idx, kept[om]).to_numpy(float))
            names += [f"{om}:{c}" for c in kept[om]] if multi else list(kept[om])
        return np.hstack(mats), names
    Xtr_r, FEATURE_NAMES = _stack(tr_d, common); ytr_a = ytr_s.loc[common].astype(int).to_numpy()
    sc = StandardScaler().fit(Xtr_r)
    if HAS_VAL:
        Xval_r, _ = _stack(val_d, cval); yval_a = yval_s.loc[cval].astype(int).to_numpy()
        _REAL = (sc.transform(Xtr_r), ytr_a, sc.transform(Xval_r), yval_a)
    else:
        _REAL = (sc.transform(Xtr_r), ytr_a, None, None)
    P = _REAL[0].shape[1]; KTRUE = 0
    TAG = f"{DATA}_B{args.B}" + (f"_pf{args.prefilter}" if args.prefilter else "") + "_coulomb"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_spectre", TAG)
os.makedirs(OUT, exist_ok=True)
if DATA == "synthetic":
    print(f"Spectre/Coulomb : n={args.n} p={args.p} K={KTRUE} signal={args.signal} rho={args.rho} "
          f"B={args.B}  τ={TAU:.2f} (α1={args.alpha1:g})  α2={args.alpha2:g}")
else:
    vtxt = (f"validation {_REAL[2].shape}" if HAS_VAL else "AUC par CV 5-fold (pas de validation)")
    print(f"Spectre/Coulomb [{DATA}] : train {_REAL[0].shape} (y={np.bincount(_REAL[1]).tolist()}), {vtxt}, "
          f"B={args.B}  τ={TAU:.2f}  α2={args.alpha2:g}")

# ───────────────────────── données ─────────────────────────
def generate(seed):
    if DATA != "synthetic":
        return _REAL
    rng = np.random.default_rng(seed)
    n = args.n + 2000; p = args.p; bs = args.block_size
    if args.rho > 0:
        X = np.empty((n, p)); b0 = 0
        while b0 < p:
            w = min(bs, p - b0); f = rng.standard_normal((n, 1))
            X[:, b0:b0+w] = np.sqrt(args.rho)*f + np.sqrt(1-args.rho)*rng.standard_normal((n, w)); b0 += w
    else:
        X = rng.standard_normal((n, p))
    if args.subtypes > 0:
        R = args.subtypes; beta_b = args.signal / np.sqrt(bs); sub = rng.integers(0, R, size=n); yl = np.zeros(n)
        for a in range(R):
            mask = sub == a
            yl[mask] = X[mask][:, a*bs:(a+1)*bs] @ np.full(bs, beta_b)
        yl += rng.standard_normal(n)
    else:
        K = args.k; beta = np.zeros(p); beta[:K] = args.signal / np.sqrt(K)
        yl = X[:, :K] @ beta[:K] + rng.standard_normal(n)
    y = (yl > 0).astype(int)
    return X[:args.n], y[:args.n], X[args.n:], y[args.n:]

# ───────── base learner : lasso L1, C réglé par CV interne par moitié ─────────
def lasso_select(Xs, ys):
    if len(np.unique(ys)) < 2:
        return np.zeros(Xs.shape[1])
    cv = min(args.cv_folds, int(np.bincount(ys).min()))
    common = dict(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=5000)
    if cv < 2:
        clf = LogisticRegression(C=float(np.median(C_GRID)), **common)
    else:
        clf = LogisticRegressionCV(Cs=C_GRID, cv=cv, scoring="roc_auc", **common)
    clf.fit(Xs, ys)
    return (np.abs(clf.coef_[0]) > 1e-8).astype(float)

# ───────── opérateur croisé M̂ par double sous-échantillonnage disjoint ─────────
def build_cross(X, y, seed):
    n, p = X.shape; m = n // 3
    def one_boot(b):
        rng = np.random.default_rng([seed, b])
        perm = rng.permutation(n); A, Bs = perm[:m], perm[m:2*m]
        return lasso_select(X[A], y[A]), lasso_select(X[Bs], y[Bs])
    res = Parallel(n_jobs=-1)(delayed(one_boot)(b) for b in range(args.B))
    SA = np.array([r[0] for r in res]); SB = np.array([r[1] for r in res])
    Mcross = (SA.T @ SB + SB.T @ SA) / (2*args.B)        # M̂ -> ppᵀ (diag p_j²)
    psel = (SA.sum(0) + SB.sum(0)) / (2*args.B)          # p̂_j (fréquence de sélection)
    return Mcross, psel

# ───────────────────────── unfolding (gaz de Coulomb) ─────────────────────────
def unfold_spacings(lam, h):
    """ξ_k = Ñ_h(λ_k) = mean_i Φ((λ_k−λ_i)/h) ; s_k = peff(ξ_k − ξ_{k+1})."""
    diff = (lam[:, None] - lam[None, :]) / h
    xi = norm.cdf(diff).mean(axis=1)                     # croissant en λ ; lam décroissant -> xi décroissant
    s = len(lam) * (xi[:-1] - xi[1:])                    # s_k ≥ 0, longueur peff−1
    return s, xi

def calibrate_h(lam, h_grid, r_hat):
    """h* = argmin_h |Var_{bulk}(s_k(h)) − 0.273| ; bulk = espacements d'indice > r̂."""
    best_h, best_d, vars_ = h_grid[0], np.inf, []
    for h in h_grid:
        s, _ = unfold_spacings(lam, h)
        signal_idx = np.where(s > TAU)[0]

        mask = np.ones(len(s), dtype=bool)
        mask[signal_idx] = False

        bulk = s[mask]                                # exclut les r̂ premiers espacements (sommet signal)
        v = float(np.var(bulk)) if len(bulk) > 2 else np.nan
        vars_.append(v)
        d = abs(v - WIG_VAR) if np.isfinite(v) else np.inf
        if d < best_d: best_d, best_h = d, h
    return best_h, np.array(vars_)

def fixed_point(lam, h_grid):
    """Point fixe bulk/r̂/h : calibrer h sur le bulk, lire r̂, exclure le sommet, recommencer."""
    r_hat, traj = 0, []
    h_star, vars_ = calibrate_h(lam, h_grid, r_hat)
    s, xi = unfold_spacings(lam, h_star)
    for _ in range(args.fp_iter):
        rm = min(args.rmax, len(s))
        signal_idx = np.where(s[:rm] > TAU)[0]
        new_r = len(signal_idx)
        traj.append((r_hat, float(h_star), new_r))
        if new_r == r_hat: break
        r_hat = new_r
        h_star, vars_ = calibrate_h(lam, h_grid, r_hat)
        s, xi = unfold_spacings(lam, h_star)
    return r_hat, signal_idx, float(h_star), s, xi, vars_, traj

def bh_select(pvals, alpha):
    """Benjamini–Hochberg : renvoie les indices retenus (FDR ≤ alpha)."""
    m = len(pvals)
    if m == 0: return np.array([], dtype=int)
    order = np.argsort(pvals); thresh = alpha * np.arange(1, m+1) / m
    passed = pvals[order] <= thresh
    if not passed.any(): return np.array([], dtype=int)
    cutoff = pvals[order][np.where(passed)[0].max()]
    return np.where(pvals <= cutoff)[0]

# ───────────────────────── 1 run complet (1 seed) ─────────────────────────
def run_one(seed):
    Xtr, ytr, Xte, yte = generate(seed)
    Mcross, psel = build_cross(Xtr, ytr, seed)
    ix_eff = np.where(psel > 0)[0]; peff = len(ix_eff)        # features sélectionnées au moins une fois
    Msub = Mcross[np.ix_(ix_eff, ix_eff)]
    w, U = np.linalg.eigh(Msub); lam = w[::-1]; U = U[:, ::-1]   # valeurs propres décroissantes
    # GAZ = valeurs propres NON nulles : les zéros (déficit de rang si 2B<p_eff) ne sont pas des particules
    tol = 1e-8 * np.abs(lam).max()
    lam_g = lam[np.abs(lam) > tol]; ngas = len(lam_g)        # gaz, décroissant ; λ_1 = lam_g[0]
    bulkv = lam_g[1:] if ngas > 3 else lam_g                 # échelle de h : bulk hors λ_1 (outlier)
    micro = max((np.percentile(bulkv, 95) - np.percentile(bulkv, 5)) / max(len(bulkv), 1), 1e-12)
    h_grid = micro * np.logspace(-0.3, 3.0, args.n_h)        # ~0.5× à ~1000× l'espacement microscopique
    r_hat, signal_idx, h_star, s, xi, vars_, traj = fixed_point(lam_g, h_grid)
    bulk_idx = np.arange(r_hat, len(s))                      # espacements du bulk (après calibration)
    var_bulk = float(np.var(s[bulk_idx])) if len(bulk_idx) > 2 else float("nan")
    # charges sur le sous-espace signal
    charges = (U[:, signal_idx]**2).sum(axis=1) if len(signal_idx) else np.zeros(peff)
    # loi nulle Beta + sélection (BH sur p-valeurs Beta) + seuil τ_c
    if r_hat >= 1 and peff - r_hat > 0:
        a_b, b_b = r_hat/2.0, (peff - r_hat)/2.0
        tau_c = float(beta_dist.ppf(1 - args.alpha2, a_b, b_b))
        pvals = beta_dist.sf(np.clip(charges, 0, 1), a_b, b_b)
        sel_eff = bh_select(pvals, args.alpha2)
    else:
        a_b = b_b = tau_c = np.nan; pvals = np.ones(peff); sel_eff = np.array([], dtype=int)
    sel = np.sort(ix_eff[sel_eff]) if len(sel_eff) else np.array([], dtype=int)
    ns = len(sel)
    nt = int((sel < KTRUE).sum()) if HAS_TRUTH else 0
    fdp = (1.0 - nt/max(1, ns)) if HAS_TRUTH else float("nan")
    # AUC
    yeval = yte if HAS_VAL else ytr
    if ns > 0:
        clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=5000)
        if HAS_VAL:
            clf.fit(Xtr[:, sel], ytr); prob = clf.predict_proba(Xte[:, sel])[:, 1]
        else:
            cvk = min(5, int(np.bincount(ytr).min()))
            prob = cross_val_predict(clf, Xtr[:, sel], ytr, cv=StratifiedKFold(cvk, shuffle=True, random_state=seed),
                                     method="predict_proba")[:, 1]
    else:
        prob = np.full(len(yeval), 0.5)
    auc = roc_auc_score(yeval, prob); fpr, tpr, _ = roc_curve(yeval, prob)
    return dict(lam=lam, lam_g=lam_g, ngas=ngas, U=U, peff=peff, ix_eff=ix_eff, h_star=h_star, h_grid=h_grid, vars_=vars_,
                s=s, xi=xi, r_hat=r_hat, var_bulk=var_bulk, traj=traj, charges=charges, pvals=pvals,
                a_b=a_b, b_b=b_b, tau_c=tau_c, sel_eff=sel_eff, sel=sel, n_sel=ns, n_true=nt, fdp=fdp,
                auc=auc, fpr=fpr, tpr=tpr, psel=psel)

# ───────────────────────── boucle seeds ─────────────────────────
SEEDS = [args.seed + i for i in range(args.n_seeds)]
RES = [run_one(s) for s in SEEDS]
for s, r in zip(SEEDS, RES):
    truth = f"(vraies={r['n_true']}/{KTRUE})  FDP={r['fdp']:.2f}  " if HAS_TRUTH else ""
    print(f"  seed {s}: r̂={r['r_hat']}  h*={r['h_star']:.4g}  Var_bulk={r['var_bulk']:.3f}  "
          f"n_sel={r['n_sel']}  {truth}AUC={r['auc']:.3f}")
AUC_m = float(np.mean([r["auc"] for r in RES])); AUC_s = float(np.std([r["auc"] for r in RES]))
FDP_m = float(np.mean([r["fdp"] for r in RES])) if HAS_TRUTH else float("nan")
RHAT_m = float(np.mean([r["r_hat"] for r in RES])); NSEL_m = float(np.mean([r["n_sel"] for r in RES]))
fdp_txt = f"FDP moyen = {FDP_m:.2f} | " if HAS_TRUTH else "FDP = N/A | "
print(f"\n=> {'AUC val' if HAS_VAL else 'AUC CV'} = {AUC_m:.3f} ± {AUC_s:.3f} | {fdp_txt}"
      f"r̂ moyen = {RHAT_m:.1f} | n_sel moyen = {NSEL_m:.1f} | τ={TAU:.2f}")

# ───────────────────────── PLOTS ─────────────────────────
r0 = RES[0]; lam = r0["lam"]; lam_g = r0["lam_g"]; ngas = r0["ngas"]; peff = r0["peff"]; s = r0["s"]; xi = r0["xi"]
r_hat = r0["r_hat"]; h_star = r0["h_star"]; charges = r0["charges"]
def wigner_pdf(x): return (np.pi/2)*x*np.exp(-(np.pi/4)*x**2)

# (1) ROC
grid = np.linspace(0, 1, 200)
tprs = np.array([np.interp(grid, r["fpr"], r["tpr"]) for r in RES]); tprs[:, 0] = 0.0
fig_roc, ax = plt.subplots(figsize=(5.2, 5))
for r in RES: ax.plot(r["fpr"], r["tpr"], color="#BBBBBB", lw=0.8, alpha=0.6)
ax.plot(grid, tprs.mean(0), color="#1f77b4", lw=2.4, label=f"ROC ({'val' if HAS_VAL else 'CV'}, AUC={AUC_m:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title(f"Coulomb — {DATA}\nr̂={r_hat}  n_sel={NSEL_m:.0f}" + (f"  FDP={FDP_m:.2f}" if HAS_TRUTH else ""), fontsize=10)
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)
fig_roc.tight_layout(); fig_roc.savefig(os.path.join(OUT, "roc.png"), dpi=140)

# (2) Spectre + CDF empirique + potentiel lissé Ñ_h*
ns_e = min(peff, 60)
fig_cdf, (c1, c2) = plt.subplots(1, 2, figsize=(13, 4.6))
c1.bar(np.arange(1, ns_e+1), lam[:ns_e], width=0.8,
       color=["#D62728" if i < r_hat else "#9AA0A6" for i in range(ns_e)])
c1.axhline(0, color="black", lw=0.6); c1.set_xlabel("indice i"); c1.set_ylabel("valeur propre λ_i de M̂")
c1.set_title(f"spectre de M̂ (croisé) — r̂={r_hat} (rouge)", fontsize=10); c1.grid(alpha=0.25, axis="y")
lam_sorted = np.sort(lam_g); Nemp = np.arange(1, ngas+1)/ngas
gridλ = np.linspace(lam_g.min(), lam_g.max(), 400)
Nsmooth = np.array([norm.cdf((g - lam_g)/h_star).mean() for g in gridλ])
c2.step(lam_sorted, Nemp, where="post", color="#9AA0A6", lw=1.2, label="N(λ) empirique (gaz)")
c2.plot(gridλ, Nsmooth, color="#1f77b4", lw=2, label=f"Ñ_h(λ) lissé (h*={h_star:.3g})")
c2.set_xlabel("λ"); c2.set_ylabel("CDF spectrale"); c2.set_title("fonction de répartition du spectre + potentiel lissé", fontsize=10)
c2.legend(fontsize=8); c2.grid(alpha=0.25)
fig_cdf.suptitle("Coulomb — spectre de M̂ et CDF spectrale (unfolding)", fontsize=11)
fig_cdf.tight_layout(rect=[0, 0, 1, 0.95]); fig_cdf.savefig(os.path.join(OUT, "eigen_cdf.png"), dpi=140)

# (3) Calibration de h : Var_bulk(h) -> 0.273, + trajectoire du point fixe
fig_cal, (k1, k2) = plt.subplots(1, 2, figsize=(13, 4.6))
k1.plot(r0["h_grid"], r0["vars_"], color="#1f77b4", lw=2, marker="o", ms=2.5)
k1.axhline(WIG_VAR, color="#D62728", ls="--", lw=1.4, label=f"Wigner Var={WIG_VAR:.3f}")
k1.axvline(h_star, color="black", ls=":", lw=1.4, label=f"h*={h_star:.3g}")
k1.set_xscale("log"); k1.set_xlabel("largeur h (log)"); k1.set_ylabel("Var_bulk(s_k)")
k1.set_title("calibration auto-cohérente de h", fontsize=10); k1.legend(fontsize=8); k1.grid(alpha=0.25)
tr = np.array(r0["traj"])
k2.plot(range(1, len(tr)+1), tr[:, 2], color="#2ca02c", lw=2, marker="o", label="r̂ (itération)")
k2.set_xlabel("itération du point fixe"); k2.set_ylabel("r̂"); k2.set_ylim(-0.5, max(2, tr[:, 2].max()+1))
k2.set_title(f"point fixe bulk/r̂/h — converge à r̂={r_hat}", fontsize=10); k2.legend(fontsize=8); k2.grid(alpha=0.25)
fig_cal.suptitle("Coulomb — calibration de h (Var→0.273) et point fixe du bulk", fontsize=11)
fig_cal.tight_layout(rect=[0, 0, 1, 0.95]); fig_cal.savefig(os.path.join(OUT, "calibration.png"), dpi=140)

# (4) Unfolding : valeurs dépliées ξ_k et espacements s_k avec seuil τ
ns_u = min(len(s), 60)
fig_unf, (u1, u2) = plt.subplots(1, 2, figsize=(13, 4.6))
u1.plot(np.arange(1, min(ngas, 60)+1), xi[:min(ngas, 60)], color="#1f77b4", lw=1.5, marker="o", ms=2.5)
u1.set_xlabel("indice k"); u1.set_ylabel("ξ_k = Ñ_h(λ_k) (déplié)")
u1.set_title("valeurs propres dépliées", fontsize=10); u1.grid(alpha=0.25)
u2.bar(np.arange(1, ns_u+1), s[:ns_u], width=0.8,
       color=["#D62728" if i < r_hat else "#9AA0A6" for i in range(ns_u)])
u2.axhline(TAU, color="black", ls="--", lw=1.4, label=f"τ={TAU:.2f} (Wigner 1−α1)")
u2.axhline(1.0, color="#2ca02c", ls=":", lw=1, label="moyenne bulk ≈ 1")
u2.set_xlabel("indice k"); u2.set_ylabel("espacement renormalisé s_k")
u2.set_title(f"espacements dépliés — r̂=max{{k:s_k>τ}}={r_hat}  (s_1={s[0]:.2f})", fontsize=10)
u2.legend(fontsize=8); u2.grid(alpha=0.25, axis="y")
fig_unf.suptitle("Coulomb — dépliement et espacements renormalisés (détection du rang)", fontsize=11)
fig_unf.tight_layout(rect=[0, 0, 1, 0.95]); fig_unf.savefig(os.path.join(OUT, "unfolding.png"), dpi=140)

# (5) Bulk déplié vs loi de Wigner (validation de l'hypothèse)
bulk_s = s[r_hat:]
fig_wig, wg = plt.subplots(figsize=(6.5, 5))
if len(bulk_s) > 2:
    wg.hist(bulk_s, bins=min(30, max(8, len(bulk_s)//5)), density=True, color="#9AA0A6", alpha=0.8, label="bulk déplié")
xx = np.linspace(0, max(3.5, float(bulk_s.max()) if len(bulk_s) else 3.5), 200)
wg.plot(xx, wigner_pdf(xx), color="#D62728", lw=2.2, label="loi de Wigner P(s)")
wg.axvline(TAU, color="black", ls="--", lw=1.2, label=f"τ={TAU:.2f}")
wg.set_xlabel("espacement s"); wg.set_ylabel("densité")
wg.set_title(f"bulk déplié vs Wigner — Var_bulk={r0['var_bulk']:.3f} (cible {WIG_VAR:.3f})", fontsize=10)
wg.legend(fontsize=9); wg.grid(alpha=0.25)
fig_wig.tight_layout(); fig_wig.savefig(os.path.join(OUT, "wigner.png"), dpi=140)

# (6) Charges : histogramme + loi Beta nulle + seuil ; charges triées avec sélection
sel_eff_set = set(r0["sel_eff"].tolist())
fig_ch, (h1, h2) = plt.subplots(1, 2, figsize=(13, 4.6))
h1.hist(charges, bins=50, density=True, color="#9AA0A6", alpha=0.85, label="charges ĉ_j")
if np.isfinite(r0["a_b"]):
    cx = np.linspace(1e-6, max(charges.max(), r0["tau_c"]*1.5, 1e-3), 300)
    h1.plot(cx, beta_dist.pdf(cx, r0["a_b"], r0["b_b"]), color="#1f77b4", lw=2,
            label=f"Beta({r0['a_b']:.1f},{r0['b_b']:.0f}) nulle")
    h1.axvline(r0["tau_c"], color="#D62728", ls="--", lw=1.4, label=f"τ_c={r0['tau_c']:.3g}")
h1.set_xlabel("charge ĉ_j"); h1.set_ylabel("densité"); h1.set_yscale("log")
h1.set_title("charges vs loi nulle Beta", fontsize=10); h1.legend(fontsize=8); h1.grid(alpha=0.25)
order = np.argsort(charges)[::-1]; nb = min(80, peff)
cols = ["#D62728" if order[i] in sel_eff_set else "#C8C8C8" for i in range(nb)]
h2.bar(np.arange(1, nb+1), charges[order][:nb], color=cols, width=1.0)
if HAS_TRUTH:
    eff_true = [i for i in range(nb) if r0["ix_eff"][order[i]] < KTRUE]
    for i in eff_true:
        h2.plot(i+1, charges[order[i]] + charges.max()*0.03, marker="*", color="#FF7F0E", ms=8)
if np.isfinite(r0["tau_c"]): h2.axhline(r0["tau_c"], color="#D62728", ls="--", lw=1.2, label=f"τ_c={r0['tau_c']:.3g}")
h2.set_xlabel("rang (charge décroissante)"); h2.set_ylabel("charge ĉ_j")
h2.set_title(f"charges triées (rouge=sél BH, ★=vraie)  |S|={r0['n_sel']}", fontsize=10)
h2.legend(fontsize=8); h2.grid(alpha=0.25, axis="y")
fig_ch.suptitle("Coulomb — charges des features et loi nulle Beta (sélection BH)", fontsize=11)
fig_ch.tight_layout(rect=[0, 0, 1, 0.95]); fig_ch.savefig(os.path.join(OUT, "charges.png"), dpi=140)

# (7) Top features avec noms (labo)
fig_top = None
if FEATURE_NAMES is not None:
    topN = min(30, peff); top_idx = order[:topN]
    names = [FEATURE_NAMES[r0["ix_eff"][j]] for j in top_idx]; vals = charges[top_idx]
    cols_t = ["#D62728" if j in sel_eff_set else "#AAAAAA" for j in top_idx]
    fig_top, axt = plt.subplots(figsize=(8.5, max(5, topN*0.3)))
    ypos = np.arange(topN)[::-1]; axt.barh(ypos, vals, color=cols_t)
    axt.set_yticks(ypos); axt.set_yticklabels(names, fontsize=7); axt.set_xlabel("charge ĉ_j"); axt.grid(alpha=0.25, axis="x")
    axt.set_title(f"{DATA} — top {topN} features par charge (rouge=sél, r̂={r_hat}, |S|={r0['n_sel']})", fontsize=10)
    fig_top.tight_layout(); fig_top.savefig(os.path.join(OUT, "top_features.png"), dpi=140)
    with open(os.path.join(OUT, "selected_features.txt"), "w") as fh:
        for j in order:
            if j in sel_eff_set:
                fh.write(f"{FEATURE_NAMES[r0['ix_eff'][j]]}\t{charges[j]:.5f}\tp={r0['pvals'][j]:.2e}\n")

# PDF + summary
with PdfPages(os.path.join(OUT, "analyse_coulomb.pdf")) as pdf:
    for f in (fig_roc, fig_cdf, fig_cal, fig_unf, fig_wig, fig_ch):
        pdf.savefig(f)
    if fig_top is not None: pdf.savefig(fig_top)
with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write(f"Spectre/Coulomb — {TAG}\n")
    fh.write(f"{'AUC val' if HAS_VAL else 'AUC CV'} = {AUC_m:.4f} ± {AUC_s:.4f}\n")
    fh.write(f"FDP moyen   = {FDP_m:.4f}\n" if HAS_TRUTH else "FDP        = N/A\n")
    fh.write(f"r_hat       = {r_hat}\n")
    fh.write(f"h_star      = {h_star:.5g}\n")
    fh.write(f"Var_bulk    = {r0['var_bulk']:.4f}   (cible Wigner {WIG_VAR:.4f})\n")
    fh.write(f"tau (Wigner)= {TAU:.4f}   (alpha1={args.alpha1:g})\n")
    fh.write(f"tau_c (Beta)= {r0['tau_c']:.5g}   (alpha2={args.alpha2:g})\n")
    fh.write(f"p_eff       = {peff}\n")
    fh.write(f"n_sel       = {r0['n_sel']}\n")
    fh.write(f"s_1 (dominante dépliée) = {s[0]:.4f}   ({'> τ -> signal' if s[0] > TAU else '<= τ -> bord du gaz'})\n")
    fh.write(f"valeurs propres top 12 : " + " ".join(f"{v:.4f}" for v in lam[:12]) + "\n")
    fh.write(f"espacements s_k top 12 : " + " ".join(f"{v:.3f}" for v in s[:12]) + "\n")
    fh.write(f"point fixe (r̂_in, h, r̂_out) : " + " ; ".join(f"({a},{b:.3g},{c})" for a, b, c in r0["traj"]) + "\n")
print(f"\nFigures + analyse_coulomb.pdf + summary.txt dans : {OUT}")
