#!/usr/bin/env python3
"""
Spectre — Cross-Reproducibility Spectral Selection (Osman Rittano) — modèle M(γ)
================================================================================
Stability selection SANS features artificielles. À chaque bootstrap on tire deux
sous-échantillons DISJOINTS A et B (tiers 1 et tiers 2, tiers 3 inutilisé), on fit le
base learner (lasso logistique, C réglé par CV interne) sur chacun, et on score par CO-REPRODUCTIBILITÉ :

    cross  C = moyenne_b [ (s_A s_Bᵀ + s_B s_Aᵀ)/2 ]      (halves indépendants -> q_j² sur la diag)
    within W = moyenne_b [ (s_A s_Aᵀ + s_B s_Bᵀ)/2 ]      (porte la structure de groupe)
    mixte  M(γ) = (1-γ) C + γ W ,   γ ∈ [0,1]

On regarde le plus grand GAP SPECTRAL de M(γ), on choisit γ* qui le maximise, puis on
sélectionne par MASSE SPECTRALE (loads = ||P_V e_j||², somme = r̂ ; on cumule jusqu'à r̂(1-ρ)).
(Le modèle de Hadamard — un seul opérateur Ĥ=Q̂∘² sans γ — est dans spectre_hadamard.py.)

Sorties (dossier results_spectre/<config>/) :
  - roc.png            : ROC AUC moyenne (sur les seeds)
  - spectral_gap.png   : gap spectral G(γ) en fonction de γ (+ γ*) et r̂(γ)
  - matrices.png       : la « gueule » de M(γ) pour plusieurs γ (heatmaps couleur)
  - eigenvalues.png, selection.png, matrices_CW.png, top_features.png + selected_features.txt
  - analyse_spectre.pdf + summary.txt (FDP, AUC, γ*, r̂, n_sel)
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from joblib import Parallel, delayed

# ───────────────────────── arguments ─────────────────────────
ap = argparse.ArgumentParser(description="Spectre — cross-reproducibility spectral selection")
ap.add_argument("--n", type=int, default=300, help="taille du train")
ap.add_argument("--p", type=int, default=500, help="nb de features")
ap.add_argument("--k", type=int, default=5, help="nb de vraies features (0..k-1)")
ap.add_argument("--signal", type=float, default=3.0, help="force du signal (beta = signal/sqrt(k))")
ap.add_argument("--rho", type=float, default=0.0, help="corrélation intra-bloc (0 = indépendant ; >0 -> groupes)")
ap.add_argument("--block-size", "--block_size", type=int, default=5, dest="block_size", help="taille des blocs corrélés")
ap.add_argument("--subtypes", type=int, default=0,
                help="0=off. Si R>0 : mixture à R sous-types, y_i piloté par le MODULE du sous-type de i "
                     "(R blocs indépendants, K=R·block_size vraies). Régime où r̂=R doit émerger.")
ap.add_argument("--B", type=int, default=500, help="nb de bootstraps (double sous-échantillonnage)")
ap.add_argument("--n-seeds", "--n_seeds", type=int, default=1, dest="n_seeds", help="nb de seeds à moyenner")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rmax", type=int, default=10, help="borne sup du nb de groupes")
ap.add_argument("--log-gap", "--log_gap", dest="log_gap", action="store_true",
                help="choisir r̂ par le plus grand gap en échelle LOG : log(λ_r) − log(λ_{r+1}) = log(λ_r/λ_{r+1}) "
                     "(sensible aux cliffs multiplicatifs vers le plancher de bruit ; sinon gap absolu).")
ap.add_argument("--n-gamma", "--n_gamma", type=int, default=21, dest="n_gamma", help="taille de la grille de γ")
ap.add_argument("--n-C", "--n_C", type=int, default=10, dest="n_C", help="taille de la grille de C explorée PAR LA CV INTERNE du base learner")
ap.add_argument("--C-min", "--C_min", type=float, default=0.01, dest="C_min", help="C min de la grille CV interne (régularisation forte)")
ap.add_argument("--C-max", "--C_max", type=float, default=1.0, dest="C_max", help="C max de la grille CV interne (régularisation faible)")
ap.add_argument("--cv-folds", "--cv_folds", type=int, default=3, dest="cv_folds", help="nb de folds de la CV interne du base learner")
ap.add_argument("--base", choices=["lasso", "elasticnet"], default="lasso",
                help="base learner : lasso (L1, liblinear) ou elasticnet (L1+L2, saga). C tuné par CV interne dans les deux cas.")
ap.add_argument("--l1-ratio", "--l1_ratio", type=float, default=0.5, dest="l1_ratio",
                help="elasticnet seulement : split L1/L2 FIXÉ (1=lasso pur, 0=ridge pur). λ_L1=l1_ratio/C, λ_L2=(1-l1_ratio)/C.")
# registre des datasets labo BINAIRES (mono-omique "single" + early fusion = concat des omiques)
LAB = {
    "covid":             dict(loader="covid", path="COVID-19",    omics=["Proteomics"]),  # a un set de validation
    "SSI_Proteomics":    dict(loader="ssi",   path="Biobank SSI", omics=["Proteomics"]),
    "SSI_CyTOF":         dict(loader="ssi",   path="Biobank SSI", omics=["CyTOF"]),
    "SSI_EarlyFusion":   dict(loader="ssi",   path="Biobank SSI", omics=["CyTOF", "Proteomics"]),
    "CFRNA":             dict(loader="cfrna", path="CFRNA",       omics=["CFRNA"]),
    "Dream_Taxonomy":    dict(loader="dream", path="Dream",       omics=["Taxonomy"]),
    "Dream_Phylotype":   dict(loader="dream", path="Dream",       omics=["Phylotype"]),
    "Dream_EarlyFusion": dict(loader="dream", path="Dream",       omics=["Phylotype", "Taxonomy"]),
}
ap.add_argument("--data", choices=["synthetic"] + list(LAB), default="synthetic",
                help="synthétique (vérité terrain) ou un dataset labo binaire (cf. registre LAB)")
ap.add_argument("--data-dir", "--data_dir", dest="data_dir", default="../benchmark_v1_v2/data",
                help="racine des données labo")
ap.add_argument("--prefilter", type=int, default=0,
                help="pré-filtre non-supervisé : garde les top-K features par variance PAR OMIQUE (0 = off)")
ap.add_argument("--nested-cv", "--nested_cv", dest="nested_cv", action="store_true",
                help="AUC sans fuite : refait TOUTE la sélection Spectre + le scaling dans chaque fold (datasets sans validation)")
args = ap.parse_args()

DATA = args.data
HAS_TRUTH = (DATA == "synthetic")          # vérité terrain (vraies = 0..k-1) seulement en synthétique
GAMMA = np.linspace(0.0, 1.0, args.n_gamma)
C_GRID = np.logspace(np.log10(args.C_min), np.log10(args.C_max), args.n_C)   # grille CV INTERNE du base learner

_REAL = None; _RAW_TR = None; HAS_VAL = True   # HAS_VAL : dispose-t-on d'un set de validation held-out ?
                                               # _RAW_TR : train BRUT (non scalé) pour la nested-CV (scaling refait par fold)
if DATA == "synthetic":
    P = args.p; FEATURE_NAMES = None
    if args.subtypes > 0:                  # mixture : K = R modules × block_size, vraies = 0..K-1
        KTRUE = args.subtypes * args.block_size
        TAG = f"n{args.n}_p{args.p}_sub{args.subtypes}x{args.block_size}_sig{args.signal:g}_rho{args.rho:g}_B{args.B}"
    else:
        KTRUE = args.k
        TAG = f"n{args.n}_p{args.p}_k{args.k}_sig{args.signal:g}_rho{args.rho:g}_B{args.B}"
else:                                      # chargement labo réel (une fois ; le seed ne touche que les bootstraps)
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

    def _prep(Xdf, idx, cols=None):        # lignes idx (+ colonnes cols) -> numérique -> fillna(moyenne)
        X = Xdf.loc[idx]
        if cols is not None:
            X = X[cols]
        X = X.apply(pd.to_numeric, errors="coerce")
        return X.fillna(X.mean())

    def _topk(Xdf):                        # pré-filtre top-K variance (faisabilité quand p≫n, ex. CFRNA)
        if args.prefilter and 0 < args.prefilter < Xdf.shape[1]:
            return Xdf[Xdf.var(axis=0).nlargest(args.prefilter).index]
        return Xdf

    common = pd.Index(ytr_s.index)
    for om in omics:
        common = common.intersection(tr_d[om].index)
    common = common.unique()
    kept = {om: list(_topk(_prep(tr_d[om], common)).columns) for om in omics}
    if HAS_VAL:                            # COVID : restreindre aux colonnes présentes en validation
        cval = pd.Index(yval_s.index)
        for om in omics:
            cval = cval.intersection(val_d[om].index)
        cval = cval.unique()
        for om in omics:
            kept[om] = [c for c in kept[om] if c in val_d[om].columns]

    def _stack(dct, idx):
        mats, names = [], []
        for om in omics:
            mats.append(_prep(dct[om], idx, kept[om]).to_numpy(float))
            names += [f"{om}:{c}" for c in kept[om]] if multi else list(kept[om])
        return np.hstack(mats), names

    Xtr_r, FEATURE_NAMES = _stack(tr_d, common); ytr_a = ytr_s.loc[common].astype(int).to_numpy()
    _RAW_TR = (Xtr_r, ytr_a)                    # brut, non scalé -> scaling refait par fold en nested-CV
    sc = StandardScaler().fit(Xtr_r)
    if HAS_VAL:
        Xval_r, _ = _stack(val_d, cval); yval_a = yval_s.loc[cval].astype(int).to_numpy()
        _REAL = (sc.transform(Xtr_r), ytr_a, sc.transform(Xval_r), yval_a)
    else:
        _REAL = (sc.transform(Xtr_r), ytr_a, None, None)
    P = _REAL[0].shape[1]; KTRUE = 0
    TAG = f"{DATA}_B{args.B}" + (f"_pf{args.prefilter}" if args.prefilter else "")
if args.base != "lasso":                       # éviter d'écraser les résultats lasso
    TAG += f"_{args.base}L1r{args.l1_ratio:g}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_spectre", TAG)
os.makedirs(OUT, exist_ok=True)
if DATA == "synthetic":
    print(f"Spectre M(γ) : n={args.n} p={args.p} k={args.k} signal={args.signal} rho={args.rho} "
          f"B={args.B} n_seeds={args.n_seeds}  (γ-grille {args.n_gamma} pts)")
else:
    vtxt = (f"validation {_REAL[2].shape} (y={np.bincount(_REAL[3]).tolist()})" if HAS_VAL
            else "AUC par CV 5-fold (pas de set de validation)")
    print(f"Spectre [{DATA}] : train {_REAL[0].shape} (y={np.bincount(_REAL[1]).tolist()}), "
          f"{vtxt}, B={args.B}, n_seeds={args.n_seeds}")

# ───────────────────────── génération des données ─────────────────────────
def generate(seed):
    """Synthétique : gaussien + signal + bruit (train + test held-out).
    Labo : train (+ validation si dispo) réels (fixes ; le seed ne change que les bootstraps)."""
    if DATA != "synthetic":
        return _REAL
    rng = np.random.default_rng(seed)
    n = args.n + 2000; p = args.p; bs = args.block_size
    # X : blocs corrélés (facteur latent INDÉPENDANT par bloc) si rho>0, sinon gaussien
    if args.rho > 0:
        X = np.empty((n, p)); b0 = 0
        while b0 < p:
            w = min(bs, p - b0); f = rng.standard_normal((n, 1))
            X[:, b0:b0+w] = np.sqrt(args.rho)*f + np.sqrt(1-args.rho)*rng.standard_normal((n, w))
            b0 += w
    else:
        X = rng.standard_normal((n, p))
    if args.subtypes > 0:
        # MIXTURE : chaque patient a un sous-type a∈{0..R-1} ; son y n'est piloté que par SON module
        # (bloc a = features [a·bs:(a+1)·bs]). Les R modules s'allument indépendamment selon la
        # proportion de leur sous-type dans chaque bootstrap -> R groupes de co-sélection -> r̂=R.
        R = args.subtypes; beta_b = args.signal / np.sqrt(bs)
        sub = rng.integers(0, R, size=n)
        yl = np.zeros(n)
        for a in range(R):
            mask = sub == a
            yl[mask] = X[mask][:, a*bs:(a+1)*bs] @ np.full(bs, beta_b)
        yl += rng.standard_normal(n)
    else:
        K = args.k
        beta = np.zeros(p); beta[:K] = args.signal / np.sqrt(K)
        yl = X[:, :K] @ beta[:K] + rng.standard_normal(n)
    y = (yl > 0).astype(int)
    return X[:args.n], y[:args.n], X[args.n:], y[args.n:]

# ───────── base learner AUTO-RÉGLÉ : C choisi par CV INTERNE (lasso L1 ou elastic-net L1+L2) ─────────
# Définition 2 du papier : f règle son hyperparamètre interne par une procédure ne dépendant
# que de la moitié S. Chaque demi-tirage choisit donc SON propre C par CV, puis renvoie le
# support binaire. La factorisation C̄=qq^T (Th. 2) est préservée verbatim quel que soit f :
# elle n'exige que (Lemme 3) indépendance des moitiés et (Lemme 1) même loi — pas un C commun.
# elastic-net : le L2 fait CO-SÉLECTIONNER les features corrélées (vs substitution du L1 pur),
# ce qui fait émerger les groupes comme valeurs propres positives de Σ -> sous-espace dim > 1.
if args.base == "elasticnet":
    PEN, SOLVER, L1R, COEF_THRESH = "elasticnet", "saga", args.l1_ratio, 1e-6   # saga : zéros numériques ~1e-7
else:
    PEN, SOLVER, L1R, COEF_THRESH = "l1", "liblinear", None, 1e-8

def lasso_select(Xs, ys):
    if len(np.unique(ys)) < 2:
        return np.zeros(Xs.shape[1])
    cv = min(args.cv_folds, int(np.bincount(ys).min()))   # pas plus de folds que la classe minoritaire
    common = dict(penalty=PEN, solver=SOLVER, class_weight="balanced", max_iter=5000)
    if cv < 2:                                            # trop peu d'une classe -> repli C médian
        clf = LogisticRegression(C=float(np.median(C_GRID)), l1_ratio=L1R, **common)
    else:
        clf = LogisticRegressionCV(Cs=C_GRID, cv=cv, scoring="roc_auc",
                                   l1_ratios=([L1R] if L1R is not None else None), **common)
    clf.fit(Xs, ys)
    return (np.abs(clf.coef_[0]) > COEF_THRESH).astype(float)

# ───────── opérateurs C (cross) et W (within) par double sous-échantillonnage disjoint ─────────
# C = moyenne_b (s_A s_Bᵀ + s_B s_Aᵀ)/2  -> ppᵀ (moitiés indépendantes : collapse q_j² sur la diag)
# W = moyenne_b (s_A s_Aᵀ + s_B s_Bᵀ)/2  -> Γ + ppᵀ (porte la structure de groupe)
def build_operators(X, y, seed):
    n, p = X.shape; m = n // 3                       # A,B disjoints non complémentaires (tiers 1 et 2)
    def one_boot(b):
        rng = np.random.default_rng([seed, b])
        perm = rng.permutation(n); A, Bs = perm[:m], perm[m:2*m]
        sA = lasso_select(X[A], y[A])                # (p,) — C réglé en interne sur A
        sB = lasso_select(X[Bs], y[Bs])             # (p,) — C réglé en interne sur B
        return sA, sB
    res = Parallel(n_jobs=-1)(delayed(one_boot)(b) for b in range(args.B))
    SA = np.array([r[0] for r in res]); SB = np.array([r[1] for r in res])  # (B, p)
    Cmat = (SA.T @ SB + SB.T @ SA) / (2*args.B)      # cross   -> ppᵀ
    Wmat = (SA.T @ SA + SB.T @ SB) / (2*args.B)      # within  -> Γ + ppᵀ
    return Cmat, Wmat

# ───────────────────────── analyse spectrale ─────────────────────────
def spectrum(Cmat, Wmat, gamma):
    M = (1-gamma)*Cmat + gamma*Wmat
    w, U = np.linalg.eigh(M)                          # ascendant
    return M, w[::-1], U[:, ::-1]                     # -> descendant

def best_gap(w):
    """plus grand gap parmi les rmax premiers : r̂ et le critère maximisé.
    --log-gap : log(λ_r) − log(λ_{r+1}) = log(λ_r/λ_{r+1}) ; sinon λ_r − λ_{r+1} (absolu)."""
    rm = min(args.rmax, len(w)-1)
    if args.log_gap:
        wpos = np.maximum(w[:rm+1], 1e-12)             # garde-fou : valeurs propres ≤ 0 -> eps
        crit = np.log(wpos[:rm]) - np.log(wpos[1:rm+1])
    else:
        crit = w[:rm] - w[1:rm+1]
    rhat = int(np.argmax(crit)) + 1
    return rhat, float(crit[rhat-1])

def select_by_mass(w, U, rhat):
    """loads ℓ_j = ||P_V e_j||² (somme = r̂) ; sélection jusqu'à masse signal r̂(1-ρ)."""
    V = U[:, :rhat]
    loads = (V**2).sum(axis=1)
    gap = (w[rhat-1] - w[rhat]) if rhat < len(w) else w[rhat-1]      # λ_r̂ − λ_{r̂+1}
    rho = float(w[rhat] / gap) if (rhat < len(w) and gap > 0) else 0.0   # ρ = λ_{r̂+1} / gap
    target = rhat * (1.0 - rho)
    order = np.argsort(loads)[::-1]
    cum = np.cumsum(loads[order])
    kstop = int(np.searchsorted(cum, target) + 1)
    kstop = max(1, min(kstop, len(order)))
    return set(order[:kstop].tolist()), loads, rho, target

# ───────────────────────── 1 run complet (1 seed) ─────────────────────────
def run_one(seed):
    Xtr, ytr, Xte, yte = generate(seed)
    Cmat, Wmat = build_operators(Xtr, ytr, seed)
    # gap spectral G(γ) -> γ* = argmax  (C est réglé dans le base learner ; γ est le primitif)
    per = []; gaps = np.zeros(len(GAMMA))               # sélection à CHAQUE γ
    for gi, g in enumerate(GAMMA):
        _, w, U = spectrum(Cmat, Wmat, g)
        rhat, gap = best_gap(w)
        sel, loads, rho, target = select_by_mass(w, U, rhat)
        sel_arr = np.array(sorted(sel)); ns = len(sel_arr)
        nt = int((sel_arr < KTRUE).sum()) if HAS_TRUTH else 0
        fdp = (1.0 - nt/max(1, ns)) if HAS_TRUTH else float("nan")
        gaps[gi] = gap
        per.append(dict(gap=gap, rhat=rhat, rho=rho, target=target, n_sel=ns, n_true=nt,
                        fdp=fdp, loads=loads, sel=sel_arr))
    gi_star = int(np.argmax(gaps)); g_star = float(GAMMA[gi_star])
    st = per[gi_star]; sel_arr = st["sel"]
    # ROC AUC à γ* : held-out si dispo (synthétique, COVID), sinon CV 5-fold (convention benchmark labo).
    yeval = yte if HAS_VAL else ytr
    if st["n_sel"] > 0:
        clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=5000)
        if HAS_VAL:
            clf.fit(Xtr[:, sel_arr], ytr); prob = clf.predict_proba(Xte[:, sel_arr])[:, 1]
        else:
            cvk = min(5, int(np.bincount(ytr).min()))
            cv = StratifiedKFold(cvk, shuffle=True, random_state=seed)
            prob = cross_val_predict(clf, Xtr[:, sel_arr], ytr, cv=cv, method="predict_proba")[:, 1]
    else:
        prob = np.full(len(yeval), 0.5)
    auc = roc_auc_score(yeval, prob); fpr, tpr, _ = roc_curve(yeval, prob)
    return dict(Cmat=Cmat, Wmat=Wmat, gaps=gaps, per=per, g_star=g_star, gi=int(gi_star),
                rhat=st["rhat"], rho=st["rho"], sel=sel_arr, n_sel=st["n_sel"],
                n_true=st["n_true"], fdp=st["fdp"], auc=auc, fpr=fpr, tpr=tpr, loads=st["loads"])

# ───────── sélection Spectre complète (opérateurs -> γ* -> features) sur un X, y donnés ─────────
def select_spectre(X, y, seed):
    """Toute la sélection : opérateurs, gap G(γ), γ*=argmax, features à γ*. Renvoie les indices."""
    Cmat, Wmat = build_operators(X, y, seed)
    gaps = np.zeros(len(GAMMA)); sels = []
    for gi, g in enumerate(GAMMA):
        _, w, U = spectrum(Cmat, Wmat, g)
        rhat, _ = best_gap(w)
        sel, *_ = select_by_mass(w, U, rhat)
        gaps[gi] = best_gap(w)[1]; sels.append(np.array(sorted(sel)))
    return sels[int(np.argmax(gaps))]

# ───────── AUC SANS FUITE : sélection Spectre + scaling refaits DANS chaque fold ─────────
def nested_cv_auc(Xraw, y, seed):
    """Chaque fold : scaler fit sur fold-train seul, sélection Spectre sur fold-train seul,
    classifieur fit sur fold-train, évalué sur fold-test jamais vu. AUC par fold (aucune fuite)."""
    cvk = min(5, int(np.bincount(y).min()))
    skf = StratifiedKFold(cvk, shuffle=True, random_state=seed)
    fold_aucs, fold_nsel = [], []
    for tr, te in skf.split(Xraw, y):
        sc = StandardScaler().fit(Xraw[tr])
        Xtr_f, Xte_f = sc.transform(Xraw[tr]), sc.transform(Xraw[te])
        sel = select_spectre(Xtr_f, y[tr], seed)            # sélection sur le fold-train SEUL
        if len(sel) == 0:
            prob = np.full(len(te), 0.5)
        else:
            clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=5000)
            clf.fit(Xtr_f[:, sel], y[tr]); prob = clf.predict_proba(Xte_f[:, sel])[:, 1]
        fold_aucs.append(roc_auc_score(y[te], prob)); fold_nsel.append(len(sel))
    return fold_aucs, fold_nsel

# ───────────────────────── boucle seeds ─────────────────────────
SEEDS = [args.seed + i for i in range(args.n_seeds)]
RES = [run_one(s) for s in SEEDS]
for s, r in zip(SEEDS, RES):
    truth = f"(vraies={r['n_true']}/{KTRUE})  FDP={r['fdp']:.2f}  " if HAS_TRUTH else ""
    print(f"  seed {s}: γ*={r['g_star']:.2f}  r̂={r['rhat']}  n_sel={r['n_sel']}  {truth}AUC={r['auc']:.3f}")

AUC_m = float(np.mean([r["auc"] for r in RES])); AUC_s = float(np.std([r["auc"] for r in RES]))
FDP_m = float(np.mean([r["fdp"] for r in RES])) if HAS_TRUTH else float("nan")
FDP_s = float(np.std([r["fdp"] for r in RES])) if HAS_TRUTH else float("nan")
GST_m = float(np.mean([r["g_star"] for r in RES])); NSEL_m = float(np.mean([r["n_sel"] for r in RES]))
fdp_txt = f"FDP moyen = {FDP_m:.2f} ± {FDP_s:.2f} | " if HAS_TRUTH else "FDP = N/A (pas de vérité terrain) | "
auc_label = "AUC validation" if HAS_VAL else "AUC CV (leaky)"
print(f"\n=> {auc_label} = {AUC_m:.3f} ± {AUC_s:.3f} | {fdp_txt}γ* moyen = {GST_m:.2f} | n_sel moyen = {NSEL_m:.1f}")

# AUC SANS FUITE par nested-CV (datasets sans validation uniquement)
NESTED = None
if args.nested_cv and not HAS_VAL and _RAW_TR is not None:
    Xraw, yraw = _RAW_TR
    fold_aucs, fold_nsel = nested_cv_auc(Xraw, yraw, args.seed)
    NESTED = (float(np.mean(fold_aucs)), float(np.std(fold_aucs)), fold_aucs, fold_nsel)
    folds_txt = "  ".join(f"f{i+1}={a:.3f}" for i, a in enumerate(fold_aucs))
    print(f"=> AUC nested-CV (SANS FUITE) = {NESTED[0]:.3f} ± {NESTED[1]:.3f}  "
          f"[{folds_txt}]  | n_sel par fold = {fold_nsel}")

# ───────────────────────── PLOTS ─────────────────────────
r0 = RES[0]
# (1) ROC AUC moyenne sur les seeds
grid = np.linspace(0, 1, 200)
tprs = np.array([np.interp(grid, r["fpr"], r["tpr"]) for r in RES]); tprs[:, 0] = 0.0
fig_roc, ax = plt.subplots(figsize=(5.2, 5))
for r in RES:
    ax.plot(r["fpr"], r["tpr"], color="#BBBBBB", lw=0.8, alpha=0.6)
ax.plot(grid, tprs.mean(0), color="#1f77b4", lw=2.4, label=f"ROC ({'val' if HAS_VAL else 'CV'}, AUC={AUC_m:.3f}±{AUC_s:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title(f"Spectre M(γ) — {DATA}\n" + (f"FDP={FDP_m:.2f}  |  " if HAS_TRUTH else "") + f"γ* = {GST_m:.2f}  n_sel={NSEL_m:.0f}", fontsize=10)
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)
fig_roc.tight_layout(); fig_roc.savefig(os.path.join(OUT, "roc.png"), dpi=140)

# (2) Courbe G(γ) + r̂(γ)
fig_gap, (axc, axr) = plt.subplots(1, 2, figsize=(12, 4.6))
allg = np.array([r["gaps"] for r in RES])
for r in RES:
    axc.plot(GAMMA, r["gaps"], color="#BBBBBB", lw=0.9, alpha=0.6)
_glab = "log-gap" if args.log_gap else "gap spectral"
_gexpr = "log(λ_r̂ / λ_r̂₊₁)" if args.log_gap else "λ_r̂ − λ_r̂₊₁"
axc.plot(GAMMA, allg.mean(0), color="#D62728", lw=2.4, label=f"G(γ) = {_gexpr}")
axc.axvline(GST_m, color="black", ls=":", lw=1.4, label=f"γ* = {GST_m:.2f}")
axc.set_xlabel("γ (mixing)"); axc.set_ylabel(f"{_glab} G(γ) = {_gexpr}")
axc.set_title(f"{_glab} G(γ) en fonction de γ — γ* = argmax", fontsize=10)
axc.legend(fontsize=9); axc.grid(alpha=0.25)
rhat_all = np.array([[d["rhat"] for d in r["per"]] for r in RES])
for r in RES:
    axr.plot(GAMMA, [d["rhat"] for d in r["per"]], color="#BBBBBB", lw=0.9, alpha=0.6)
axr.plot(GAMMA, rhat_all.mean(0), color="#1f77b4", lw=2.4, marker="o", ms=3, label="r̂(γ) moyen")
axr.axvline(GST_m, color="black", ls=":", lw=1.4)
axr.set_xlabel("γ (mixing)"); axr.set_ylabel("r̂ = nb de groupes = dim(V̂)")
axr.set_ylim(0, max(2, args.rmax)); axr.set_title("r̂(γ) — dim du sous-espace V̂", fontsize=10)
axr.legend(fontsize=9); axr.grid(alpha=0.25)
fig_gap.suptitle(f"Spectre M(γ) — {_glab} G(γ) en fonction de γ et dimension du sous-espace", fontsize=11)
fig_gap.tight_layout(rect=[0, 0, 1, 0.95]); fig_gap.savefig(os.path.join(OUT, "spectral_gap.png"), dpi=140)

# (3) heatmaps M(γ) pour plusieurs γ — seed 0
Cmat, Wmat = r0["Cmat"], r0["Wmat"]
PZ = min(P, 60)
show_gammas = np.linspace(0, 1, min(6, args.n_gamma))
ncol = 3; nrow = (len(show_gammas) + ncol - 1) // ncol
fig_mat, axes = plt.subplots(nrow, ncol, figsize=(4.4*ncol, 4.2*nrow), squeeze=False); axes = axes.ravel()
subs = [((1-g)*Cmat + g*Wmat)[:PZ, :PZ] for g in show_gammas]
vmax = float(max(np.percentile(s, 99) for s in subs)) or max(s.max() for s in subs)
for ax in axes[len(show_gammas):]:
    ax.axis("off")
for ax, g, Msub in zip(axes, show_gammas, subs):
    im = ax.imshow(Msub, cmap="hot", vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), KTRUE, KTRUE, fill=False, edgecolor="#00BFFF", lw=1.6))
    _, w, _ = spectrum(Cmat, Wmat, g); rh, gp = best_gap(w)
    ax.set_title(f"γ={g:.2f}  (r̂={rh}, gap={gp:.3f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig_mat.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label="score de co-sélection (chaud = grand)")
fig_mat.suptitle(f"Spectre M(γ)=(1−γ)C+γW  (zoom features 0..{PZ-1}" + (f" ; cadre bleu = vraies 0..{KTRUE-1})" if HAS_TRUTH else ")"), fontsize=11, y=0.99)
fig_mat.savefig(os.path.join(OUT, "matrices.png"), dpi=140, bbox_inches="tight")

# (3b) C et W séparées
fig_cw, axcw = plt.subplots(1, 2, figsize=(11, 5))
for ax, Mx, ttl in [(axcw[0], Cmat, "C — cross (moitiés INDÉPENDANTES)\ndiag = q_j² (écrasement)"),
                    (axcw[1], Wmat, "W — within (MÊME moitié)\ndiag = q_j, off-diag = groupes")]:
    sub = Mx[:PZ, :PZ]; vm = float(np.percentile(sub, 99)) or float(sub.max())
    im = ax.imshow(sub, cmap="hot", vmin=0, vmax=vm, interpolation="nearest")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), KTRUE, KTRUE, fill=False, edgecolor="#00BFFF", lw=1.6))
    ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig_cw.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
fig_cw.suptitle(f"Spectre — matrices C et W (zoom features 0..{PZ-1}" + (", cadre bleu = vraies)" if HAS_TRUTH else ")"), fontsize=11)
fig_cw.tight_layout(); fig_cw.savefig(os.path.join(OUT, "matrices_CW.png"), dpi=140)

# (3c) Spectre des valeurs propres de M(γ)
nshow = min((KTRUE if HAS_TRUTH else 45) + 20, P)
fig_eig, axe = plt.subplots(1, 2, figsize=(12, 4.6))
cols = plt.cm.viridis(np.linspace(0, 0.88, len(show_gammas)))
for g, c in zip(show_gammas, cols):
    w = spectrum(Cmat, Wmat, g)[1]
    axe[0].plot(range(1, nshow+1), w[:nshow], marker="o", ms=3, color=c, label=f"γ={g:.2f}")
    axe[1].plot(range(2, nshow+1), w[1:nshow], marker="o", ms=3, color=c)
for a in axe:
    if HAS_TRUTH: a.axvline(KTRUE + 0.5, color="grey", ls=":", lw=1.0)
    a.set_xlabel("indice i"); a.set_ylabel("valeur propre λ_i"); a.grid(alpha=0.25)
axe[0].set_title("spectre complet (λ₁ rang-1 inclus)"); axe[1].set_title("zoom à partir de λ₂" + (f" (k={KTRUE})" if HAS_TRUTH else ""))
axe[0].legend(fontsize=7, ncol=2, title="γ")
fig_eig.suptitle("Spectre — valeurs propres de M(γ)=(1−γ)C+γW selon γ", fontsize=11)
fig_eig.tight_layout(); fig_eig.savefig(os.path.join(OUT, "eigenvalues.png"), dpi=140)

# (3d) ρ(γ), S(γ), charges à γ* (seed 0)
per = r0["per"]
rho_g  = np.array([d["rho"]   for d in per])
nsel_g = np.array([d["n_sel"] for d in per])
ntru_g = np.array([d["n_true"] for d in per])
fig_sel, axs = plt.subplots(1, 3, figsize=(15.5, 4.5))
axs[0].plot(GAMMA, rho_g, color="#9467BD", marker="o", ms=3)
axs[0].axvline(r0["g_star"], color="black", ls=":", lw=1.3, label=f"γ*={r0['g_star']:.2f}")
axs[0].set_xlabel("γ"); axs[0].set_ylabel("ρ = λ_{r̂+1} / gap")
axs[0].set_title(f"ρ(γ) — résidu spectral relatif  (ρ(γ*)={r0['rho']:.3f})")
axs[0].grid(alpha=0.25); axs[0].legend(fontsize=8)
axs[1].plot(GAMMA, nsel_g, color="#1f77b4", marker="o", ms=3, label="|S(γ)| sélectionnées")
if HAS_TRUTH:
    axs[1].plot(GAMMA, ntru_g, color="#2ca02c", marker="s", ms=3, label=f"vraies trouvées (/{KTRUE})")
    axs[1].axhline(KTRUE, color="grey", ls="--", lw=0.8)
axs[1].axvline(r0["g_star"], color="black", ls=":", lw=1.3)
axs[1].set_xlabel("γ"); axs[1].set_ylabel("nb de features")
axs[1].set_title("ensemble S(γ) : taille et recall"); axs[1].grid(alpha=0.25); axs[1].legend(fontsize=8)
loads = r0["loads"]; order = np.argsort(loads)[::-1]; sel_set = set(r0["sel"].tolist())
nshow2 = min(80, len(order)); ranks = np.arange(1, nshow2+1)
cols3 = ["#D62728" if order[i] in sel_set else "#C8C8C8" for i in range(nshow2)]
axs[2].bar(ranks, loads[order][:nshow2], color=cols3, width=1.0)
for i in range(nshow2):
    if HAS_TRUTH and order[i] < KTRUE:
        axs[2].plot(ranks[i], loads[order[i]] + loads.max()*0.03, marker="*", color="#FF7F0E", ms=8)
axs[2].set_xlabel("rang (charge décroissante)"); axs[2].set_ylabel("charge ℓ_j")
axs[2].set_title(f"charges à γ*  (rouge = sélectionnée, ★ = vraie)  |S|={r0['n_sel']}")
axs[2].grid(alpha=0.25, axis="y")
fig_sel.suptitle("Spectre — ρ(γ), ensemble sélectionné S(γ), et charges des features", fontsize=11)
fig_sel.tight_layout(rect=[0, 0, 1, 0.95]); fig_sel.savefig(os.path.join(OUT, "selection.png"), dpi=140)

# (3f) LOG-SPECTRE à γ* : log(λ_i) et distribution des log-gaps  (sélection = plus grand log-gap)
w_star = spectrum(r0["Cmat"], r0["Wmat"], r0["g_star"])[1]
nsl = min(args.rmax + 12, len(w_star))
logmu = np.log(np.maximum(w_star[:nsl], 1e-12))                       # log λ_i (plancher 1e-12 si ≤0)
loggaps = logmu[:nsl-1] - logmu[1:nsl]                                # log(λ_i/λ_{i+1})
rm = min(args.rmax, nsl-1); rhat_log = int(np.argmax(loggaps[:rm])) + 1   # r̂ = plus grand log-gap
fig_log, (l1, l2) = plt.subplots(1, 2, figsize=(13, 4.6))
l1.bar(np.arange(1, nsl+1), logmu, width=0.8,
       color=["#D62728" if i < rhat_log else "#9AA0A6" for i in range(nsl)])
l1.axvline(rhat_log + 0.5, color="black", ls=":", lw=1.4)
l1.set_xlabel("indice i"); l1.set_ylabel("log λ_i")
l1.set_title(f"log-spectre de M(γ*={r0['g_star']:.2f}) — r̂={rhat_log} (rouge)", fontsize=10); l1.grid(alpha=0.25, axis="y")
l2.bar(np.arange(1, len(loggaps)+1), loggaps, width=0.8,
       color=["#D62728" if i == rhat_log-1 else "#9AA0A6" for i in range(len(loggaps))])
l2.set_xlabel("entre log λ_i et log λ_{i+1}"); l2.set_ylabel("log-gap = log(λ_i / λ_{i+1})")
l2.set_title(f"distribution des log-gaps — max à i={rhat_log}", fontsize=10); l2.grid(alpha=0.25, axis="y")
fig_log.suptitle("Spectre M(γ) — log-spectre à γ* et log-gaps (sélection par le plus grand log-gap)", fontsize=11)
fig_log.tight_layout(rect=[0, 0, 1, 0.95]); fig_log.savefig(os.path.join(OUT, "log_spectrum.png"), dpi=140)

# (3g) log-spectre et log-gaps EN FONCTION DE γ (une courbe par γ sur la grille show_gammas)
nlg = min(50, P)
fig_lg, (g1, g2) = plt.subplots(1, 2, figsize=(13, 4.6))
cols_lg = plt.cm.viridis(np.linspace(0, 0.88, len(show_gammas)))
for g, c in zip(show_gammas, cols_lg):
    wg = spectrum(Cmat, Wmat, g)[1]
    lwg = np.log(np.maximum(wg[:nlg], 1e-12))
    g1.plot(range(1, nlg+1), lwg, marker="o", ms=2.5, color=c, label=f"γ={g:.2f}")
    g2.plot(range(1, nlg), lwg[:nlg-1] - lwg[1:nlg], marker="o", ms=2.5, color=c)
if HAS_TRUTH:
    for a in (g1, g2): a.axvline(KTRUE + 0.5, color="grey", ls=":", lw=1.0)
g1.set_xlabel("indice i"); g1.set_ylabel("log λ_i")
g1.set_title("log-spectre selon γ", fontsize=10); g1.grid(alpha=0.25); g1.legend(fontsize=7, ncol=2, title="γ")
g2.set_xlabel("entre log λ_i et log λ_{i+1}"); g2.set_ylabel("log-gap = log(λ_i / λ_{i+1})")
g2.set_title("distribution des log-gaps selon γ", fontsize=10); g2.grid(alpha=0.25)
fig_lg.suptitle("Spectre M(γ) — log-spectre et log-gaps en fonction de γ", fontsize=11)
fig_lg.tight_layout(rect=[0, 0, 1, 0.95]); fig_lg.savefig(os.path.join(OUT, "log_spectrum_gamma.png"), dpi=140)

# (3e) Top features avec noms (labo) + fichier texte
fig_top = None
if FEATURE_NAMES is not None:
    topN = min(30, len(order)); top_idx = order[:topN]
    names = [FEATURE_NAMES[j] for j in top_idx]; vals = loads[top_idx]
    cols_t = ["#D62728" if j in sel_set else "#AAAAAA" for j in top_idx]
    fig_top, axt = plt.subplots(figsize=(8.5, max(5, topN*0.3)))
    ypos = np.arange(topN)[::-1]
    axt.barh(ypos, vals, color=cols_t)
    axt.set_yticks(ypos); axt.set_yticklabels(names, fontsize=7)
    axt.set_xlabel("charge ℓ_j"); axt.grid(alpha=0.25, axis="x")
    axt.set_title(f"{DATA} — top {topN} features par charge décroissante\n(rouge = sélectionnée, γ*={r0['g_star']:.2f}, |S|={r0['n_sel']})", fontsize=10)
    fig_top.tight_layout(); fig_top.savefig(os.path.join(OUT, "top_features.png"), dpi=140)
    with open(os.path.join(OUT, "selected_features.txt"), "w") as fh:
        for rk, j in enumerate(order):
            if j in sel_set:
                fh.write(f"{rk+1}\t{FEATURE_NAMES[j]}\t{loads[j]:.5f}\n")

# (4) PDF combiné + résumé
with PdfPages(os.path.join(OUT, "analyse_spectre.pdf")) as pdf:
    for f in (fig_roc, fig_gap, fig_mat, fig_cw, fig_eig, fig_sel, fig_log, fig_lg):
        pdf.savefig(f)
    if fig_top is not None:
        pdf.savefig(fig_top)
with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write(f"Spectre M(γ) — {TAG}\n")
    fh.write(f"{'AUC moyenne' if HAS_VAL else 'AUC CV (leaky)'} = {AUC_m:.4f} ± {AUC_s:.4f}\n")
    if NESTED is not None:
        fh.write(f"AUC nested-CV (SANS FUITE) = {NESTED[0]:.4f} ± {NESTED[1]:.4f}  "
                 f"folds={['%.3f'%a for a in NESTED[2]]}  n_sel/fold={NESTED[3]}\n")
    fh.write(f"FDP moyen   = {FDP_m:.4f} ± {FDP_s:.4f}\n" if HAS_TRUTH else "FDP        = N/A (pas de vérité terrain)\n")
    fh.write(f"gamma*      = {GST_m:.3f}\n")
    fh.write(f"n_sel moyen = {NSEL_m:.2f}\n")
    for s, r in zip(SEEDS, RES):
        fh.write(f"  seed {s}: gamma*={r['g_star']:.2f} rhat={r['rhat']} n_sel={r['n_sel']} "
                 + (f"true={r['n_true']}/{KTRUE} FDP={r['fdp']:.3f} " if HAS_TRUTH else "") + f"AUC={r['auc']:.3f}\n")
print(f"\nFigures + analyse_spectre.pdf + summary.txt dans : {OUT}")
