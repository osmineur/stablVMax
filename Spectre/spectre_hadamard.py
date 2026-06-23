#!/usr/bin/env python3
"""
Spectre — Hadamard Co-Reproducibility Spectral Selection (Osman Rittano)
=======================================================================
Stability selection SANS features artificielles, modèle de HADAMARD (plus de γ).
À chaque bootstrap on tire deux sous-échantillons DISJOINTS A et B (tiers 1 et tiers 2,
tiers 3 inutilisé), on fit le base learner (lasso logistique) sur chacun -> vecteurs binaires
a(A_b), a(B_b). On forme la matrice de CO-SÉLECTION puis son CARRÉ DE HADAMARD débiaisé :

    Q̂_jk = (1/2B) Σ_b [ a_j(A_b)a_k(A_b) + a_j(B_b)a_k(B_b) ]      (diag = fréquence de sélection q_j)
    Ĥ_jk = Q̂_jk²  −  Q̂_jk(1−Q̂_jk)/(B−1)                            (carré débiaisé : co-reproductibilité)

Le carré écrase l'amplitude des nulles en q_j² (collapse quadratique), donc les vraies valeurs
propres de groupe dominent. On lit le plus grand GAP SPECTRAL de Ĥ -> r̂ (nb de groupes),
puis on sélectionne par MASSE SPECTRALE (loads ĉ_j = ||P_V̂ e_j||², somme = r̂ ; cumul jusqu'à r̂(1−ρ̂)).

Sorties (dossier results_spectre/<config>/) :
  - roc.png        : ROC AUC (FDP, n_sel)
  - spectrum.png   : valeurs propres μ_i, gaps consécutifs (+ r̂), cumul de masse + cible
  - loads.png      : histogramme des charges (certificat), charges triées, collapse q_j -> q_j²
  - matrices.png   : heatmaps couleur de Q̂ et Ĥ (le carré de Hadamard visible)
  - top_features.png + selected_features.txt (données labo, avec noms)
  - analyse_spectre.pdf + summary.txt (AUC, FDP, r̂, ρ̂, n_sel, valeurs propres)
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
ap.add_argument("--n-gamma", "--n_gamma", type=int, default=21, dest="n_gamma", help="(inutilisé — modèle Hadamard, plus de γ)")
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
    print(f"Spectre/Hadamard : n={args.n} p={args.p} k={args.k} signal={args.signal} rho={args.rho} "
          f"B={args.B} n_seeds={args.n_seeds}")
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

# ───────── matrice de co-sélection Q̂ par double sous-échantillonnage disjoint ─────────
# Q̂_jk = (1/2B) Σ_b [a_j(A_b)a_k(A_b) + a_j(B_b)a_k(B_b)] : fréquence de co-sélection sur une MÊME
# moitié. Diag Q̂_jj = fréquence de sélection q_j. Chaque moitié produit UN vecteur binaire.
def build_Q(X, y, seed):
    n, p = X.shape; m = n // 3                       # A,B disjoints non complémentaires (tiers 1 et 2)
    def one_boot(b):
        rng = np.random.default_rng([seed, b])
        perm = rng.permutation(n); A, Bs = perm[:m], perm[m:2*m]
        sA = lasso_select(X[A], y[A])                # (p,) — C réglé en interne sur A
        sB = lasso_select(X[Bs], y[Bs])             # (p,) — C réglé en interne sur B
        return sA, sB
    res = Parallel(n_jobs=-1)(delayed(one_boot)(b) for b in range(args.B))
    SA = np.array([r[0] for r in res]); SB = np.array([r[1] for r in res])  # (B, p)
    return (SA.T @ SA + SB.T @ SB) / (2*args.B)      # co-sélection Q̂ ; diag = q_j

# ───────────────────────── opérateur de Hadamard + analyse spectrale ─────────────────────────
def hadamard(Q):
    """Carré de Hadamard débiaisé : Ĥ_jk = Q̂_jk² − Q̂_jk(1−Q̂_jk)/(B−1). Le carré écrase
    l'amplitude des nulles en q_j² (collapse quadratique) ; la correction enlève le gonflement
    dû à la variance d'estimation (sur la diag : Var empirique d'une Bernoulli sur B tirages)."""
    return Q*Q - Q*(1.0 - Q)/(args.B - 1)

def spectrum(H):
    w, U = np.linalg.eigh(H)                          # symétrique ; ascendant
    return w[::-1], U[:, ::-1]                        # -> descendant

def best_gap(w):
    """Plus grand gap spectral SOUS CONTRAINTE ρ̂(r)<1 (sélection non dégénérée).
    gap_r = μ_r − μ_{r+1} ; ρ̂(r) = μ_{r+1}/gap_r ; ρ̂<1 ⟺ μ_r > 2μ_{r+1}.
    On prend le plus grand gap_r parmi les r où ρ̂(r)<1 ; si aucun, repli sur le plus grand gap."""
    rm = min(args.rmax, len(w)-1)
    gaps = w[:rm] - w[1:rm+1]                          # gap_r, r=1..rm
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(gaps > 0, w[1:rm+1] / gaps, np.inf)   # ρ̂(r) = μ_{r+1}/gap_r
    valid = rho < 1.0
    cand = np.where(valid, gaps, -np.inf)
    idx = int(np.argmax(cand)) if valid.any() else int(np.argmax(gaps))   # repli si aucun r non-dégénéré
    return idx + 1, float(gaps[idx])

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

def analyze_operator(M):
    """Pipeline spectral complet sur une matrice M : valeurs propres, gap -> r̂, sélection par masse."""
    w, U = spectrum(M)
    rhat, gap = best_gap(w)
    sel, loads, rho, target = select_by_mass(w, U, rhat)
    sel_arr = np.array(sorted(sel))
    nt = int((sel_arr < KTRUE).sum()) if HAS_TRUTH else 0
    fdp = (1.0 - nt/max(1, len(sel_arr))) if HAS_TRUTH else float("nan")
    return dict(w=w, rhat=rhat, gap=gap, rho=rho, target=target, loads=loads,
                sel=sel_arr, n_sel=len(sel_arr), n_true=nt, fdp=fdp)

# ───────────────────────── 1 run complet (1 seed) ─────────────────────────
def eval_auc(sel_arr, Xtr, ytr, Xte, yte, seed):
    """AUC d'un classifieur l2 sur les features sélectionnées. Held-out si dispo, sinon CV 5-fold."""
    yeval = yte if HAS_VAL else ytr
    if len(sel_arr) > 0:
        clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=5000)
        if HAS_VAL:
            clf.fit(Xtr[:, sel_arr], ytr); prob = clf.predict_proba(Xte[:, sel_arr])[:, 1]
        else:
            cvk = min(5, int(np.bincount(ytr).min()))
            cv = StratifiedKFold(cvk, shuffle=True, random_state=seed)
            prob = cross_val_predict(clf, Xtr[:, sel_arr], ytr, cv=cv, method="predict_proba")[:, 1]
    else:
        prob = np.full(len(yeval), 0.5)
    fpr, tpr, _ = roc_curve(yeval, prob)
    return roc_auc_score(yeval, prob), fpr, tpr

def run_one(seed):
    Xtr, ytr, Xte, yte = generate(seed)
    Q = build_Q(Xtr, ytr, seed)
    H = hadamard(Q)                                    # opérateur de co-reproductibilité
    aH = analyze_operator(H)                            # sélection PRIMAIRE (Hadamard)
    aQ = analyze_operator(Q)                            # COMPARAISON : même pipeline sur Q̂ brut
    auc, fpr, tpr = eval_auc(aH["sel"], Xtr, ytr, Xte, yte, seed)
    auc_Q = eval_auc(aQ["sel"], Xtr, ytr, Xte, yte, seed)[0]   # AUC de la sélection sur Q̂ (comparaison)
    return dict(Q=Q, H=H, aH=aH, aQ=aQ, auc_Q=auc_Q,
                w=aH["w"], gap=aH["gap"], rhat=aH["rhat"], rho=aH["rho"], target=aH["target"],
                loads=aH["loads"], sel=aH["sel"], n_sel=aH["n_sel"], n_true=aH["n_true"], fdp=aH["fdp"],
                auc=auc, fpr=fpr, tpr=tpr)

# ───────── sélection Spectre complète (Q -> Ĥ -> r̂ -> features) sur un X, y donnés ─────────
def select_spectre(X, y, seed):
    """Toute la sélection : Q̂, Ĥ, gap spectral -> r̂, sélection par masse. Renvoie les indices."""
    w, U = spectrum(hadamard(build_Q(X, y, seed)))
    rhat, _ = best_gap(w)
    sel, *_ = select_by_mass(w, U, rhat)
    return np.array(sorted(sel))

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
    print(f"  seed {s}: r̂={r['rhat']}  ρ̂={r['rho']:.3f}  n_sel={r['n_sel']}  {truth}AUC={r['auc']:.3f}")
    aq = r["aQ"]; tq = f"(vraies={aq['n_true']}/{KTRUE}) FDP={aq['fdp']:.2f} " if HAS_TRUTH else ""
    print(f"         [Q̂ brut]  r̂={aq['rhat']}  ρ̂={aq['rho']:.3f}  n_sel={aq['n_sel']}  {tq}AUC={r['auc_Q']:.3f}")

AUC_m = float(np.mean([r["auc"] for r in RES])); AUC_s = float(np.std([r["auc"] for r in RES]))
FDP_m = float(np.mean([r["fdp"] for r in RES])) if HAS_TRUTH else float("nan")
FDP_s = float(np.std([r["fdp"] for r in RES])) if HAS_TRUTH else float("nan")
RHAT_m = float(np.mean([r["rhat"] for r in RES])); RHO_m = float(np.mean([r["rho"] for r in RES]))
NSEL_m = float(np.mean([r["n_sel"] for r in RES]))
fdp_txt = f"FDP moyen = {FDP_m:.2f} ± {FDP_s:.2f} | " if HAS_TRUTH else "FDP = N/A (pas de vérité terrain) | "
auc_label = "AUC validation" if HAS_VAL else "AUC CV (leaky)"
print(f"\n=> {auc_label} = {AUC_m:.3f} ± {AUC_s:.3f} | {fdp_txt}r̂ moyen = {RHAT_m:.1f} | ρ̂ moyen = {RHO_m:.3f} | n_sel moyen = {NSEL_m:.1f}")

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
r0 = RES[0]; w0 = r0["w"]; loads = r0["loads"]; sel_set = set(r0["sel"].tolist())
rhat = r0["rhat"]; rho = r0["rho"]; target = r0["target"]
order = np.argsort(loads)[::-1]
TAG_T = f"{DATA}" + (f"  (FDP={FDP_m:.2f})" if HAS_TRUTH else "")

# (1) ROC AUC moyenne sur les seeds
grid = np.linspace(0, 1, 200)
tprs = np.array([np.interp(grid, r["fpr"], r["tpr"]) for r in RES]); tprs[:, 0] = 0.0
fig_roc, ax = plt.subplots(figsize=(5.2, 5))
for r in RES:
    ax.plot(r["fpr"], r["tpr"], color="#BBBBBB", lw=0.8, alpha=0.6)
ax.plot(grid, tprs.mean(0), color="#1f77b4", lw=2.4, label=f"ROC ({'val' if HAS_VAL else 'CV'}, AUC={AUC_m:.3f}±{AUC_s:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title(f"Spectre/Hadamard — {TAG_T}\nr̂={int(round(RHAT_m))}  ρ̂={RHO_m:.2f}  n_sel={NSEL_m:.0f}", fontsize=10)
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)
fig_roc.tight_layout(); fig_roc.savefig(os.path.join(OUT, "roc.png"), dpi=140)

# (2) SPECTRE : valeurs propres μ_i, gaps consécutifs (+ r̂), cumul de masse + cible
nshow = min(args.rmax + 12, len(w0))
mu = w0[:nshow]; gaps = w0[:nshow-1] - w0[1:nshow]
fig_spec, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.6))
# (a) valeurs propres : r̂ premières en rouge (groupes), reste gris (résidu) + plancher μ_{r̂+1}
cols_e = ["#D62728" if i < rhat else "#9AA0A6" for i in range(nshow)]
a1.bar(np.arange(1, nshow+1), mu, color=cols_e, width=0.8)
if rhat < len(w0):
    a1.axhline(w0[rhat], color="#2ca02c", ls="--", lw=1.2, label=f"plancher μ_{{r̂+1}}={w0[rhat]:.3f}")
a1.axvline(rhat + 0.5, color="black", ls=":", lw=1.4)
a1.set_xlabel("indice i"); a1.set_ylabel("valeur propre μ_i de Ĥ")
a1.set_title(f"spectre de Ĥ — r̂={rhat} groupes (rouge)", fontsize=10)
a1.legend(fontsize=8); a1.grid(alpha=0.25, axis="y")
# (b) gaps consécutifs : le max (à r̂) encadré
cols_g = ["#D62728" if i == rhat-1 else "#9AA0A6" for i in range(len(gaps))]
a2.bar(np.arange(1, len(gaps)+1), gaps, color=cols_g, width=0.8)
a2.set_xlabel("entre μ_i et μ_{i+1}"); a2.set_ylabel("gap μ_i − μ_{i+1}")
a2.set_title(f"gaps spectraux — max à i={rhat} (gap={r0['gap']:.3f})", fontsize=10)
a2.grid(alpha=0.25, axis="y")
# (c) cumul de masse trié : cible r̂(1−ρ̂) et arrêt m̂
cum = np.cumsum(loads[order]); mhat = r0["n_sel"]; ncum = min(len(cum), max(40, mhat + 20))
a3.plot(np.arange(1, ncum+1), cum[:ncum], color="#1f77b4", lw=2)
a3.axhline(target, color="#D62728", ls="--", lw=1.3, label=f"cible r̂(1−ρ̂)={target:.2f}")
a3.axvline(mhat, color="black", ls=":", lw=1.3, label=f"arrêt m̂={mhat}")
a3.set_xlabel("rang (charge décroissante)"); a3.set_ylabel("masse cumulée Σĉ")
a3.set_title(f"sélection par masse — ρ̂={rho:.3f}", fontsize=10)
a3.legend(fontsize=8); a3.grid(alpha=0.25)
fig_spec.suptitle("Spectre/Hadamard — spectre de Ĥ, gap → r̂, et sélection par masse cumulée", fontsize=11)
fig_spec.tight_layout(rect=[0, 0, 1, 0.95]); fig_spec.savefig(os.path.join(OUT, "spectrum.png"), dpi=140)

# (3) CHARGES : histogramme (certificat), charges triées, collapse q_j → Ĥ_jj
fig_load, (b1, b2, b3) = plt.subplots(1, 3, figsize=(16, 4.6))
# (a) histogramme des charges — le creux (vallée) est le certificat de séparabilité
cut = float(loads[r0["sel"]].min()) if r0["n_sel"] > 0 else 0.0
b1.hist(loads, bins=60, color="#9AA0A6", log=True)
b1.axvline(cut, color="#D62728", ls="--", lw=1.4, label=f"cutoff sélection ({cut:.3f})")
b1.set_xlabel("charge ĉ_j"); b1.set_ylabel("nb de features (log)")
b1.set_title("histogramme des charges — vallée = certificat", fontsize=10)
b1.legend(fontsize=8); b1.grid(alpha=0.25)
# (b) charges triées : sélectionnées en rouge, vraies marquées ★
nb = min(80, len(order)); ranks = np.arange(1, nb+1)
cols3 = ["#D62728" if order[i] in sel_set else "#C8C8C8" for i in range(nb)]
b2.bar(ranks, loads[order][:nb], color=cols3, width=1.0)
for i in range(nb):
    if HAS_TRUTH and order[i] < KTRUE:
        b2.plot(ranks[i], loads[order[i]] + loads.max()*0.03, marker="*", color="#FF7F0E", ms=8)
b2.set_xlabel("rang"); b2.set_ylabel("charge ĉ_j")
b2.set_title(f"charges triées (rouge=sél, ★=vraie)  |S|={r0['n_sel']}", fontsize=10)
b2.grid(alpha=0.25, axis="y")
# (c) collapse quadratique : diag(Q̂)=q_j  vs  diag(Ĥ)≈q_j²
qd = np.diag(r0["Q"]); hd = np.diag(r0["H"])
if HAS_TRUTH:
    b3.scatter(qd[KTRUE:], hd[KTRUE:], s=8, color="#9AA0A6", label="nulles")
    b3.scatter(qd[:KTRUE], hd[:KTRUE], s=24, color="#D62728", label="vraies")
else:
    issel = np.array([j in sel_set for j in range(len(qd))])
    b3.scatter(qd[~issel], hd[~issel], s=8, color="#9AA0A6", label="non sél.")
    b3.scatter(qd[issel], hd[issel], s=24, color="#D62728", label="sélectionnées")
xx = np.linspace(0, max(qd.max(), 1e-6), 100)
b3.plot(xx, xx**2, color="black", ls=":", lw=1.2, label="y=q² (collapse)")
b3.set_xlabel("q_j = diag(Q̂)  (fréquence de sélection)"); b3.set_ylabel("diag(Ĥ) ≈ q_j²")
b3.set_title("collapse quadratique de Hadamard", fontsize=10)
b3.legend(fontsize=8); b3.grid(alpha=0.25)
fig_load.suptitle("Spectre/Hadamard — charges des features et collapse quadratique", fontsize=11)
fig_load.tight_layout(rect=[0, 0, 1, 0.95]); fig_load.savefig(os.path.join(OUT, "loads.png"), dpi=140)

# (4) MATRICES Q̂ et Ĥ en couleur (zoom sur le bloc le plus co-sélectionné)
if HAS_TRUTH:                                     # synthétique : features 0..PZ-1 (bloc vrai = top-gauche)
    PZ = min(P, 60); idx_show = np.arange(PZ)
else:                                             # labo : top-PZ features par fréquence de sélection
    PZ = min(P, 60); idx_show = np.argsort(np.diag(r0["Q"]))[::-1][:PZ]
Qs = r0["Q"][np.ix_(idx_show, idx_show)]; Hs = r0["H"][np.ix_(idx_show, idx_show)]
fig_mat, (m1, m2) = plt.subplots(1, 2, figsize=(12, 5.4))
for ax, M, ttl in [(m1, Qs, "Q̂ — co-sélection (diag = q_j)"),
                   (m2, Hs, "Ĥ = Q̂∘² débiaisé (nulles écrasées en q²)")]:
    vm = float(np.percentile(M, 99)) or float(M.max())
    im = ax.imshow(M, cmap="hot", vmin=0, vmax=max(vm, 1e-9), interpolation="nearest")
    if HAS_TRUTH:
        ax.add_patch(plt.Rectangle((-0.5, -0.5), KTRUE, KTRUE, fill=False, edgecolor="#00BFFF", lw=1.8))
    ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig_mat.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
note = f"bloc vrai 0..{KTRUE-1} (cadre bleu)" if HAS_TRUTH else f"top {PZ} features par q_j"
fig_mat.suptitle(f"Spectre/Hadamard — matrices Q̂ et Ĥ  (zoom : {note})", fontsize=11)
fig_mat.tight_layout(rect=[0, 0, 1, 0.95]); fig_mat.savefig(os.path.join(OUT, "matrices.png"), dpi=140)

# (4b) COMPARAISON Q̂ (brut) vs Ĥ (carré de Hadamard) — même pipeline spectral sur les deux
def _draw_row(axrow, a, name):
    w = a["w"]; ld = a["loads"]; rh = a["rhat"]; tgt = a["target"]
    od = np.argsort(ld)[::-1]; nv = min(args.rmax + 12, len(w)); mu = w[:nv]; gp = w[:nv-1] - w[1:nv]
    axrow[0].bar(np.arange(1, nv+1), mu, width=0.8,
                 color=["#D62728" if i < rh else "#9AA0A6" for i in range(nv)])
    if rh < len(w): axrow[0].axhline(w[rh], color="#2ca02c", ls="--", lw=1)
    fdp_t = f"  FDP={a['fdp']:.2f}" if HAS_TRUTH else ""
    axrow[0].set_ylabel(f"{name}\nr̂={rh}  n_sel={a['n_sel']}{fdp_t}", fontsize=10)
    axrow[0].set_title("valeurs propres (r̂ en rouge)", fontsize=9)
    axrow[1].bar(np.arange(1, len(gp)+1), gp, width=0.8,
                 color=["#D62728" if i == rh-1 else "#9AA0A6" for i in range(len(gp))])
    axrow[1].set_title(f"gaps — max@{rh} (gap={a['gap']:.3g})", fontsize=9)
    cut = float(ld[a["sel"]].min()) if a["n_sel"] > 0 else 0.0
    axrow[2].hist(ld, bins=50, color="#9AA0A6", log=True); axrow[2].axvline(cut, color="#D62728", ls="--", lw=1.2)
    axrow[2].set_title("histogramme des charges", fontsize=9)
    cum = np.cumsum(ld[od]); nc = min(len(cum), max(40, a["n_sel"]+20))
    axrow[3].plot(np.arange(1, nc+1), cum[:nc], color="#1f77b4", lw=2)
    axrow[3].axhline(tgt, color="#D62728", ls="--", lw=1.2); axrow[3].axvline(a["n_sel"], color="black", ls=":", lw=1.2)
    axrow[3].set_title(f"masse : cible r̂(1−ρ̂)={tgt:.2f}, ρ̂={a['rho']:.2f}" + ("  (dégénéré)" if tgt <= 0 else ""), fontsize=9)
    for ax in axrow:
        ax.grid(alpha=0.25)
fig_cmp, axc2 = plt.subplots(2, 4, figsize=(18, 8))
_draw_row(axc2[0], r0["aQ"], "Q̂ (brut)")
_draw_row(axc2[1], r0["aH"], "Ĥ (Hadamard)")
fig_cmp.suptitle("Spectre — comparaison du pipeline spectral sur Q̂ brut (haut) vs Ĥ carré de Hadamard (bas)", fontsize=12)
fig_cmp.tight_layout(rect=[0, 0, 1, 0.96]); fig_cmp.savefig(os.path.join(OUT, "compare_QH.png"), dpi=140)

# (5) Top features par charge AVEC NOMS (données réelles labo) + fichier texte
fig_top = None
if FEATURE_NAMES is not None:
    topN = min(30, len(order)); top_idx = order[:topN]
    names = [FEATURE_NAMES[j] for j in top_idx]; vals = loads[top_idx]
    cols_t = ["#D62728" if j in sel_set else "#AAAAAA" for j in top_idx]
    fig_top, axt = plt.subplots(figsize=(8.5, max(5, topN*0.3)))
    ypos = np.arange(topN)[::-1]
    axt.barh(ypos, vals, color=cols_t)
    axt.set_yticks(ypos); axt.set_yticklabels(names, fontsize=7)
    axt.set_xlabel("charge ĉ_j"); axt.grid(alpha=0.25, axis="x")
    axt.set_title(f"{DATA} — top {topN} features par charge\n(rouge = sélectionnée, r̂={rhat}, |S|={r0['n_sel']})", fontsize=10)
    fig_top.tight_layout(); fig_top.savefig(os.path.join(OUT, "top_features.png"), dpi=140)
    with open(os.path.join(OUT, "selected_features.txt"), "w") as fh:
        for rk, j in enumerate(order):
            if j in sel_set:
                fh.write(f"{rk+1}\t{FEATURE_NAMES[j]}\t{loads[j]:.5f}\n")

# (6) PDF combiné + résumé
with PdfPages(os.path.join(OUT, "analyse_spectre.pdf")) as pdf:
    for f in (fig_roc, fig_spec, fig_load, fig_mat, fig_cmp):
        pdf.savefig(f)
    if fig_top is not None:
        pdf.savefig(fig_top)
with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write(f"Spectre/Hadamard — {TAG}\n")
    fh.write(f"{'AUC moyenne' if HAS_VAL else 'AUC CV (leaky)'} = {AUC_m:.4f} ± {AUC_s:.4f}\n")
    if NESTED is not None:
        fh.write(f"AUC nested-CV (SANS FUITE) = {NESTED[0]:.4f} ± {NESTED[1]:.4f}  "
                 f"folds={['%.3f'%a for a in NESTED[2]]}  n_sel/fold={NESTED[3]}\n")
    fh.write(f"FDP moyen   = {FDP_m:.4f} ± {FDP_s:.4f}\n" if HAS_TRUTH else "FDP        = N/A (pas de vérité terrain)\n")
    fh.write(f"r_hat moyen = {RHAT_m:.2f}\n")
    fh.write(f"rho_H moyen = {RHO_m:.4f}\n")
    fh.write(f"n_sel moyen = {NSEL_m:.2f}\n")
    aq0 = RES[0]["aQ"]
    fh.write(f"[comparaison Q̂ brut, seed0] rhat={aq0['rhat']} rho={aq0['rho']:.4f} "
             f"n_sel={aq0['n_sel']} AUC={RES[0]['auc_Q']:.4f}\n")
    fh.write(f"valeurs propres (seed 0, top {min(args.rmax+5, len(w0))}) : "
             + " ".join(f"{v:.4f}" for v in w0[:args.rmax+5]) + "\n")
    for s, r in zip(SEEDS, RES):
        fh.write(f"  seed {s}: rhat={r['rhat']} rho={r['rho']:.3f} n_sel={r['n_sel']} "
                 + (f"true={r['n_true']}/{KTRUE} FDP={r['fdp']:.3f} " if HAS_TRUTH else "") + f"AUC={r['auc']:.3f}\n")
print(f"\nFigures + analyse_spectre.pdf + summary.txt dans : {OUT}")
