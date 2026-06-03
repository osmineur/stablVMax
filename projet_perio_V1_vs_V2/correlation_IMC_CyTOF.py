"""
correlation_IMC_CyTOF.py
========================
Corrélation croisée entre les features IMC (gencive, sélectionnées par STABL/modèles pénalisés)
et les features CyTOF (sang, Analysis 2025-04-28).

Itère sur tous les runs et tous les modèles définis dans params.json.

Usage :
  python3 correlation_IMC_CyTOF.py                        # cherche params.json dans le dossier courant
  python3 correlation_IMC_CyTOF.py --params ./params.json

Sorties dans correlation_results/<run_name>/<model_name>/ :
  ranking.csv              toutes les paires (r, p, FDR), triées par |r|
  top_pairs_barplot.pdf    top 30 paires
  heatmap_<layer>.pdf      matrice IMC × CyTOF par layer
  network.pdf              réseau de corrélations
"""

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ── Paramètres ─────────────────────────────────────────────────────────────────

ANALYSIS_DIR       = Path(__file__).parent / "Analysis 2025-04-28" / "Perio2_penalized_by_effectsize"
OUT_ROOT           = Path(__file__).parent / "correlation_results"

FRACTION_THRESHOLD = 0.5    # fraction_folds min pour retenir une feature IMC
FDR_THRESHOLD      = 0.05
ABS_R_THRESHOLD    = 0.5
TOP_N_PAIRS        = 30

ALL_MODELS = [
    "STABL Lasso", "STABL ALasso", "STABL ElasticNet",
    "Lasso", "ALasso", "ElasticNet", "XGBoost",
]

# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_imc(data_path: Path) -> pd.DataFrame:
    density  = pd.read_csv(data_path / "PerioII_PatientFeature_03052026_IMCdensity.csv",  index_col=0) * 1e6
    function = pd.read_csv(data_path / "PerioII_PatientFeature_03052026_IMCfunction.csv", index_col=0)
    neighbor = pd.read_csv(data_path / "PerioII_PatientFeature_03052026_IMCneighbor.csv", index_col=0)
    return pd.concat([density, function, neighbor], axis=1)


def _load_cytof() -> dict:
    files = {
        "Frequencies": "PerioPhase2_frequencies_BL.csv",
        "Unstim":      "PerioPhase2_functional_BL_Unstim.csv",
        "LPS":         "PerioPhase2_functional_BL_LPS.csv",
        "IL12":        "PerioPhase2_functional_BL_IL12.csv",
        "IL246_TNFa":  "PerioPhase2_functional_BL_IL246.csv",
    }
    return {
        name: pd.read_csv(ANALYSIS_DIR / fname, index_col=0)
        for name, fname in files.items()
        if (ANALYSIS_DIR / fname).exists()
    }


def _get_features_for_model(stabl_dir: Path, model: str) -> pd.DataFrame:
    """
    Retourne les features sélectionnées par un modèle dans un run.
    Pour les modèles STABL : filtre par fraction_folds >= FRACTION_THRESHOLD.
    Pour les modèles pénalisés classiques (Lasso, ALasso, EN) : retourne toutes
    les features sélectionnées au moins une fois (fraction_folds > 0).
    """
    path = stabl_dir / f"Selected Features {model}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "fraction_folds"])
    df = pd.read_csv(path)
    if "fraction_folds" not in df.columns:
        # Ancien format avec juste une liste de features
        df["fraction_folds"] = 1.0
    threshold = FRACTION_THRESHOLD if "STABL" in model else 0.0
    return df[df["fraction_folds"] > threshold][["feature", "fraction_folds"]].reset_index(drop=True)


# ── Corrélation ────────────────────────────────────────────────────────────────

def _spearman_matrix(X_imc: pd.DataFrame, X_cytof: pd.DataFrame):
    r_vals = np.empty((len(X_imc.columns), len(X_cytof.columns)))
    p_vals = np.empty_like(r_vals)
    for i, ic in enumerate(X_imc.columns):
        for j, cc in enumerate(X_cytof.columns):
            mask = X_imc[ic].notna() & X_cytof[cc].notna()
            if mask.sum() < 5:
                r_vals[i, j] = p_vals[i, j] = np.nan
            else:
                r, p = spearmanr(X_imc.loc[mask, ic], X_cytof.loc[mask, cc])
                r_vals[i, j], p_vals[i, j] = r, p
    df_r = pd.DataFrame(r_vals, index=X_imc.columns, columns=X_cytof.columns)
    df_p = pd.DataFrame(p_vals, index=X_imc.columns, columns=X_cytof.columns)
    return df_r, df_p


def _fdr_correct(df_p: pd.DataFrame) -> pd.DataFrame:
    flat = df_p.values.flatten()
    nan_mask = np.isnan(flat)
    fdr_flat = np.full_like(flat, np.nan)
    if (~nan_mask).sum() > 0:
        _, fdr_vals, _, _ = multipletests(flat[~nan_mask], method="fdr_bh")
        fdr_flat[~nan_mask] = fdr_vals
    return pd.DataFrame(fdr_flat.reshape(df_p.shape), index=df_p.index, columns=df_p.columns)


def _build_ranking(df_r, df_p, df_fdr, cytof_dict) -> pd.DataFrame:
    records = []
    for ic in df_r.index:
        for cc in df_r.columns:
            r_val = df_r.loc[ic, cc]
            if np.isnan(r_val):
                continue
            cytof_source = next((k for k, df in cytof_dict.items() if cc in df.columns), "unknown")
            records.append({
                "imc_feature":   ic,
                "cytof_feature": cc,
                "cytof_source":  cytof_source,
                "r":             round(r_val, 4),
                "abs_r":         round(abs(r_val), 4),
                "p_value":       round(df_p.loc[ic, cc], 6),
                "fdr":           round(df_fdr.loc[ic, cc], 6),
                "significant":   df_fdr.loc[ic, cc] < FDR_THRESHOLD,
            })
    return pd.DataFrame(records).sort_values("abs_r", ascending=False).reset_index(drop=True)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_heatmap(df_r, df_fdr, cytof_name, out_path, n_patients, model, run_name):
    if df_r.empty:
        return
    row_order = df_r.abs().max(axis=1).sort_values(ascending=False).index
    col_order = df_r.abs().max(axis=0).sort_values(ascending=False).index[:40]
    R = df_r.loc[row_order, col_order]
    F = df_fdr.loc[row_order, col_order]

    fig_h = max(4, len(row_order) * 0.4)
    fig_w = max(6, len(col_order) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(R.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            v = F.iloc[i, j]
            if not np.isnan(v) and v < FDR_THRESHOLD:
                ax.text(j, i, "*", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(col_order))); ax.set_xticklabels(col_order, rotation=90, fontsize=5)
    ax.set_yticks(range(len(row_order))); ax.set_yticklabels(row_order, fontsize=6)
    plt.colorbar(im, ax=ax, label="Spearman r", fraction=0.02, pad=0.02)
    ax.set_title(f"[{run_name} / {model}] IMC × CyTOF {cytof_name}\n(n={n_patients}, * FDR<{FDR_THRESHOLD}, top-40 CyTOF)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_pairs_barplot(df_ranking, out_path, model, run_name):
    top = df_ranking.head(TOP_N_PAIRS).copy()
    if top.empty:
        return
    colors = ["#C41E3A" if r > 0 else "#001A7B" for r in top["r"]]
    labels = [f"{row.imc_feature}  ×  {row.cytof_feature}" for _, row in top.iterrows()]
    fig, ax = plt.subplots(figsize=(10, max(5, TOP_N_PAIRS * 0.28)))
    bars = ax.barh(range(len(top)), top["abs_r"], color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("|Spearman r|")
    ax.set_title(f"[{run_name} / {model}] Top {TOP_N_PAIRS} paires IMC–CyTOF par |r|")
    ax.axvline(ABS_R_THRESHOLD, color="gray", ls="--", lw=0.8)
    for i, (bar, row) in enumerate(zip(bars, top.itertuples())):
        sig = "**" if row.fdr < 0.01 else ("*" if row.fdr < FDR_THRESHOLD else "")
        ax.text(bar.get_width() + 0.01, i, sig, va="center", fontsize=7)
    pos_patch = mpatches.Patch(color="#C41E3A", alpha=0.8, label="r > 0")
    neg_patch = mpatches.Patch(color="#001A7B", alpha=0.8, label="r < 0")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_network(df_ranking, out_path, model, run_name):
    sig = df_ranking[(df_ranking["abs_r"] >= ABS_R_THRESHOLD) & (df_ranking["fdr"] < FDR_THRESHOLD)]
    label = f"|r|≥{ABS_R_THRESHOLD}, FDR<{FDR_THRESHOLD}"
    if sig.empty:
        # Seuil relaxé si rien de significatif (n faible)
        sig = df_ranking[(df_ranking["abs_r"] >= 0.4)].head(60)
        label = f"|r|≥0.4 (relaxed, n faible)"
    if sig.empty:
        return

    G = nx.Graph()
    for n in sig["imc_feature"].unique():
        G.add_node(n, layer="imc")
    for n in sig["cytof_feature"].unique():
        G.add_node(n, layer="cytof")
    for _, row in sig.iterrows():
        G.add_edge(row["imc_feature"], row["cytof_feature"],
                   weight=row["abs_r"], sign=int(np.sign(row["r"])))

    pos = nx.spring_layout(G, seed=42, k=2.5)
    node_colors = ["#C41E3A" if G.nodes[n]["layer"] == "imc" else "#4A90D9" for n in G.nodes()]
    edge_colors = ["#C41E3A" if G[u][v]["sign"] > 0 else "#001A7B" for u, v in G.edges()]
    edge_widths = [G[u][v]["weight"] * 3 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=300, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5.5, font_weight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#C41E3A", label="Feature IMC (gencive)"),
        mpatches.Patch(color="#4A90D9", label="Feature CyTOF (sang)"),
        mpatches.Patch(color="#C41E3A", alpha=0.6, label="r > 0"),
        mpatches.Patch(color="#001A7B", alpha=0.6, label="r < 0"),
    ], fontsize=9, loc="upper left")
    ax.set_title(
        f"[{run_name} / {model}] Réseau IMC–CyTOF ({label})\n"
        f"{sum(1 for n in G.nodes() if G.nodes[n]['layer']=='imc')} IMC — "
        f"{sum(1 for n in G.nodes() if G.nodes[n]['layer']=='cytof')} CyTOF — "
        f"{len(G.edges())} liens"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Analyse pour un (run, model) ───────────────────────────────────────────────

def run_correlation(run_name, save_path, data_path, model, X_imc_full, cytof_dict, cytof_all):
    stabl_dir = Path(save_path) / "Training CV"
    feat_df   = _get_features_for_model(stabl_dir, model)

    if feat_df.empty:
        print(f"    [{run_name} / {model}] Aucune feature sélectionnée, ignoré.")
        return None

    imc_feats = [f for f in feat_df["feature"] if f in X_imc_full.columns]
    if not imc_feats:
        print(f"    [{run_name} / {model}] Features introuvables dans l'IMC brut, ignoré.")
        return None

    common = X_imc_full.index.intersection(cytof_all.index)
    X_imc   = X_imc_full.loc[common, imc_feats]
    X_cytof = cytof_all.loc[common]

    df_r, df_p = _spearman_matrix(X_imc, X_cytof)
    df_fdr     = _fdr_correct(df_p)
    df_ranking = _build_ranking(df_r, df_p, df_fdr, cytof_dict)

    out_dir = OUT_ROOT / run_name / model.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ranking.to_csv(out_dir / "ranking.csv", index=False)

    n_sig = df_ranking["significant"].sum()
    top_r = df_ranking["abs_r"].iloc[0] if not df_ranking.empty else 0
    print(f"    [{run_name} / {model}] {len(imc_feats)} features IMC, "
          f"{len(df_ranking)} paires, {n_sig} FDR<{FDR_THRESHOLD}, top |r|={top_r:.3f}")

    plot_top_pairs_barplot(df_ranking, out_dir / "top_pairs_barplot.pdf", model, run_name)
    for cytof_name, df_layer in cytof_dict.items():
        cols = [c for c in df_layer.columns if c in df_r.columns]
        if cols:
            plot_heatmap(df_r[cols], df_fdr[cols], cytof_name,
                         out_dir / f"heatmap_{cytof_name}.pdf",
                         len(common), model, run_name)
    plot_network(df_ranking, out_dir / "network.pdf", model, run_name)

    return {
        "run": run_name, "model": model,
        "n_imc_features": len(imc_feats),
        "n_pairs": len(df_ranking),
        "n_significant": n_sig,
        "top_abs_r": round(top_r, 4),
        "top_imc_feature":   df_ranking["imc_feature"].iloc[0]   if not df_ranking.empty else "",
        "top_cytof_feature": df_ranking["cytof_feature"].iloc[0] if not df_ranking.empty else "",
        "top_cytof_source":  df_ranking["cytof_source"].iloc[0]  if not df_ranking.empty else "",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="./params.json")
    args = parser.parse_args()

    params_path = Path(args.params)
    with open(params_path) as f:
        params = json.load(f)

    data_path = Path(params["data_path"])
    # Chemin relatif → absolu par rapport au dossier de params.json
    if not data_path.is_absolute():
        data_path = params_path.parent / data_path

    print("=== Chargement des données communes ===")
    X_imc_full = _load_imc(data_path)
    cytof_dict = _load_cytof()
    cytof_all  = pd.concat(cytof_dict.values(), axis=1)
    common     = X_imc_full.index.intersection(cytof_all.index)
    print(f"  IMC : {X_imc_full.shape}")
    print(f"  CyTOF : {cytof_all.shape[1]} features ({', '.join(f'{k}:{v.shape[1]}' for k,v in cytof_dict.items())})")
    print(f"  Patients communs : {len(common)}")

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for run in params["runs"]:
        run_name  = run["name"]
        save_path = run["save_path"]
        if not Path(save_path).is_absolute():
            save_path = params_path.parent / save_path

        print(f"\n{'='*60}")
        print(f"RUN : {run_name}  ({save_path})")
        print(f"{'='*60}")

        for model in ALL_MODELS:
            row = run_correlation(run_name, save_path, data_path,
                                  model, X_imc_full, cytof_dict, cytof_all)
            if row:
                summary_rows.append(row)

    # Tableau récapitulatif global
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).sort_values("top_abs_r", ascending=False)
        df_summary.to_csv(OUT_ROOT / "summary.csv", index=False)
        print(f"\n{'='*60}")
        print("RÉSUMÉ GLOBAL (trié par top |r|)")
        print(df_summary.to_string(index=False))
        print(f"\nSummary → {OUT_ROOT}/summary.csv")

    print(f"\nTerminé — résultats dans {OUT_ROOT}/")


if __name__ == "__main__":
    main()
