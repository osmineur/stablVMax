import os
from pathlib import Path
from warnings import warn
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from knockpy.knockoffs import GaussianSampler
from knockpy.utilities import shift_until_PSD,calc_mineig
from scipy.stats import rankdata
from scipy.stats import norm as scipy_norm
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectorMixin, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.model_selection import ParameterGrid,  GroupShuffleSplit
from sklearn.utils import safe_mask
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted
from tqdm.autonotebook import tqdm
from .unionfind import UnionFind
import warnings
from .utils import auto_mode_lambda_grid
from .visualization import boxplot_features, scatterplot_features

from copy import deepcopy


def gaussianize(X):
    """Gaussianization par transformation rang-normal (Blom 1958).

    Pour chaque feature j : g_j = Phi^{-1}( rank(X_j) / (n+1) )

    Cette transformation réduit d_TV(L(X), N(mu, Sigma)) vers 0,
    ce qui resserre la borne de l'erreur FDP+ :
        E[FDP(t)] <= E[FDP+(t)] + 2*|H0|*eps / E[D(t)]
    En pratique, sur des données biologiques non-gaussiennes (protéomique,
    métabolomique), la gaussianisation permet à GaussianSampler2 de
    générer des knockoffs plus valides, renforçant la garantie théorique.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Données brutes.

    Returns
    -------
    X_gauss : np.ndarray, shape (n, p)
        Données transformées, chaque colonne ~ N(0,1) marginal.
    """
    n = X.shape[0]
    X_gauss = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        ranks = rankdata(X[:, j], method='average')
        X_gauss[:, j] = scipy_norm.ppf(ranks / (n + 1))
    return X_gauss

class GaussianSampler2(GaussianSampler):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.Lk,self.mu_k = None, None
        self.prime_knockoffs()
    
    def prime_knockoffs(self):

        # Calculate MX knockoff moments...
        n, p = self.X.shape
        invSigma_S = np.dot(self.invSigma, self.S)
        mu_k = self.X - np.dot(self.X - self.mu.reshape(1, -1), invSigma_S)  # This is a bottleneck??
        Vk = 2 * self.S - np.dot(self.S, invSigma_S)

        try:
            Lk = np.linalg.cholesky(Vk)
        except np.linalg.LinAlgError:
            min_eig = calc_mineig(Vk)
            warnings.warn(
                f"Minimum eigenvalue of Vk is {min_eig}, cholesky decomp failed, FDR violations possible"
            )
            Vk = shift_until_PSD(Vk, self.sample_tol)
            Lk = np.linalg.cholesky(Vk)

        self.Lk = Lk
        self.mu_k = np.expand_dims(mu_k,axis=2)
        
    def get_core(self):
        return self.Lk, self.mu_k

    def sample_knockoffs(self, random_state= None,check_psd=False):
        """ Samples knockoffs. returns n x p knockoff matrix.

        Parameters
        ----------
        check_psd : bool
            If True, will check and enforce that S is a valid S-matrix.
            Defalts to False.
        """
        if check_psd: 
            self.gs.check_PSD_condition(self.Sigma, self.S)


        n,p = self.X.shape
        if random_state is not None:
            rng = np.random.default_rng(random_state)
            kf = np.dot(self.Lk, rng.standard_normal((n,p,1)))
        else:
            kf = np.dot(self.Lk, np.random.randn(n,p, 1))
        kf = np.transpose(kf, [1, 0, 2])
        kf = kf + self.mu_k
        return kf[:, :, 0]

def classic_bootstrap(y, n_subsamples, replace=True, class_weight=None, rng=np.random.default_rng(None), **kwargs):
    """Function to create a bootstrap sample from the original dataset.
    Weights can be used to make some samples more likely to be selected.

    Parameters
    ----------
    y : array-like, shape(n_repeats, )
        The outcome array for classification or regression

    n_subsamples : int
        The number of subsamples indices returned by the bootstrap 

    replace : bool, default=True
        Whether to replace samples when bootstrapping

    class_weight: str or dict or None, default=None
        This is the sampling weights used in the bootstrap process
            - If None, no weights are used.
            - If 'balanced', the weights are automatically computed so that
            the weights are balanced and the probabilities of sampling different
            classes are adjusted.
            - Can also be a dictionary of this format {class1:value1, class2:value2}
            values are weights not probabilities. They will automatically be converted
            into probabilities.

    rng: np.random.default_rng, default=np.random.default_rng(None)
        RandomState generator

    Returns
    -------
    sampled_indices : array-like, shape(n_subsamples, )
        Sampled indices

    """
    n_samples = y.shape[0]

    if n_subsamples > n_samples and replace is False:
        raise ValueError("When `replace` is set to False, n_subsamples cannot be greater than the "
                         f"number of samples in the original dataset. Got `n_repeats`={n_samples} "
                         f"and `n_subsamples`={n_subsamples}")

    if class_weight is not None:
        samples_weight = compute_sample_weight(class_weight, y)
        sampling_probs = samples_weight / samples_weight.sum()

    else:
        sampling_probs = None

    sampled_indices = rng.choice(
        a=n_samples,
        size=n_subsamples,
        replace=replace,
        p=sampling_probs
    )

    # Handling the case of binary classification where we only select one class
    if len(np.unique(y[sampled_indices])) < 2:
        sampled_indices = classic_bootstrap(
            y,
            n_subsamples,
            replace=replace,
            class_weight=class_weight,
            rng=rng
        )

    return sampled_indices


def group_bootstrap(y, n_subsamples, groups, replace=False, rng=np.random.RandomState(None), **kwargs):
    """Function to create a bootstrap sample from the original dataset.
    Weights can be used to make some samples more likely to be selected.

    Parameters
    ----------
    y : array-like, shape(n_repeats, )
        The outcome array for classification or regression

    n_subsamples : int
        The number of subsamples indices returned by the bootstrap 

    replace : bool, default=True
        Whether to replace samples when bootstrapping

    class_weight: str or dict or None, default=None
        This is the sampling weights used in the bootstrap process
            - If None, no weights are used.
            - If 'balanced', the weights are automatically computed so that
            the weights are balanced and the probabilities of sampling different
            classes are adjusted.
            - Can also be a dictionary of this format {class1:value1, class2:value2}
            values are weights not probabilities. They will automatically be converted
            into probabilities.

    rng: np.random.default_rng, default=np.random.default_rng(None)
        RandomState generator

    Returns
    -------
    sampled_indices : array-like, shape(n_subsamples, )
        Sampled indices

    """
    n_samples = y.shape[0]

    if n_subsamples > n_samples and replace is False:
        raise ValueError("When `replace` is set to False, n_subsamples cannot be greater than the "
                         f"number of samples in the original dataset. Got `n_repeats`={n_samples} "
                         f"and `n_subsamples`={n_subsamples}")

    subsample_prop = n_subsamples / n_samples

    sampled_indices = GroupShuffleSplit(n_splits=1, train_size=subsample_prop, random_state=rng).split(y, groups=groups)
    sampled_indices = next(sampled_indices)[0]
    # Handling the case of binary classification where we only select one class
    if len(np.unique(y[sampled_indices])) < 2:
        sampled_indices = group_bootstrap(
            y,
            n_subsamples,
            groups=groups,
            replace=replace,
            rng=rng
        )

    return sampled_indices


def _bootstrap_generator(
        n_bootstraps,
        bootstrap_func,
        y,
        n_subsamples,
        replace,
        random_state=None,
        **kwargs
):
    """Function that creates bootstrapped indices, used in the Stabl process.
    The function returns a generator containing the indices for each bootstrap.

    Parameters
    ----------
    n_bootstraps: int
        Number of bootstraps for each value of the lambda parameter.

    bootstrap_func: python function
        The function use to draw the indices. 
        Should have at least the following parameters:
            - y: target array
            - n_subsamples: number of samples to draw from the original data set
            - replace: boolean indicating if we want to replace the samples 

    y: array-like, size(n_repeats, )
        Targets

    n_subsamples: int
        number of samples to draw from the original data set

    replace: bool
        If set to True, the bootstrap will be done such that the samples are 
        replaced during the process.

    random_state: int,
        Random state for reproducibility.

    **kwargs: arguments
        Further arguments we want to pass to bootstrap_func.
    """
    rng = np.random.RandomState(random_state)
    subsamples = []
    for _ in range(n_bootstraps):

        # Generating the bootstrapped indices
        subsample = bootstrap_func(
            y=y,
            n_subsamples=n_subsamples,
            replace=replace,
            rng=rng,
            **kwargs
        )
        subsamples.append(subsample)
    return subsamples


def export_stabl_to_csv(stabl, path):
    """
    Export Stabl scores to csv. They can later be used to plot the stabl path again.

    Parameters
    ----------
    stabl: Stabl
        Fitted Stabl instance.

    path: str or Path
        The path where csv files will be saved

    Returns
    -------
    None
    """

    check_is_fitted(stabl, 'stabl_scores_')

    if hasattr(stabl, 'feature_names_in_'):
        X_columns = stabl.feature_names_in_
    else:
        X_columns = [f'x.{i + 1}' for i in range(stabl.n_features_in_)]

    columns = list(ParameterGrid(stabl.fitted_lambda_grid_))

    df_real = pd.DataFrame(data=stabl.stabl_scores_,
                           index=X_columns, columns=columns)
    df_real.to_csv(Path(path, 'STABL scores.csv'))

    df_max_probs = pd.DataFrame(
        data={"Max Proba": stabl.stabl_scores_.max(axis=1)},
        index=X_columns
    )
    df_max_probs = df_max_probs.sort_values(by='Max Proba', ascending=False)
    df_max_probs.to_csv(Path(path, 'Max STABL scores.csv'))

    if stabl.artificial_type is not None:
        synthetic_index = [f'artificial.{i+1}' for i in range(stabl.stabl_scores_artificial_.shape[0])]

        df_noise = pd.DataFrame(
            data=stabl.stabl_scores_artificial_,
            index=synthetic_index,
            columns=columns
        )
        df_noise.to_csv(Path(path, 'STABL artificial scores.csv'))

        df_max_probs_noise = pd.DataFrame(
            data={"Max Proba": stabl.stabl_scores_artificial_.max(axis=1)},
            index=synthetic_index
        )
        df_max_probs_noise = df_max_probs_noise.sort_values(
            by='Max Proba', ascending=False)
        df_max_probs_noise.to_csv(
            Path(path, 'Max STABL artificial scores.csv'))


def plot_fdr_graph(
        stabl,
        show_fig=True,
        export_file=False,
        path='./FDR estimate graph.pdf',
        figsize=(8, 4)
):
    """
    Plots the FDR graph.
    The user can also export it to pdf of other format

    Parameters
    ----------
    stabl : Stabl
        Fitted Stabl instance.

    show_fig : bool, default=True
        Whether to display the figure

    export_file: bool, default=False
        If set to True, it will export the plot using the path

    path: str or Path
        Should be the string of the path/name. Use name of the file plus extension

    figsize: tuple
        Size of the Stabl fdr graph

    Returns
    -------
    figure, axis
    """

    check_is_fitted(stabl, 'stabl_scores_')

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    thresh_grid  = np.array(stabl.fdr_threshold_range)
    FDPs         = np.array(stabl.FDRs_)
    max_scores   = np.max(stabl.stabl_scores_, axis=1)

    # ── FDP+(t) curve ────────────────────────────────────────────────────────
    ax.plot(thresh_grid, FDPs, color="#4D4F53", label='FDP+(t)', lw=2)

    # ── Zone frontière Maurer-Pontil (mode constrained uniquement) ───────────
    mode = getattr(stabl, 'selection_mode', None)
    if mode == "constrained" and hasattr(stabl, 'eps_B_total_fw_') and stabl.eps_B_total_fw_ is not None:
        eps_tot_j    = stabl.eps_B_total_fw_
        bdry_density = np.array([
            np.sum((max_scores > t) & (max_scores <= t + eps_tot_j)) / max(1, int(np.sum(max_scores > t)))
            for t in thresh_grid
        ])
        ylo, yhi = FDPs.min(), FDPs.max()
        pad = (yhi - ylo) * 0.12 if yhi > ylo else 0.05
        ax2 = ax.twinx()
        ax2.fill_between(thresh_grid, 0, bdry_density, alpha=0.08, color="#2CA02C",
                         step="post", label="|∂⁺(t)|/D(t)")
        ax2.set_ylabel("|∂⁺(t)|/D(t)", fontsize=8, color="#2CA02C")
        ax2.tick_params(axis='y', labelcolor="#2CA02C", labelsize=7)
        ax2.spines['right'].set_visible(True)
        ax.set_ylim(ylo - pad, yhi + pad)

    # ── t* (threshold sélectionné) ───────────────────────────────────────────
    t_star      = stabl.fdr_min_threshold_
    constrained = getattr(stabl, 'constrained_threshold_', False)
    margin      = getattr(stabl, 'min_margin_fw_', None)
    if margin is not None: margin = float(margin)

    if stabl.min_fdr_ > 1:
        t_star_label = "No features selected"
    elif mode == "wj":
        n_s1 = int(np.sum(stabl.w_paired_ > stabl.eps_paired_)) if stabl.w_paired_ is not None else 0
        t_star_label = f"WJ: |S₁|={n_s1} (FDP=0 w.p.≥1-δ) | t* ref={t_star:.3f}"
    else:
        suffix = "exact" if constrained else "fallback"
        margin_str = f", margin={margin:.4f}" if margin is not None else ""
        t_star_label = f"t*={t_star:.3f} ({suffix}{margin_str})"

    ax.axvline(t_star, ls='--', lw=2, color="#C41E3A", label=t_star_label)

    # ── Unconstrained argmin (référence grisée, modes constrained/wj) ────────
    if stabl.min_fdr_ <= 1 and mode in ("constrained", "wj"):
        t_unc = float(thresh_grid[np.argmin(FDPs)])
        if abs(t_unc - t_star) > 1e-6:
            ax.axvline(t_unc, ls=':', lw=1.2, color="gray",
                       label=f"t* unconstrained={t_unc:.3f}")

    ax.set_xlabel('Threshold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=8)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    if export_file:
        fig.savefig(path, dpi=95)

    if not show_fig:
        plt.close()

    return fig, ax


def plot_fdr_graph_table(
        stabl,
        show_fig=True,
        export_file=False,
        path='./FDR table estimate graph.pdf',
        figsize=(8, 4)
):
    """
    Plots the FDR graph for all lambda.
    The user can also export it to pdf of other format

    Parameters
    ----------
    stabl : Stabl
        Fitted Stabl instance.

    show_fig : bool, default=True
        Whether to display the figure

    export_file: bool, default=False
        If set to True, it will export the plot using the path

    path: str or Path
        Should be the string of the path/name. Use name of the file plus extension

    figsize: tuple
        Size of the Stabl fdr graph

    Returns
    -------
    figure, axis
    """

    check_is_fitted(stabl, 'stabl_scores_')

    def dict_format(d, form="{:6.3f}"):
        if not isinstance(form, dict):
            form = {k: form for k in d.keys()}
        res = "{"
        for k, v in d.items():
            res += k + ":" + form[k].format(v)
            res += ", "
        res = res[:-2]
        res += "}"
        return res

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    thresh_grid      = np.array(stabl.fdr_threshold_range)
    FDPs             = np.array(stabl.FDRs_)
    max_scores       = np.max(stabl.stabl_scores_, axis=1)
    lambda_grid_list = list(ParameterGrid(stabl.fitted_lambda_grid_))

    for i, l in enumerate(lambda_grid_list):
        ax.plot(thresh_grid, stabl.fdrs_table[i], label=None, lw=0.5)

    ax.plot(thresh_grid, FDPs, color="#4D4F53", label='FDP+(t)', lw=2)

    # ── Zone frontière (mode constrained uniquement) ──────────────────────────
    mode_tbl = getattr(stabl, 'selection_mode', None)
    if mode_tbl == "constrained" and hasattr(stabl, 'eps_B_total_fw_') and stabl.eps_B_total_fw_ is not None:
        eps_tot_j   = stabl.eps_B_total_fw_
        bdry_density = np.array([
            np.sum((max_scores > t) & (max_scores <= t + eps_tot_j)) / max(1, int(np.sum(max_scores > t)))
            for t in thresh_grid
        ])
        ylo, yhi = FDPs.min(), FDPs.max()
        pad = (yhi - ylo) * 0.12 if yhi > ylo else 0.05
        ax2 = ax.twinx()
        ax2.fill_between(thresh_grid, 0, bdry_density, alpha=0.08, color="#2CA02C",
                         step="post", label="|∂⁺(t)|/D(t)")
        ax2.set_ylabel("|∂⁺(t)|/D(t)", fontsize=7, color="#2CA02C")
        ax2.tick_params(axis='y', labelcolor="#2CA02C", labelsize=6)
        ax.set_ylim(ylo - pad, yhi + pad)

    # ── t* ────────────────────────────────────────────────────────────────────
    t_star      = stabl.fdr_min_threshold_
    constrained = getattr(stabl, 'constrained_threshold_', False)
    suffix      = "exact" if constrained else "fallback"
    if stabl.min_fdr_ > 1:
        t_star_label = "No features selected"
    elif mode_tbl == "wj":
        n_s1 = int(np.sum(stabl.w_paired_ > stabl.eps_paired_)) if stabl.w_paired_ is not None else 0
        t_star_label = f"WJ: |S₁|={n_s1} | t* ref={t_star:.3f}"
    else:
        t_star_label = f"t*={t_star:.3f} ({suffix})"
    ax.axvline(t_star, ls='--', lw=2, color="#C41E3A", label=t_star_label)

    # ── Per-lambda argmin (référence) ─────────────────────────────────────────
    argmin_table          = np.unravel_index(np.argmin(stabl.fdrs_table), stabl.fdrs_table.shape)
    table_optimal_threshold = thresh_grid[argmin_table[1]]
    selected_lambda_grid  = dict_format(lambda_grid_list[argmin_table[0]])
    ax.axvline(table_optimal_threshold, ls='--', lw=1.5, color="#e7a5b0",
               label=f"per-λ argmin={table_optimal_threshold:.2f}; {selected_lambda_grid}")

    # ── Unconstrained argmin (référence grisée) ───────────────────────────────
    t_unc = float(thresh_grid[np.argmin(FDPs)])
    if abs(t_unc - t_star) > 1e-6:
        ax.axvline(t_unc, ls=':', lw=1.2, color="gray",
                   label=f"t* unconstrained={t_unc:.3f}")

    ax.set_xlabel('Threshold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=7)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    if export_file:
        fig.savefig(path, dpi=95)

    if not show_fig:
        plt.close()

    return fig, ax


def plot_stabl_path(
        stabl,
        new_hard_threshold=None,
        show_fig=True,
        export_file=False,
        path='./Stabl path.pdf',
        figsize=(4, 8)
):
    """Plots Stabl path.
    The user can also export it to pdf or any other format

    Parameters
    ----------
    stabl: Stabl
        Fitted Stabl instance.

    new_hard_threshold: float or None, default=None
        Threshold defining the minimum cutoff value for the
        stabl scores. This is a hard threshold: FDR control
        will be ignored if this is not None.

    show_fig: bool, default=True
        Whether to display the figure

    export_file: bool
        If set to True, it will export the plot using the path

    path: str or Path
        Should be the string of the path/name. Use name of the file plus extension

    figsize: tuple
        Size of the STABL path

    Returns
    -------
    figure, axis
    """

    check_is_fitted(stabl, 'stabl_scores_')

    threshold = stabl.hard_threshold if new_hard_threshold is None else new_hard_threshold

    if isinstance(threshold, float) and not (0.0 < threshold <= 1):
        raise ValueError(
            f'If new_hard_threshold is set, it must be a float in (0, 1], got {threshold}')

    paths_to_highlight = stabl.get_support(new_hard_threshold=threshold)

    x_grid_list = []
    x_padding_list = []
    order_list = []
    different_params = stabl.get_different_parameters()
    nb_different_params = len(different_params)
    if nb_different_params <= 1:
        if 'alpha' in stabl.fitted_lambda_grid_:
            x_grid_tmp = 1.0 / np.array(stabl.fitted_lambda_grid_["alpha"])
            order_list = [np.arange(len(stabl.fitted_lambda_grid_["alpha"]))]
            x_grid_list = [x_grid_tmp]
            x_padding_list = [0]
        elif 'C' in stabl.fitted_lambda_grid_:
            x_grid_tmp = np.array(stabl.fitted_lambda_grid_["C"])
            order_list = [np.arange(len(stabl.fitted_lambda_grid_["C"]))]
            x_grid_list = [x_grid_tmp]
            x_padding_list = [0]
    elif nb_different_params == 2:
        params = list(ParameterGrid(stabl.fitted_lambda_grid_))
        ordered_params = dict()
        for i, k in enumerate(params):
            l1_ratio = k["l1_ratio"]
            penalty = k["alpha"] if "alpha" in k else k["C"]
            if l1_ratio in ordered_params:
                order = ordered_params[l1_ratio][0]
                x_grid = ordered_params[l1_ratio][1]
            else:
                order = []
                x_grid = []
            order.append(i)
            x_grid.append(penalty)
            ordered_params[l1_ratio] = (order, x_grid)
        figsize = (figsize[0] * len(ordered_params.keys()), figsize[1])
        x_padding = 0
        for l1_ratio in sorted(ordered_params.keys()):
            order = np.array(ordered_params[l1_ratio][0])
            penalties = np.array(ordered_params[l1_ratio][1])
            if "alpha" in different_params:
                x_grid = 1.0 / penalties
            elif "C" in different_params:
                x_grid = penalties
            x_padding += np.max(x_grid) - np.min(x_grid) + 1e-5
            x_grid_list.append(x_grid)
            order_list.append(order)
            x_padding_list.append(x_padding)
    else:
        warnings.warn("Cannot plot the STABL path for more than 2 parameters")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x_order = []
    x_list = []

    for i, o in enumerate(order_list):
        x_grid = x_grid_list[i]
        x_padding = x_padding_list[i]
        x_grid += x_padding
        for j in x_grid:
            x_list.append(j)
        x_order.extend(o)

        if not paths_to_highlight.all():
            ax.plot(
                x_grid,
                stabl.stabl_scores_[~paths_to_highlight][:, o].T,
                alpha=1,
                lw=1.5,
                color="#4D4F53",
                label="Noisy features"
            )

        if paths_to_highlight.any():
            ax.plot(
                x_grid,
                stabl.stabl_scores_[paths_to_highlight][:, o].T,
                alpha=1,
                lw=2,
                color="#C41E3A",
                label="Stable features"
            )

        if threshold is not None:
            ax.plot(
                x_grid,
                threshold * np.ones(len(x_grid)),
                c="black",
                ls="--",
                label=f"Hard threshold={threshold: .2f}"
            )

        elif stabl.artificial_type is not None:
            ax.plot(
                x_grid,
                stabl.stabl_scores_artificial_[:, o].T,
                color="gray",
                ls=":",
                alpha=.4,
                lw=1,
                label="Artificial features"
            )

            ax.plot(
                x_grid,
                stabl.fdr_min_threshold_ * np.ones(len(x_grid)),
                c="black",
                ls="--",
                label=f"FDP+ threshold={stabl.fdr_min_threshold_: .2f}"
            )

        if stabl.explore_threshold is not None:
            ax.plot(
                x_grid,
                stabl.explore_threshold * np.ones(len(x_grid)),
                c="#487fad",
                ls="--",
                label=f"Explore threshold={stabl.explore_threshold: .2f}"
            )
        if i != len(order_list) - 1:
            x_vert_gray = np.max(x_grid)
            ax.axvline(x=x_vert_gray, c="gray", ls="--", lw=1)

    ax.set_xscale("log")
    ax.tick_params(left=True, right=False, labelleft=True,
                   labelbottom=True, bottom=True)
    ax.set_xlabel(r"$1/\lambda$")
    ax.set_ylabel(f"Frequency of selection")
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1))

    fig.tight_layout()

    if export_file:
        fig.savefig(path, dpi=95)

    if not show_fig:
        plt.close()

    return fig, ax, x_list, x_order


def save_stabl_results(
        stabl,
        path,
        df_X,
        y,
        figure_fmt='pdf',
        new_hard_threshold=None,
        task_type="binary",
        override=False,
):
    """
    Function to automatically save all the results of a Stabl fitted instance.
    The user must define the input DataFrame and the output to plot the stable
    features.

    Parameters
    ----------
    stabl: Stabl
        Must be a fitted Stabl object.

    path: Path or str
        The path where to save the results. If the path already exists an error will be raised

    df_X: pd.DataFrame, shape=(n_repeats, n_features)
        input DataFrame

    y: pd.Series, shape=(n_repeats)
        Series of output

    figure_fmt: str
        Format of the figures.

    new_hard_threshold: float or None, default=None
        Threshold defining the minimum cutoff value for the
        stability scores. This is a hard threshold: FDR control
        will be ignored if this is not None

    task_type: str, default="binary"
        Type of performed task.
        Choose "binary" for binary classification and "regression" for regression tasks or "multiclass"

    override: bool, default=False
        If True, this existing folder will be overwritten
    """

    check_is_fitted(stabl)

    path = Path(path, '')

    try:
        os.makedirs(path, exist_ok=override)
    except FileExistsError:
        raise FileExistsError(f"Folder with path={path} already exists.")

    # Saving the stability scores
    export_stabl_to_csv(stabl=stabl, path=path)

    if stabl.artificial_type is not None:
        plot_fdr_graph(
            stabl=stabl,
            show_fig=False,
            export_file=True,
            path=Path(path, f'FDR Graph.{figure_fmt}'),
            figsize=(8, 4)
        )
        plot_fdr_graph_table(
            stabl=stabl,
            show_fig=False,
            export_file=True,
            path=Path(path, f'FDR table Graph.{figure_fmt}'),
            figsize=(12, 8)
        )

    plot_stabl_path(
        stabl=stabl,
        new_hard_threshold=new_hard_threshold,
        show_fig=False,
        export_file=True,
        path=Path(path, f'Stability Path.{figure_fmt}'),
        figsize=(4, 8)
    )

    selected_features = stabl.get_feature_names_out(
        new_hard_threshold=new_hard_threshold)

    nb_selected_features = len(selected_features)
    df_selected_features = pd.DataFrame(
        data={"Feature Name": selected_features},
        index=[f"Feature n°{i + 1}" for i in range(nb_selected_features)]
    )

    Path(path, 'Selected Features').mkdir(parents=True, exist_ok=override)
    df_selected_features.to_csv(
        Path(path, "Selected Features", "Selected features.csv"))

    if task_type in ["binary", "multiclass"]:
        boxplot_features(
            features=selected_features,
            df_X=df_X,
            y=y,
            categorical_features=6,
            show_fig=False,
            export_file=True,
            path=Path(path, 'Selected Features'),
            fmt=figure_fmt
        )

    elif task_type == "regression":
        scatterplot_features(
            features=selected_features,
            df_X=df_X,
            y=y,
            categorical_features=6,
            show_fig=False,
            export_file=True,
            path=Path(path, 'Selected Features'),
            fmt=figure_fmt
        )


def fit_bootstrapped_sample(
        base_estimator,
        X,
        y,
        lambda_val,
        corr_groups=None,
        threshold=None
):
    """
    Fits base_estimator on a bootstrap sample of the original data,
    and returns a mas of the variables that are selected by the fitted model.

    Parameters
    ----------
    base_estimator: estimator
        This is the estimator to be fitted on the data

    X: {array-like, sparse matrix}, shape = [n_repeats, n_features]
        The training input samples.

    y: array-like, shape = [n_repeats]
        The target values.

    lambda_val: dict of parameters
        Penalization parameters of base_estimator

    corr_groups: array-like, default=None
        Groups of features based on the correlation matrix. It is used for sparse group lasso.

    threshold: string or float, default=None
        The hard_threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``hard_threshold`` value is
        the median (resp. the mean) of the feature importance. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the hard_threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    Returns
    -------
    selected_variables: array-like, shape=(n_features, )
        Boolean mask of the selected variables.
    """
    base_estimator.set_params(**lambda_val)
    if hasattr(base_estimator, "groups"):
        base_estimator.set_params(groups=corr_groups)
    base_estimator.fit(X, y)

    features_selection = SelectFromModel(
        estimator=base_estimator,
        threshold=threshold,
        prefit=True
    )

    return features_selection.get_support()


class Stabl(SelectorMixin, BaseEstimator):
    """In a STABL process, the estimator `base_estimator` is fitted
    several time on bootstrap samples of the original data set, for different values of
    the regularization parameter for `base_estimator`. Features that
    get selected significantly by the model in these bootstrap samples are
    considered to be stable variables. This implementation also allows the user
    to use synthetic features to automatically set the hard_threshold of selection by
    FDR control.

    Parameters
    ----------
    base_estimator: sklearn.base_estimator, default=LogisticRegression
        The base estimator used for stability selection. The estimator
        must have either a ``feature_importances_`` or ``coef_``
        attribute after fitting.

    lambda_grid: dict or "auto", default={"C": np.linspace(0.01, 1, 30)}
        Grid of values for the penalization parameter to iterate over.
        The "auto" mode works only when the base_estimator is :
        - LogisticRegression with l1 penalty (penalty='l1' or penalty='elasticnet')
        - Lasso
        - ElasticNet with l1_ratio > 0
        or an extension of these classes.

    n_lambda: int or None, default=None
        If lambda_grid is set to "auto", this is the number of lambdas to test

    n_bootstraps: int, default=1000
        Number of bootstrap iterations for each value of lambda.

    artificial_type: str or None
        If str can either be "random_permutation" or "knockoff"
        If None, we do not inject artificial features, the user must therefore define an arbitrary hard_threshold.
        When the artificial_type is none, we fall back into the classic stability selection process.

    artificial_proportion: float, default=1.0
        The proportion of artificial features to generate.

    sample_fraction: float, default=0.5
        The fraction of samples to be used in each bootstrap sample.
        Can be greater than 1 if we replace in the boostrap technique.

    replace: bool, default=False
        Whether to sample with replacement or not.

    hard_threshold: float, default=None
        Threshold defining the cutoff value for the stability selection.
        If the hard_threshold is defined, the FDRc will be bypassed.
        The default value is None: the user must set a value if no random permutation/knockoff is used.

    fdr_threshold_range: array-like, default=np.arange(0., 1., .01)
        When using random permutation or knockoff features, the user can change the tested values for the hard_threshold
        For each value, the FDRc will be computed.

    explore : bool, default=False
        If True, Stabl will select `n_explore` best features if no features are selected by the FDR control.

    n_explore : int, default=5
        Number of features to select if no features are selected by the FDR control.

    bootstrap_func: python function, default=classic_bootstrap
        Function to create a bootstrap sample from the original dataset. Look at `classic_bootstrap` for an example.

    sample_weight_bootstrap: array-like, default=None
        Class weight used in the bootstrap function.

    bootstrap_threshold: string or float, default=None
        The hard_threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``hard_threshold`` value is
        the median (resp. the mean) of the feature importance. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the hard_threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    perc_corr_group_threshold : float, default=None
        Threshold used to define the groups based on the correlation.

    sgl_groups : array-like, default=None
        Group of real features.

    verbose: int, default=0
        Controls the verbosity: the higher, the more messages.

    n_jobs: int, default=-1
        Number of jobs to run in parallel.

    random_state: int or None, default=None
        Random state for reproducibility matters.

    Attributes
    ----------
    n_features_in_: int
        Number of features seen during fit.

    feature_names_in_: ndarray of shape (n_features_in_, )
        Names of features seen during fit. Defined only when X has feature names that are all strings.

    stabl_scores_: array, shape(n_features, n_alphas)
        Array of stability scores for each feature and for each value of the
        penalization parameter.

    stabl_scores_artificial_: array, shape(n_features, n_alphas)
        Array of stability scores for each decoy/knockoff feature and for each value of the
        penalization parameter. Can only be accessed if we used decoy or knockoff in the
        training.

    X_artificial_: array, shape(n_repeats, n_features)
        Array of synthetic features. Can only be returned if we used decoy or knockoffs in the
        training.

    FDRs_: array
        The array of False Discovery Rates.
        Can only be retrieved if we used decoy or knockoffs in the training

    min_fdr_: float
        The Smallest FDR achieved
        Can only be retrieved if we used decoy or knockoffs in the training

    fdr_min_threshold_: float
        The hard_threshold achieving the desired FDR. Can only be retrieved if we used decoy or knockoff
        in the training and if no hard hard_threshold where defined.
    """

    def __init__(
            self,
            base_estimator=LogisticRegression(
                penalty='l1',
                solver='liblinear',
                class_weight='balanced',
                max_iter=int(1e6),
                random_state=42
            ),
            lambda_grid=None,
            n_lambda=None,
            n_bootstraps=1000,
            artificial_type="random_permutation",
            artificial_proportion=1.,
            sample_fraction=0.5,
            replace=False,
            hard_threshold=None,
            fdr_threshold_range=None,
            explore=False,
            n_explore=5,
            bootstrap_func=classic_bootstrap,
            sample_weight_bootstrap=None,
            bootstrap_threshold=1e-5,
            perc_corr_group_threshold=None,
            sgl_groups=None,
            verbose=0,
            n_jobs=-1,
            random_state=None,
            gaussianize_knockoffs=False,
            delta=0.05,
            selection_mode="unconstrained",
            cov_matrix=None,
            knockoff_method='sdp',
            alpha=1.0
    ):
        if fdr_threshold_range is None:
            fdr_threshold_range = np.arange(0., 1., .01)

        self.base_estimator = base_estimator
        self.lambda_grid = dict(C=np.linspace(0.01, 1, 10)) if lambda_grid is None else lambda_grid
        self.n_lambda = n_lambda
        self._check_lambda_grid()
        self.n_bootstraps = n_bootstraps
        self.artificial_type = artificial_type
        self.artificial_proportion = artificial_proportion
        self.sample_fraction = sample_fraction
        self.hard_threshold = hard_threshold
        self.fdr_threshold_range = fdr_threshold_range
        self.explore = explore
        self.n_explore = n_explore
        self.bootstrap_func = bootstrap_func
        self.sample_weight_bootstrap = sample_weight_bootstrap
        self.bootstrap_threshold = bootstrap_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.replace = replace
        self.random_state = random_state
        self.perc_corr_group_threshold = perc_corr_group_threshold
        self.sgl_groups = sgl_groups
        self.gaussianize_knockoffs = gaussianize_knockoffs
        self.delta = delta
        self.selection_mode = selection_mode
        self.cov_matrix = cov_matrix
        self.knockoff_method = knockoff_method
        # Offset du numérateur de FDP+ : num = (1/r)|S_ko(t)| + alpha.
        # alpha=1.0 -> +1 standard (Barber-Candès) ; alpha<1 dégonfle l'offset.
        self.alpha = alpha
        self.noise_group = np.array([])
        self.stabl_scores_ = None
        self.stabl_scores_artificial_ = None
        self.FDRs_ = None
        self.min_fdr_ = None
        self.fdr_min_threshold_ = None
        self.explore_threshold = None
        self.fitted_lambda_grid_ = None
        self.gs = None
        # Variance tracking (V2 stability validation)
        self.score_variance_    = None  # (p, K_lambda) — real features
        self.ko_score_variance_ = None  # (n_injected_noise, K_lambda) — knockoff features
        self.wj_mask_           = None  # (p,) bool — S₁ = {W_j > εWJ,j} (mode wj_constrained)

    def _validate_data(self, X=None, y=None, **kwargs):
        try:
            from sklearn.utils.validation import validate_data
            if X is not None and y is not None:
                return validate_data(self, X=X, y=y, **kwargs)
            elif X is not None:
                return validate_data(self, X=X, **kwargs)
            else:
                return validate_data(self, y=y, **kwargs)
        except ImportError:
            return super()._validate_data(X=X, y=y, **kwargs) if y is not None else super()._validate_data(X, **kwargs)

    def _check_lambda_grid(self):
        """Check if the lambda_grid is valid. Raise error if not.
        """
        if isinstance(self.lambda_grid, str):
            if self.lambda_grid != "auto":
                raise ValueError(
                    f'If lambda_grid is a string, it must be "auto", got {self.lambda_grid}'
                )
            base_estimator = self.base_estimator
            while not isinstance(base_estimator, (LogisticRegression, Lasso, ElasticNet)) and hasattr(base_estimator, "model"):
                base_estimator = base_estimator.model
            if not isinstance(base_estimator, (LogisticRegression, Lasso, ElasticNet)):
                raise ValueError(
                    f'If lambda_grid is "auto", the base_estimator must be a LogisticRegression, '
                    f'Lasso or ElasticNet, got {base_estimator}'
                )
            # Vérification l1_ratio uniquement quand le penalty est effectivement elasticnet.
            # En sklearn 1.8+, l1_ratio vaut 0.0 par défaut même pour penalty='l1' → on ne
            # peut pas checker l1_ratio sans vérifier d'abord le type de penalty.
            is_elasticnet_logit = (
                isinstance(base_estimator, LogisticRegression) and
                getattr(base_estimator, "penalty", None) == "elasticnet"
            )
            is_elasticnet_reg = (
                isinstance(base_estimator, ElasticNet) and
                not isinstance(base_estimator, Lasso)
            )
            if (is_elasticnet_logit or is_elasticnet_reg) and \
                    base_estimator.l1_ratio is not None and \
                    base_estimator.l1_ratio <= 0:
                raise ValueError(
                    f"If lambda_grid is 'auto' and the base_estimator is a LogisticRegression(penalty='elasticnet') "
                    f"or ElasticNet, the l1_ratio must be greater than 0, got {self.base_estimator.l1_ratio}"
                )
        return

    def _get_optimized_lambda_grid(self, X, y):
        """Return the optimized lambda_grid for the base_estimator if lambda_grid is set to "auto".

        Parameters
        ----------
        X : array-like, shape=(n_repeats, n_features)
            Input data matrix
        y : array-like, shape=(n_repeats, )
            Outcomes

        Returns
        -------
        lambda_grid : dict of parameters
            Optimized lambda_grid for the base_estimator
        """
        if self.lambda_grid != "auto":
            return self.lambda_grid

        l1_ratio = None
        n_lambda = 30 if self.n_lambda is None else self.n_lambda
        if isinstance(self.base_estimator, LogisticRegression) or isinstance(getattr(self.base_estimator, "model", None), LogisticRegression):
            task_type = "classification"
            if getattr(self.base_estimator, "penalty", getattr(getattr(self.base_estimator, "model", None), "penalty", None)) == "elasticnet":
                l1_ratio = [0.5, 0.7, 0.9]
                n_lambda = 10 if self.n_lambda is None else self.n_lambda
        else:
            task_type = "regression"
            if (isinstance(self.base_estimator, ElasticNet) and not isinstance(self.base_estimator, Lasso)) \
                    or (isinstance(getattr(self.base_estimator, "model", None), ElasticNet) and not isinstance(getattr(self.base_estimator, "model", None), Lasso)):
                l1_ratio = [0.5, 0.7, 0.9]
                n_lambda = 10 if self.n_lambda is None else self.n_lambda

        lambda_grid = auto_mode_lambda_grid(X, y, task_type, l1_ratio, n_lambda)
        return lambda_grid

    def _validate_input(self):
        """ Validate the input parameters. Raise error if not valid. """
        if not isinstance(self.n_bootstraps, int) or self.n_bootstraps <= 0:
            raise ValueError(
                f'n_bootstraps should be a positive integer, got {self.n_bootstraps}')

        if not isinstance(self.sample_fraction, float) or not (0.0 < self.sample_fraction):
            raise ValueError(
                f'sample_fraction should be a float in (0, 1], got {self.sample_fraction}')

        if isinstance(self.hard_threshold, float) and not (0.0 < self.hard_threshold <= 1):
            raise ValueError(
                f'If hard_threshold is set, it must be a float in (0, 1], got {self.hard_threshold}')

        if self.hard_threshold is None and self.artificial_type is None:
            raise ValueError(
                f'When not using synthetic features (random permutations, knockoff or gaussian noise), '
                f'the user must define a hard_threshold of selection, got {self.hard_threshold}'
            )

        if self.artificial_type is not None and not (0.0 < self.artificial_proportion <= 1.):
            raise ValueError(
                f"When injecting noise, the noise proportion must be between 0 and 1, "
                f"got {self.artificial_proportion}"
            )

    def _make_groups(self, X):
        """Make groups for self configuration.
           If self.per_corr_group_threshold is not None, it will use the correlation matrix to make groups.

           If self.sgl_groups is not None, it will use the groups defined by the user. 
           The corresponding noise features will be in the same group as its real feature.

        Parameters
        ----------
        X : array-like, shape=(n_repeats, n_features)
            Data matrix with noise features

        Returns
        -------
        groups : list of array-like, shape=(n_groups, ) or None
            list of groups of features
        """
        nb_real = X.shape[1] - self.noise_group.shape[0]
        X_real = X[:, :nb_real]

        n = X_real.shape[1]

        if self.perc_corr_group_threshold is not None:
            u = UnionFind(elements=range(X.shape[1]))
            corr_mat = pd.DataFrame(X_real).corr().values
            corr_val = corr_mat[np.triu_indices_from(corr_mat, k=1)]
            threshold = np.percentile(corr_val, self.perc_corr_group_threshold) - 0.1

            for i in np.arange(n):
                for j in np.arange(n):
                    if corr_mat[i, j] > threshold:
                        u.union(i, j)
            for idx, i in enumerate(self.noise_group):
                u.union(i, idx + n)

            return list(map(np.array, map(list, u.components())))

        elif self.sgl_groups is not None:
            ng = self.noise_group
            u = UnionFind(elements=range(X.shape[1]))
            for l_i in self.sgl_groups:
                for i in l_i:
                    u.union(i, l_i[0])
            for idx, i in enumerate(ng):
                u.union(i, idx + n)
            return list(map(np.array, map(list, u.components())))


    def fit(self, X, y, groups=None):
        """Fit the stability selection model on the given data.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_repeats, n_features)
            The training input samples.
        y : array-like, shape=(n_repeats, )
            The target values.
        groups : array-like, shape=(n_features, ) or None
            Groups for the samples used while splitting the dataset into
        """

        self._validate_input()

        X, y = self._validate_data(
            X=X,
            y=y,
            reset=True,
            validate_separately=False
        )

        n_samples, n_features = X.shape
        n_subsamples = int(np.floor(self.sample_fraction * n_samples))
        self.fitted_lambda_grid_ = self._get_optimized_lambda_grid(X, y)
        param_grid = list(ParameterGrid(self.fitted_lambda_grid_))

        n_lambdas = len(param_grid)

        # Defining the number of injected noisy features
        n_injected_noise = int(X.shape[1] * self.artificial_proportion)

        base_estimator = clone(self.base_estimator)

        # Initializing the Stabl scores
        self.stabl_scores_ = np.zeros((n_features, n_lambdas))

        # __Synthetic features and coefs__
        if self.artificial_type is not None:
            # Only initialize those score if we use artificial features
            self.stabl_scores_artificial_ = np.zeros(
                (n_injected_noise, n_lambdas))

        # Réinitialisation explicite à chaque appel de fit() pour que score_variance_
        # soit cohérent avec n_features courant (le même objet peut être refit sur
        # plusieurs omics de tailles différentes dans multi_omic_stabl_cv).
        self.score_variance_    = np.zeros((n_features, n_lambdas))
        if self.artificial_type is not None:
            self.ko_score_variance_ = np.zeros((n_injected_noise, n_lambdas))
 
        
        corr_groups = None
        if self.perc_corr_group_threshold is not None or self.sgl_groups is not None:
            corr_groups = self._make_groups(X)

        # Generating the bootstrap indices
        bootstrap_indices = _bootstrap_generator(
            n_bootstraps=self.n_bootstraps,
            bootstrap_func=self.bootstrap_func,
            y=y,
            n_subsamples=n_subsamples,
            replace=self.replace,
            groups=groups,
            class_weight=self.sample_weight_bootstrap,
            random_state=self.random_state
        )

        if self.artificial_type == "knockoff":
            # Gaussianisation optionnelle : réduit d_TV(L(X), N(mu,Sigma))
            # => borne d'erreur FDP+ plus serrée sur données biologiques non-gaussiennes.
            # X_gauss est utilisé UNIQUEMENT pour calibrer le sampler de knockoffs ;
            # le modèle de sélection tourne toujours sur X original.
            X_for_knockoffs = gaussianize(X) if self.gaussianize_knockoffs else X
            self.Lk, self.mu_k = GaussianSampler2(
                X_for_knockoffs,
                Sigma=self.cov_matrix,
                method=self.knockoff_method
            ).get_core()

            batch_size = 1  # chaque bootstrap est une tâche parallèle indépendante
            num_batches = (self.n_bootstraps + batch_size - 1) // batch_size
            bootstrap_seeds = np.random.default_rng(self.random_state).integers(0, 2**32 - 1, size=num_batches).tolist()

            bootstrap_batches = [(bootstrap_seeds[j],bootstrap_indices[i:i + batch_size])
                for j,i in enumerate(range(0, self.n_bootstraps, batch_size))
            ]

            gen_func = self._process_batch_knockoff
            gen_iterant = bootstrap_batches

        elif self.artificial_type == "random_permutation":
            bootstrap_seeds = np.random.default_rng(self.random_state).integers(0, 2**32 - 1, size=self.n_bootstraps).tolist()

            gen_func = self._process_single_random_permutation
            gen_iterant = [(bootstrap_seeds[i], bootstrap_indices[i])
                           for i in range(self.n_bootstraps)]
        elif self.artificial_type is None:
            # No artificial features: classic stability selection with hard_threshold
            gen_func = self._process_single_no_artificial
            gen_iterant = [(None, bootstrap_indices[i]) for i in range(self.n_bootstraps)]
        else:
            raise ValueError("The type of artificial feature must be in ['random_permutation', 'knockoff', None]."
                             f" Got {self.artificial_type}")
        

        # --Loop--
        leave = (self.verbose > 0)
        for idx, lambda_val in tqdm(
                enumerate(param_grid),
                'Stabl progress',
                total=n_lambdas,
                colour='#001A7B',
                leave=leave,
                file=sys.stdout,
                disable=(not leave)
        ):
            # Computing the frequencies


            selected_variables = Parallel(
                n_jobs=self.n_jobs,
                verbose=0,
                pre_dispatch='2*n_jobs'
            )(delayed(gen_func)(
                clone(base_estimator),
                X=X,
                y=y,
                jobInfo = grp,
                corr_groups=corr_groups,
                lambda_val=lambda_val,
                threshold=self.bootstrap_threshold,
                nb_noise=n_injected_noise
            )
                for grp in gen_iterant
            )


            # Flatten batch results (knockoff returns lists of results per batch)
            if self.artificial_type == "knockoff":
                flat_variables = [res for batch in selected_variables for res in batch]
            else:
                flat_variables = selected_variables

            all_selections = np.vstack(flat_variables)  # (n_bootstraps, n_features + n_artificial)

            if self.artificial_type is not None:
                self.stabl_scores_artificial_[:, idx] = all_selections[:, n_features:].mean(axis=0)
            self.stabl_scores_[:, idx] = all_selections[:, :n_features].mean(axis=0)

            # Variance tracking (real features): Var[score_j] = E_b[(sel_bj - mean_j)^2]
            # En V2, cette variance -> p_j(1-p_j)/B (variance d'une moyenne de Bernoulli i.i.d.)
            # En V1, un terme additionnel Var_{X~}(p_j(X~)) subsiste, irreductible en B.
            per_bootstrap_scores = all_selections[:, :n_features]  # (B, p)
            self.score_variance_[:, idx] = per_bootstrap_scores.var(axis=0)

            # Variance tracking (knockoff features): même logique, utilisée pour eps_ko_j
            # dans la borne Bernstein feature-wise (Thm 5 Bis).
            if self.artificial_type is not None:
                per_bootstrap_ko = all_selections[:, n_features:]   # (B, n_injected_noise)
                self.ko_score_variance_[:, idx] = per_bootstrap_ko.var(axis=0)

        if self.artificial_type is not None:
            self._compute_FDPplus()
            # Attribut de compatibilité avec save_stabl_results (stabl.py).
            # V2 ne stocke pas de matrice X_artificial_ unique (knockoffs régénérés
            # à chaque bootstrap), mais stabl.py y accède via .shape[1].
            # On expose un tableau fantôme de la bonne forme.
            self.X_artificial_ = np.zeros((1, n_injected_noise))
        return self

    def _process_batch_knockoff(self, base_estimator, X, y, jobInfo, 
                   corr_groups,lambda_val, threshold,nb_noise):
        
        seed,bootstrap_indices = jobInfo
        rng = np.random.default_rng(seed=seed)
        n,p = X.shape
        copies = len(bootstrap_indices)
        knockoffs = np.dot(self.Lk, rng.standard_normal((n,p,copies))) # result is (p, n, copies)
        knockoffs = np.transpose(knockoffs, [1, 0, 2])
        knockoffs = knockoffs + self.mu_k  

        # Ordre fixe : knockoff j reste toujours dans le slot j.
        # Cela garantit que ko_score_variance_[j] mesure la variance du knockoff
        # de la feature j, permettant un eps_ko_j feature-wise dans _compute_FDPplus.
        knockoffs = knockoffs[:, :nb_noise, :]

        batch_results = []
        for i,subsample_idx in enumerate(bootstrap_indices):

            X_artificial = knockoffs[:,:,i]

            X_aug = np.concatenate([X, X_artificial], axis=1)
            res = fit_bootstrapped_sample(
                clone(base_estimator),
                X=X_aug[safe_mask(X_aug, subsample_idx), :],
                y=y[subsample_idx],
                corr_groups=corr_groups,
                lambda_val=lambda_val,
                threshold=self.bootstrap_threshold
            )
            batch_results.append(res)
        return batch_results


    def _process_single_random_permutation(self, base_estimator, X, y, jobInfo, 
                   corr_groups,lambda_val, threshold,nb_noise):
        
        random_state, subsample_indices = jobInfo
        rng = np.random.default_rng(seed=random_state)

        X_artificial = X.copy()
        indices = rng.choice(a=X_artificial.shape[1], size=nb_noise, replace=False)
        X_artificial = X_artificial[:, indices]

        for i in range(X_artificial.shape[1]):
            rng.shuffle(X_artificial[:, i])

        X_aug = np.concatenate([X, X_artificial], axis=1)
        return fit_bootstrapped_sample(
                clone(base_estimator),
                X=X_aug[safe_mask(X_aug, subsample_indices), :],
                y=y[subsample_indices],
                corr_groups=corr_groups,
                lambda_val=lambda_val,
                threshold=self.bootstrap_threshold
            )
        

    def _process_single_no_artificial(self, base_estimator, X, y, jobInfo,
                                       corr_groups, lambda_val, threshold, nb_noise):
        """Process one bootstrap without artificial features (artificial_type=None)."""
        _, subsample_indices = jobInfo
        return fit_bootstrapped_sample(
            clone(base_estimator),
            X=X[safe_mask(X, subsample_indices), :],
            y=y[subsample_indices],
            corr_groups=corr_groups,
            lambda_val=lambda_val,
            threshold=self.bootstrap_threshold
        )

    def get_support(self, indices=False, new_hard_threshold=None):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        new_hard_threshold: float or None, default=None
            Threshold defining the minimum cutoff value for the
            stability scores. This is a hard hard_threshold: FDR control
            will be ignored if this is not None

        Returns
        -------
        support : array-like
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        mask = self._get_support_mask(new_hard_threshold=new_hard_threshold)
        return mask if not indices else np.where(mask)[0]

    def get_feature_names_out(self, input_features=None, new_hard_threshold=None):
        """Mask feature names according to selected features.

        Parameters
        ----------
        new_hard_threshold: float or None, default=None
            Threshold defining the minimum cutoff value for the
            stability scores. This is a hard threshold: FDR control
            will be ignored if this is not None

        input_features : array-like of str or None, default=None
            Input features.
            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        input_features = _check_feature_names_in(self, input_features)
        return input_features[self.get_support(new_hard_threshold=new_hard_threshold)]

    def transform(self, X, new_hard_threshold=None):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape=(n_repeats, n_features)
            The input array.

        new_hard_threshold: float or None, default=None
            Threshold defining the cutoff value for the
            stabl scores.
            When None the value set during the instantiation will be used.

        Returns
        -------
        X_out : array of shape=(n_repeats, n_selected_features)
            The input samples with only the selected features.
        """
        X = self._validate_data(X, reset=False)

        mask = self.get_support(
            indices=False, new_hard_threshold=new_hard_threshold)

        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        if not mask.any():
            warn("No features were selected: either the data is"
                 " too noisy or the selection test too strict.",
                 UserWarning)
            return np.empty(0).reshape((X.shape[0], 0))

        return X[:, safe_mask(X, mask)]

    def get_importances(self):
        """Get the feature importances (stability scores)

        Returns
        -------
        numpy.ndarray of shape (n_features_in_,)
            Feature importances, computed as the max of the stability scores
        """
        check_is_fitted(self, 'stabl_scores_')
        return np.max(self.stabl_scores_, axis=1)

    def _get_support_mask(self, new_hard_threshold=None):
        """Get a mask, or integer index, of the features selected

        Parameters
        ----------
        new_hard_threshold: float or None, default=None
            Threshold defining the cutoff value for the
            stabl scores.
            When None the value set during the instantiation will be used.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            This is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. 
        """
        check_is_fitted(self, 'stabl_scores_')

        new_threshold = self.hard_threshold if new_hard_threshold is None else new_hard_threshold

        # WJ mode: S = {j : |W_j| > εWJ,j} — deux côtés (W_j > εWJ,j OU W_j < −εWJ,j)
        if (new_threshold is None
                and self.selection_mode == "wj"
                and getattr(self, 'w_paired_', None) is not None):
            return np.abs(self.w_paired_) > self.eps_paired_

        # constrained_core mode: Ŝ = {j : score(j) > t* + εWJ,j} (barrière ∂+(t*) retirée).
        # Même t* que "constrained" ; le cutoff feature-wise écarte le cœur incertain.
        if (new_threshold is None
                and self.selection_mode == "constrained_core"
                and getattr(self, 'eps_B_total_fw_', None) is not None):
            max_scores = np.max(self.stabl_scores_, axis=1)
            return max_scores > self.fdr_min_threshold_ + self.eps_B_total_fw_

        # wj_constrained mode: Ŝ = {j ∈ S₁ : score(j) > t*} (WJ pré-filtre + seuil sur S₁).
        if (new_threshold is None
                and self.selection_mode == "wj_constrained"
                and getattr(self, 'wj_mask_', None) is not None):
            max_scores = np.max(self.stabl_scores_, axis=1)
            return self.wj_mask_ & (max_scores > self.fdr_min_threshold_)

        if new_threshold is None:
            final_cutoff = self.fdr_min_threshold_
        else:
            final_cutoff = new_threshold

        max_scores = np.max(self.stabl_scores_, axis=1)
        mask = max_scores > final_cutoff

        if np.sum(mask) == 0 and self.explore is True:
            n_explore = min(self.n_explore, len(max_scores))
            final_cutoff = np.sort(max_scores)[-n_explore] - 0.01
            self.explore_threshold = final_cutoff
            mask = max_scores > final_cutoff
        else:
            self.explore_threshold = None

        return mask


    def _compute_FDPplus(self):
        """Compute FDP+(t) and t* according to self.selection_mode.

        Modes
        -----
        "unconstrained" : t* = argmin FDP+(t) — no frontier, no Wj.
        "constrained"   : t* = argmin [FDP+(t) + |∂+(t)|/D(t)] — Maurer-Pontil
                          feature-wise frontier term, no Wj filter.
                          Selection S = {j : score(j) > t*} (garde la barrière ∂+(t*)).
        "constrained_core" : t*_c = argmin FDP+_c(t), FDP+_c(t)=((1/r)|S_ko(t)|+1)/|S_c(t)|
                          avec S_c(t)={j:score(j)>t+εWJ,j} ; retourne Ŝ = S_c(t*_c).
                          Definition COHERENTE : on minimise la borne de l'ensemble
                          RETOURNE (pas de vidage post-hoc). Sur Ω, FDP(Ŝ) ≤ FDP+_c(t*_c)
                          sans terme frontiere (le +1 = correction conservatrice standard).
        "wj"            : S = {j : |W_j| > εWJ,j} (deux côtés : W_j > εWJ,j OU W_j < −εWJ,j).
                          t* = argmin FDP+(t) stored for FDR curve visualization only.
        "wj_constrained": S₁ = {j : W_j > εWJ,j}, puis Thm 7 RESTREINT à S₁ :
                          t* = argmin_t [FDP+_restr(t) + |∂+_restr(t)|/D_restr(t)] (tout sur S₁),
                          Ŝ = {j∈S₁ : score(j)>t*}. Garde FDP=0 de WJ, bornes plus tight.

        In all modes, ∂+(t) = {j : t < score(j) ≤ t + εB,j + εB,j,ko} with Maurer-Pontil
        feature-wise tolerances. εWJ,j = εB,j + εB,j,ko.

        Stored attributes
        -----------------
        eps_B_              : float — Hoeffding ε (backward compat)
        eps_B_featurewise_  : ndarray (p,) — Maurer-Pontil εB,j per real feature
        eps_B_ko_           : ndarray (p,) — Maurer-Pontil εB,j,ko per knockoff
        eps_B_total_fw_     : ndarray (p,) — εWJ,j = εB,j + εB,j,ko per feature
        constrained_threshold_ : bool — True iff mode is "constrained"
        min_margin_fw_      : float or None — min_j(score(j) − t* − εWJ,j) over selected
        w_paired_           : ndarray (p,) or None — W_j = score(j) − score_ko(j) [wj mode]
        eps_paired_         : ndarray (p,) or None — εWJ,j per feature              [wj mode]
        etas_               : ndarray — uniform gap η(t) per threshold (backward compat)
        """
        artificial_proportion = self.artificial_proportion
        max_scores_artificial = np.max(self.stabl_scores_artificial_, axis=1)
        max_scores            = np.max(self.stabl_scores_,             axis=1)
        thresh_grid           = self.fdr_threshold_range
        n_thresh              = len(thresh_grid)

        # ── Empirical FDP+(t) ────────────────────────────────────────────────
        # num = (1/r)|S_ko(t)| + alpha(t). alpha=1 -> offset Barber-Candès standard.
        # alpha peut être :
        #   - un float                    -> offset constant
        #   - une fonction t -> alpha      -> offset variable en t
        #   - une fonction (t, D) -> alpha -> offset dépendant aussi de D(t)=#{score>t}.
        #     Ex. alpha(t,D)=t^gamma·D : la contribution à FDP+ vaut alpha/D = t^gamma,
        #     pénalité déterministe croissante indépendante de D (ne s'annule pas en queue).
        D_for_alpha = np.array([max(1, int(np.sum(max_scores > t))) for t in thresh_grid])
        if callable(self.alpha):
            import inspect
            try:
                nparams = len(inspect.signature(self.alpha).parameters)
            except (ValueError, TypeError):
                nparams = 1
            if nparams >= 2:
                alpha_arr = np.asarray([float(self.alpha(thresh_grid[i], D_for_alpha[i]))
                                        for i in range(n_thresh)])
            else:
                alpha_arr = np.asarray([float(self.alpha(t)) for t in thresh_grid])
        else:
            alpha_arr = np.full(n_thresh, float(self.alpha))
        self.alpha_arr_ = alpha_arr      # exposé pour visualisation / debug

        FDPs = []
        for i, thresh in enumerate(thresh_grid):
            num   = np.sum((1 / artificial_proportion) * (max_scores_artificial > thresh)) + alpha_arr[i]
            denum = max(1, int(np.sum(max_scores > thresh)))
            FDPs.append(num / denum)
        FDPs = np.array(FDPs)

        fdrs_table = np.zeros((self.stabl_scores_.shape[1], n_thresh))
        for i in range(self.stabl_scores_.shape[1]):
            ms_art = self.stabl_scores_artificial_[:, i]
            ms     = self.stabl_scores_[:, i]
            for j, thresh in enumerate(thresh_grid):
                num   = np.sum((1 / artificial_proportion) * (ms_art > thresh)) + alpha_arr[j]
                denum = max(1, int(np.sum(ms > thresh)))
                fdrs_table[i, j] = num / denum

        self.fdrs_table = fdrs_table
        self.FDRs_      = list(FDPs)
        self.min_fdr_   = float(FDPs.min())

        # ── Paramètres communs ───────────────────────────────────────────────
        p        = self.n_features_in_
        K_lambda = self.stabl_scores_.shape[1]
        B        = self.n_bootstraps
        delta    = self.delta
        log_term = max(np.log(4 * p * K_lambda / delta), 0.0)
        B_eff    = max(B - 1, 1)

        # ── Hoeffding ε_B (backward compat) ─────────────────────────────────
        self.eps_B_ = float(np.sqrt(log_term / (2 * B))) if B > 0 else 0.0

        # ── Maurer-Pontil feature-wise εB,j and εB,j,ko ─────────────────────
        # ε = [(7/3)·L + √((7/3)²·L² + 8(B-1)·L·σ̂²)] / (2(B-1))  with L = log(4pK/δ).
        # Used in "constrained" (frontier objective) and "wj" (Wj criterion).
        sigma2_real = self.score_variance_.max(axis=1)                        # (p,)
        disc_real   = (7/3)**2 * log_term**2 + 8 * B_eff * log_term * sigma2_real
        eps_j       = ((7/3) * log_term + np.sqrt(np.maximum(disc_real, 0.0))) / (2 * B_eff)
        self.eps_B_featurewise_ = eps_j                                       # (p,)

        if self.ko_score_variance_ is not None and self.ko_score_variance_.shape[0] == p:
            sigma2_ko_j = self.ko_score_variance_.max(axis=1)                 # (p,)
        else:
            sigma2_ko_j = np.full(p, 0.25)                                    # Hoeffding worst-case
        disc_ko  = (7/3)**2 * log_term**2 + 8 * B_eff * log_term * sigma2_ko_j
        eps_ko_j = ((7/3) * log_term + np.sqrt(np.maximum(disc_ko, 0.0))) / (2 * B_eff)
        self.eps_B_ko_ = eps_ko_j                                             # (p,)

        eps_tot_j = eps_j + eps_ko_j                                          # (p,)  = εWJ,j
        self.eps_B_total_fw_ = eps_tot_j

        # ── η(t) uniforme (backward compat) ──────────────────────────────────
        self.etas_ = np.array([
            float(max_scores[max_scores > t].min() - t) if np.any(max_scores > t) else float(1.0 - t)
            for t in thresh_grid
        ])

        # ── Mode: unconstrained — t* = argmin FDP+(t) ────────────────────────
        if self.selection_mode == "unconstrained":
            best_idx                    = int(np.where(FDPs == FDPs.min())[0][0])
            self.constrained_threshold_ = False
            self.w_paired_              = None
            self.eps_paired_            = None
            self.min_margin_fw_         = None

        # ── Mode: constrained — t* = argmin [FDP+(t) + |∂+(t)|/D(t)] ────────
        # S = {score > t*} (garde la barrière ∂+(t*)).
        elif self.selection_mode == "constrained":
            D_t_arr  = np.array([max(1, int(np.sum(max_scores > t))) for t in thresh_grid])
            bdry_arr = np.array([
                np.sum((max_scores > t) & (max_scores <= t + eps_tot_j)) / D_t_arr[i]
                for i, t in enumerate(thresh_grid)
            ])
            obj      = FDPs + bdry_arr
            best_idx                    = int(np.where(obj == obj.min())[0][0])
            self.constrained_threshold_ = True
            self.w_paired_              = None
            self.eps_paired_            = None
            t_star   = thresh_grid[best_idx]
            sel_mask = max_scores > t_star
            self.min_margin_fw_ = float(
                np.min(max_scores[sel_mask] - t_star - eps_tot_j[sel_mask])
            ) if sel_mask.any() else 0.0

        # ── Mode: constrained_core — t*_c = argmin FDP+_c(t), Ŝ = {score > t*_c + εWJ,j} ──
        # FDP+_c(t) = ((1/r)|S_ko(t)| + 1) / max(1,|S_c(t)|),  S_c(t) = {score > t + εWJ,j}.
        # Inclusion S_c∩M0 ⊆ S_ko(t) => |S_c∩M0| ≤ |S_ko(t)| : PAS de terme frontière.
        # Le +1 est la correction conservatrice standard (façon knockoff/BH). On minimise
        # la borne de l'ensemble RETOURNÉ => définition cohérente, pas de vidage post-hoc.
        elif self.selection_mode == "constrained_core":
            n_ko_arr = np.array([
                np.sum((1 / artificial_proportion) * (max_scores_artificial > t))
                for t in thresh_grid
            ])
            Dc_arr   = np.array([
                max(1, int(np.sum(max_scores > t + eps_tot_j))) for t in thresh_grid
            ])
            obj      = (n_ko_arr + 1) / Dc_arr      # FDP+_c(t) = ((1/r)|S_ko(t)| + 1) / |S_c(t)|
            best_idx                    = int(np.where(obj == obj.min())[0][0])
            self.constrained_threshold_ = True
            self.w_paired_              = None
            self.eps_paired_            = None
            t_star   = thresh_grid[best_idx]
            sel_mask = max_scores > t_star + eps_tot_j
            self.min_margin_fw_ = float(
                np.min(max_scores[sel_mask] - t_star - eps_tot_j[sel_mask])
            ) if sel_mask.any() else 0.0

        # ── Mode: wj_constrained — S₁={W_j>εWJ,j}, puis V2_constr RESTREINT à S₁ ──
        # Theorem 7 appliqué à S₁ : t* = argmin_t [FDP+_restr(t) + |∂+_restr(t)|/D_restr(t)],
        # tout compté sur S₁ (numérateur knockoff inclus). Ŝ = {j∈S₁ : score(j)>t*}.
        # Sur Ω, S₁ n'a aucune nulle (garantie WJ, FDP=0) ; et la borne est plus tight car
        # knockoffs, frontière et nulles sont tous réduits sur S₁. Preuve identique à Thm 7.
        elif self.selection_mode == "wj_constrained":
            paired_ok = (
                self.ko_score_variance_ is not None
                and self.stabl_scores_artificial_.shape[0] == p
                and self.ko_score_variance_.shape[0] == p
            )
            if paired_ok:
                wj_mask = (max_scores - max_scores_artificial) > eps_tot_j     # S₁
            else:
                wj_mask = np.ones(p, dtype=bool)        # fallback = constrained sur tout
            self.wj_mask_ = wj_mask
            ms      = max_scores[wj_mask]               # scores réels restreints à S₁
            ms_art  = max_scores_artificial[wj_mask]    # scores knockoffs restreints à S₁
            eps_s1  = eps_tot_j[wj_mask]
            D_t_arr  = np.array([max(1, int(np.sum(ms > t))) for t in thresh_grid])
            n_ko_arr = np.array([
                np.sum((1 / artificial_proportion) * (ms_art > t)) for t in thresh_grid
            ])
            bdry_arr = np.array([
                np.sum((ms > t) & (ms <= t + eps_s1)) / D_t_arr[i]
                for i, t in enumerate(thresh_grid)
            ])
            obj      = (n_ko_arr + 1) / D_t_arr + bdry_arr   # FDP+_restr + frontière_restr
            best_idx                    = int(np.where(obj == obj.min())[0][0])
            self.constrained_threshold_ = True
            self.w_paired_              = None
            self.eps_paired_            = None
            t_star   = thresh_grid[best_idx]
            sel_mask = wj_mask & (max_scores > t_star)
            self.min_margin_fw_ = float(
                np.min(max_scores[sel_mask] - t_star - eps_tot_j[sel_mask])
            ) if sel_mask.any() else 0.0

        # ── Mode: wj — S₁ = {j : W_j > εWJ,j},  FDP(S₁) = 0 w.p. ≥ 1-δ ────
        elif self.selection_mode == "wj":
            # W_j = score(j) - score_ko(j)  where both use their respective max over λ.
            # Using max_scores_artificial (= max_λ score_ko(j,λ)) ensures that for null j,
            # W_j → p_j - p_j = 0 ≤ εWJ,j on Ω, preserving FDP(S₁) = 0.
            # Using score_ko(j, k_star) instead would give W_j → p_j - p_j(k_star) ≥ 0,
            # which can be strictly positive for nulls and breaks the guarantee.
            paired_ok = (
                self.ko_score_variance_ is not None
                and self.stabl_scores_artificial_.shape[0] == p
                and self.ko_score_variance_.shape[0] == p
            )
            if paired_ok:
                self.w_paired_   = max_scores - max_scores_artificial         # W_j
                self.eps_paired_ = eps_tot_j                                  # εWJ,j = εB,j + εB,j,ko
            else:
                self.w_paired_   = None
                self.eps_paired_ = None
            # t* = argmin FDP+(t) pour visualisation uniquement
            best_idx                    = int(np.where(FDPs == FDPs.min())[0][0])
            self.constrained_threshold_ = False
            self.min_margin_fw_         = None

        else:
            raise ValueError(
                f"selection_mode must be 'unconstrained', 'constrained', "
                f"'constrained_core', 'wj_constrained', or 'wj'; "
                f"got '{self.selection_mode}'"
            )

        # ── Seuil FDP+ final ─────────────────────────────────────────────────
        if self.min_fdr_ > 1.:
            self.fdr_min_threshold_ = 1.
        else:
            self.fdr_min_threshold_ = float(np.minimum(thresh_grid[best_idx], 1.0))

    def get_different_parameters(self):
        """Get all the parameters modified in the gridSearch of a Stabl object.
        """
        check_is_fitted(self, 'fitted_lambda_grid_')
        keys = set()
        for p in ParameterGrid(self.fitted_lambda_grid_):
            keys.update(p.keys())
        return list(keys)