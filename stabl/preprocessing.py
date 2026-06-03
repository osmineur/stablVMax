import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
try:
    from sklearn.utils.validation import validate_data as _sklearn_validate_data
    _USE_NEW_VALIDATE = True
except ImportError:
    _USE_NEW_VALIDATE = False


def remove_low_info_samples(X, threshold=1.0):
    """Removes low info samples

    A sample is considered to have sufficient info if the nan fraction is below the
    input hard_threshold.
    
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_repeats, n_features)
        Data from which to compute NaN proportion, where `n_repeats` is
        the number of samples and `n_features` is the number of features.

    threshold : float, default=1.0
        Samples with a proportion of NaN greater than this value will be removed.
    
    Returns
    -------
    X_reduced : array, shape(n_samples_out, n_features)
        The reduced array of siwe n_samples_out, n_features
    """
    if not isinstance(threshold, float) or (threshold < 0. or threshold > 1.):
        raise ValueError(f"Nan fraction must be between 0 and 1 Got: {threshold}")

    nan_fraction = np.isnan(X).sum(1) / X.shape[1]
    mask = nan_fraction < threshold
    return X[mask]


class LowInfoFilter(SelectorMixin, BaseEstimator):
    """Feature selector that removes all low-variance features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    A feature is considered to be a low info one if the proportion of nan
    values is above a given hard_threshold set by the user.

    Parameters
    ----------
    max_nan_fraction : float, default=0.2
        Features with a proportion of nan values greater than this hard_threshold will
        be removed. By default, the proportion is set to 0.2.

    Attributes
    ----------
    nan_counts_ : array, shape (n_features,)
        Count of nan values for each individual feature.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_, )
        Names of features seen during the fit. Defined only when X
        has feature names that are all strings.

    Notes
    -----
    Allows NaN in the input.
    Raises ValueError if no feature in X meets the low info hard_threshold.
    """

    def __init__(self, max_nan_fraction=0.2):
        self.max_nan_fraction = max_nan_fraction
        self.n_samples = None
        self.nan_counts_ = None

    def fit(self, X, y=None):
        """Learn empirical Nan proportion in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_repeats, n_features)
            Data from which to compute NaN proportion, where `n_repeats` is
            the number of samples and `n_features` is the number of features.

        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        validate_kwargs = dict(
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
        )
        try:
            import sklearn
            if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 6):
                validate_kwargs["ensure_all_finite"] = "allow-nan"
            else:
                validate_kwargs["force_all_finite"] = "allow-nan"
        except Exception:
            validate_kwargs["ensure_all_finite"] = "allow-nan"

        if _USE_NEW_VALIDATE:
            X = _sklearn_validate_data(self, X, **validate_kwargs)
        else:
            X = self._validate_data(X, **validate_kwargs)

        if self.max_nan_fraction > 1 or self.max_nan_fraction < 0:
            raise ValueError(
                f"Nan fraction must be between 0 and 1 Got: {self.max_nan_fraction}")

        n_samples = X.shape[0]
        self.n_samples = n_samples
        self.nan_counts_ = np.isnan(np.array(X)).sum(0)

        if np.all(~np.isfinite(self.nan_counts_) | (
                self.nan_counts_ > self.max_nan_fraction * self.n_samples)):
            msg = "No feature in X meets the low info hard_threshold {0:.5f}"
            if n_samples == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.max_nan_fraction))

        return self

    def _get_support_mask(self):
        """Get a mask, or integer index, of the features selected
            
        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            This is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. 
        """
        check_is_fitted(self)

        return self.nan_counts_ <= self.max_nan_fraction * self.n_samples

    def _more_tags(self):
        return {"allow_nan": True}

    def __sklearn_tags__(self):
        tags = super(LowInfoFilter, self).__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class CorrelationFilter(SelectorMixin, BaseEstimator):
    """Feature selector basé sur les composantes connexes du graphe de corrélation.

    Algorithme :
    1. On construit un graphe où chaque feature est un sommet et on trace une
       arête entre deux features si |r| > threshold.
    2. On trouve les composantes connexes de ce graphe (groupes de features
       reliées directement ou indirectement).
    3. Dans chaque composante, on garde la feature avec la somme de corrélations
       absolues la plus élevée avec les autres membres (la plus "centrale").

    Parameters
    ----------
    threshold : float, default=0.9
        Seuil de corrélation absolue de Pearson au-delà duquel deux features
        sont considérées comme redondantes.

    Attributes
    ----------
    support_mask_ : ndarray of shape (n_features_in_,)
        Masque booléen des features conservées.

    n_features_in_ : int
        Nombre de features vues lors du fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Noms des features vues lors du fit.
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    @staticmethod
    def _count_non_trivial_components(abs_corr, threshold):
        """Compte les composantes connexes de taille > 1 pour un threshold donné."""
        n = abs_corr.shape[0]
        adj = abs_corr > threshold
        np.fill_diagonal(adj, False)
        visited = np.zeros(n, dtype=bool)
        count = 0
        for start in range(n):
            if visited[start]:
                continue
            component = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbor in np.where(adj[node])[0]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            if len(component) > 1:
                count += 1
        return count

    def _find_optimal_threshold(self, abs_corr):
        """Balaye les thresholds et retourne celui qui maximise
        le nombre de composantes connexes non-triviales."""
        candidates = np.round(np.arange(0.05, 1.0, 0.01), 2)
        counts = [self._count_non_trivial_components(abs_corr, t) for t in candidates]
        best_idx = int(np.argmax(counts))
        best_threshold = float(candidates[best_idx])
        print(f"CorrelationFilter threshold='auto': optimal threshold = {best_threshold} "
              f"({counts[best_idx]} groupes non-triviaux)")
        return best_threshold

    def fit(self, X, y=None):
        validate_kwargs = dict(accept_sparse=False, dtype=np.float64)
        try:
            import sklearn
            if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 6):
                validate_kwargs["ensure_all_finite"] = "allow-nan"
            else:
                validate_kwargs["force_all_finite"] = "allow-nan"
        except Exception:
            validate_kwargs["ensure_all_finite"] = "allow-nan"

        if _USE_NEW_VALIDATE:
            X = _sklearn_validate_data(self, X, **validate_kwargs)
        else:
            X = self._validate_data(X, **validate_kwargs)

        # Remplacement des NaN par la moyenne de chaque colonne pour le calcul
        arr = X.copy()
        col_means = np.nanmean(arr, axis=0)
        nan_mask = np.isnan(arr)
        arr[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        n = arr.shape[1]
        abs_corr = np.abs(np.corrcoef(arr, rowvar=False))

        threshold = (
            self._find_optimal_threshold(abs_corr)
            if self.threshold == "auto"
            else self.threshold
        )
        self.threshold_ = threshold

        # ── Étape 1 : matrice d'adjacence du graphe de corrélation ──────────
        # A[i,j] = 1 si |r(i,j)| > threshold (et i != j)
        adj = (abs_corr > threshold).astype(bool)
        np.fill_diagonal(adj, False)

        # ── Étape 2 : composantes connexes par BFS ───────────────────────────
        # On parcourt chaque sommet non encore visité et on explore tous ses
        # voisins (directs et indirects) pour former un groupe.
        visited = np.zeros(n, dtype=bool)
        components = []

        for start in range(n):
            if visited[start]:
                continue
            # BFS depuis ce sommet
            component = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbor in np.where(adj[node])[0]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(component)

        # ── Étape 3 : sélection du représentant par centralité ───────────────
        # Dans chaque composante, on garde la feature dont la somme de
        # corrélations absolues avec les autres membres est la plus élevée.
        kept = []
        components_info = []
        feature_names = (
            self.feature_names_in_
            if hasattr(self, "feature_names_in_")
            else np.arange(n).astype(str)
        )

        for comp_id, component in enumerate(components):
            if len(component) == 1:
                best = component[0]
                avg_intra_corr = None
            else:
                scores = [
                    sum(abs_corr[i, j] for j in component if j != i)
                    for i in component
                ]
                best = component[int(np.argmax(scores))]
                pairs = [
                    abs_corr[i, j]
                    for idx, i in enumerate(component)
                    for j in component[idx + 1:]
                ]
                avg_intra_corr = round(float(np.mean(pairs)), 3)
            kept.append(best)
            components_info.append({
                "component_id":   comp_id,
                "size":           len(component),
                "representative": feature_names[best],
                "avg_intra_corr": avg_intra_corr,
            })

        self.components_info_ = components_info
        self.n_components_ = len(components)

        # ── Log des composantes connexes ─────────────────────────────────────
        sizes = sorted([len(c) for c in components], reverse=True)
        non_trivial = [s for s in sizes if s > 1]
        print(f"CorrelationFilter (threshold={self.threshold_}): "
              f"{len(components)} composantes connexes — "
              f"{len(non_trivial)} groupes corrélés (tailles : {non_trivial}), "
              f"{sizes.count(1)} features isolées")

        self.support_mask_ = np.zeros(n, dtype=bool)
        self.support_mask_[kept] = True
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_

    def _more_tags(self):
        return {"allow_nan": True}

    def __sklearn_tags__(self):
        tags = super(CorrelationFilter, self).__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
