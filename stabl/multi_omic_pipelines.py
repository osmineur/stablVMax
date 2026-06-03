import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
from .unionfind import UnionFind
import sys
from tqdm import tqdm
from .pipelines_utils import save_plots, compute_scores_table, compute_pvalues_table
from .stacked_generalization import stacked_multi_omic
from .metrics import jaccard_matrix
from .stabl import save_stabl_results
from .preprocessing import remove_low_info_samples, LowInfoFilter
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, GroupShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn import clone
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from xgboost import XGBRegressor,XGBClassifier

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', category=ConvergenceWarning)
ConvergenceWarning('ignore')

outter_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

inner_reg_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
inner_group_cv = GroupShuffleSplit(n_splits=25, test_size=0.2, random_state=42)

nb_param = 50

logit = LogisticRegression(C=np.inf, class_weight="balanced", max_iter=int(1e6), random_state=42)
linreg = LinearRegression()

xgboost = XGBClassifier(n_jobs =1)

def _make_preprocessing_pipeline():
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(0.0)),
            ("lif", LowInfoFilter()),
            ("impute", SimpleImputer(strategy="median")),
            ("std", StandardScaler())
        ]
    )


def _make_groups(X, percentile):
    n = X.shape[1]
    u = UnionFind(elements=range(n))
    corr_mat = pd.DataFrame(X).corr().values
    corr_val = corr_mat[np.triu_indices_from(corr_mat, k=1)]
    threshold = np.percentile(corr_val, percentile)
    for i in np.arange(n):
        for j in np.arange(n):
            if abs(corr_mat[i, j]) > threshold:
                u.union(i, j)
    res = list(map(list, u.components()))
    res = list(map(np.array, res))
    return res


@ignore_warnings(category=ConvergenceWarning)
def multi_omic_stabl_cv(
        data_dict,
        y,
        outer_splitter,
        estimators,
        task_type,
        save_path,
        models,
        outer_groups=None,
        early_fusion=False,
        late_fusion=True,
        n_iter_lf=10000,
        preprocessing_overrides=None,
):
    """
    Performs a cross validation on the data_dict using the models and saves the results in save_path.

    Parameters
    ----------
    data_dict: dict
        Dictionary containing the input omic-files. the input omic-files should be pandas DataFrames

    y: pd.Series
        pandas Series containing the outcomes for the use case. Note that y should contain the union of outcomes for
        the data_dict.

    outer_splitter: sklearn.model_selection._split.BaseCrossValidator
        Outer cross validation splitter

    estimators : dict[str, sklearn estimator]
        Dict of feature selectors to benchmark with their name as key. They must implement `fit`,
        `get_support(indices=True)`.
        It should contain :
        - "lasso" : Lasso in GridSearchCV
        - "alasso" : ALasso in GridSearchCV
        - "en" : ElasticNet in GridSearchCV
        - "sgl" : SGL in GridSearchCV
        - "stabl_lasso" : Stabl with Lasso as base estimator
        - "stabl_alasso" : Stabl with ALasso as base estimator
        - "stabl_en" : Stabl with ElasticNet as base estimator
        - "stabl_sgl" : Stabl with SGL as base estimator

    task_type: str
        Can either be "binary" for binary classification or "regression" for regression tasks.

    save_path: Path or str
        Where to save the results

    models: list of str
        List of models to use. Can contain :
        - "Lasso" : Lasso
        - "STABL Lasso" : Stabl with Lasso as base estimator
        - "ALasso" : ALasso
        - "STABL ALasso" : Stabl with ALasso as base estimator
        - "ElasticNet" : ElasticNet
        - "STABL ElasticNet" : Stabl with ElasticNet as base estimator

    outer_groups: pd.Series, default=None
        If used, should be the same size as y and should indicate the groups of the samples.

    early_fusion: bool, default=False
        If True, it will perform early fusion for each estimator.

    late_fusion: bool, default=False
        If True, it will perform late fusion for each estimator.

    n_iter_lf: int, default=10000
        Number of iterations for the late fusion.


    Returns
    -------
    predictions_dict: dict
        Dictionary containing the predictions of each model for each sample in Cross-Validation.
    """
    if early_fusion:
        models += ["EF " + model for model in models if "STABL" not in model]

    lasso = estimators["lasso"]
    alasso = estimators["alasso"]
    en = estimators["en"]

    stabl = estimators["stabl_lasso"]
    stabl_alasso = estimators["stabl_alasso"]
    stabl_en = estimators["stabl_en"]
    xgb_estimator = estimators.get("xgboost", None)
    svm = estimators.get("svm", None)
    random_forest = estimators.get("random_forest", None)
    stabl_xgb = estimators.get("stabl_xgboost", None)

    preprocessing = _make_preprocessing_pipeline()

    os.makedirs(Path(save_path, "Training CV"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"), exist_ok=True)

    # Initializing the df containing the data of all omics
    X_tot = pd.concat(data_dict.values(), axis="columns")

    predictions_dict = dict()
    selected_features_dict = dict()
    stabl_features_dict = dict()

    for model in models:
        predictions_dict[model] = pd.DataFrame(data=None, index=y.index)
        selected_features_dict[model] = []
        stabl_features_dict[model] = dict()
        for omic_name in data_dict.keys():
            if "STABL" in model:
                stabl_features_dict[model][omic_name] = pd.DataFrame(data=None, columns=["Threshold", "min FDP+"])

    corr_filter_rows = {}  # {omic_name: [rows]}

    k = 1
    for train, test in (tbar := tqdm(
            outer_splitter.split(X_tot, y, groups=outer_groups),
            total=outer_splitter.get_n_splits(X=X_tot, y=y, groups=outer_groups),
            file=sys.stdout
    )):
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index
        groups = outer_groups.loc[train_idx].values if outer_groups is not None else None

        predictions_dict_late_fusion = dict()
        fold_selected_features = dict()
        for model in models:
            fold_selected_features[model] = []
            if "STABL" not in model and "EF" not in model:
                predictions_dict_late_fusion[model] = dict()
                for omic_name in data_dict.keys():
                    predictions_dict_late_fusion[model][omic_name] = pd.DataFrame(data=None, index=test_idx)

        tbar.set_description(f"{len(train_idx)} train samples, {len(test_idx)} test samples")

        for omic_name, X_omic in data_dict.items():
            test_idx_tmp = X_omic.index.intersection(test_idx)
            X_tmp = X_omic.drop(index=test_idx, errors="ignore")
            X_test_tmp = X_omic.loc[test_idx_tmp]
            # Preprocessing of X_tmp
            X_tmp = remove_low_info_samples(X_tmp)
            y_tmp = y.loc[X_tmp.index]
            omic_groups = outer_groups[X_tmp.index] if outer_groups is not None else None

            omic_preprocessing = (
                preprocessing_overrides[omic_name]
                if preprocessing_overrides and omic_name in preprocessing_overrides
                else preprocessing
            )

            X_tmp_std = pd.DataFrame(
                data=omic_preprocessing.fit_transform(X_tmp),
                index=X_tmp.index,
                columns=omic_preprocessing.get_feature_names_out()
            )

            # Accumulation des composantes connexes si le pipeline contient un CorrelationFilter
            from stabl.preprocessing import CorrelationFilter
            from sklearn.pipeline import Pipeline as _Pipeline
            if isinstance(omic_preprocessing, _Pipeline):
                for _, step in omic_preprocessing.steps:
                    if isinstance(step, CorrelationFilter) and hasattr(step, "components_info_"):
                        if omic_name not in corr_filter_rows:
                            corr_filter_rows[omic_name] = []
                        else:
                            # ligne de séparation entre folds
                            corr_filter_rows[omic_name].append({})
                        for row in step.components_info_:
                            corr_filter_rows[omic_name].append({
                                "fold":         k,
                                "threshold":    step.threshold_,
                                "n_components": step.n_components_,
                                **row,
                            })
                        break

            X_test_tmp_std = pd.DataFrame(
                data=omic_preprocessing.transform(X_test_tmp),
                index=X_test_tmp.index,
                columns=omic_preprocessing.get_feature_names_out()
            )

            # __STABL__
            if "STABL Lasso" in models:
                # fit STABL Lasso
                print("Fitting of STABL Lasso")
                stabl.fit(X_tmp_std, y_tmp, groups=omic_groups)
                tmp_sel_features = list(stabl.get_feature_names_out())
                fold_selected_features["STABL Lasso"].extend(tmp_sel_features)
                print(
                    f"STABL Lasso finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL Lasso"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl.min_fdr_
                stabl_features_dict["STABL Lasso"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl,
                        path=Path(save_path, "Training CV", f"STABL Lasso results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type,
                        override=True
                    )

            if "STABL ALasso" in models:
                # fit STABL ALasso
                print("Fitting of STABL ALasso")
                stabl_alasso.fit(X_tmp_std, y_tmp, groups=omic_groups)
                tmp_sel_features = list(stabl_alasso.get_feature_names_out())
                fold_selected_features["STABL ALasso"].extend(tmp_sel_features)
                print(
                    f"STABL ALasso finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL ALasso"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_alasso.min_fdr_
                stabl_features_dict["STABL ALasso"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_alasso.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_alasso,
                        path=Path(save_path, "Training CV", f"STABL ALasso results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type,
                        override=True
                    )

            if "STABL ElasticNet" in models:
                # fit STABL ElasticNet
                print("Fitting of STABL ElasticNet")
                stabl_en.fit(X_tmp_std, y_tmp, groups=omic_groups)
                tmp_sel_features = list(stabl_en.get_feature_names_out())
                fold_selected_features["STABL ElasticNet"].extend(tmp_sel_features)
                print(
                    f"STABL ElasticNet finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL ElasticNet"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_en.min_fdr_
                stabl_features_dict["STABL ElasticNet"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_en.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_en,
                        path=Path(save_path, "Training CV", f"STABL ElasticNet results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type,
                        override=True
                    )

            if "STABL XGBoost" in models and stabl_xgb is not None:
                # fit STABL XGBoost
                print("Fitting of STABL XGBoost")
                stabl_xgb.fit(X_tmp_std, y_tmp, groups=omic_groups)
                tmp_sel_features = list(stabl_xgb.get_feature_names_out())
                fold_selected_features["STABL XGBoost"].extend(tmp_sel_features)
                print(
                    f"STABL XGBoost finished on {omic_name} ({X_tmp.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                stabl_features_dict["STABL XGBoost"][omic_name].loc[f'Fold n°{k}', "min FDP+"] = stabl_xgb.min_fdr_
                stabl_features_dict["STABL XGBoost"][omic_name].loc[f'Fold n°{k}', "Threshold"] = stabl_xgb.fdr_min_threshold_
                if k == 1:
                    save_stabl_results(
                        stabl=stabl_xgb,
                        path=Path(save_path, "Training CV", f"STABL XGBoost results on {omic_name}"),
                        df_X=X_tmp_std,
                        y=y_tmp,
                        task_type=task_type,
                        override=True
                    )

            if "Lasso" in models:
                # __Lasso__
                print("Fitting of Lasso")
                fitted_model = clone(lasso)
                if task_type == "binary":
                    predictions = fitted_model.fit(X_tmp_std, y_tmp, groups=omic_groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = fitted_model.fit(X_tmp_std, y_tmp, groups=omic_groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(fitted_model.best_estimator_.coef_.flatten())])
                fold_selected_features["Lasso"].extend(tmp_sel_features)
                print(
                    f"Lasso finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["Lasso"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "ALasso" in models:
                # __ALasso__
                print("Fitting of ALasso")
                fitted_model = clone(alasso)
                if task_type == "binary":
                    predictions = fitted_model.fit(X_tmp_std, y_tmp, groups=omic_groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = fitted_model.fit(X_tmp_std, y_tmp, groups=omic_groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(fitted_model.best_estimator_.coef_.flatten())])
                fold_selected_features["ALasso"].extend(tmp_sel_features)
                print(
                    f"ALasso finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["ALasso"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "ElasticNet" in models:
                # __EN__
                print("Fitting of ElasticNet")
                fitted_model = clone(en)
                if task_type == "binary":
                    predictions = fitted_model.fit(X_tmp_std, y_tmp, groups=omic_groups).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = fitted_model.fit(X_tmp_std, y_tmp, groups=omic_groups).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[np.where(fitted_model.best_estimator_.coef_.flatten())])
                fold_selected_features["ElasticNet"].extend(tmp_sel_features)
                print(
                    f"ElasticNet finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict_late_fusion["ElasticNet"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "XGBoost" in models and xgb_estimator is not None:
                # __XGBoost__
                print("Fitting of XGBoost")
                fitted_model = clone(xgb_estimator)
                if task_type in ("binary", "binary_xgboost"):
                    predictions = fitted_model.fit(X_tmp_std, y_tmp).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = fitted_model.fit(X_tmp_std, y_tmp).predict(X_test_tmp_std)
                tmp_sel_features = list(X_tmp_std.columns[fitted_model.best_estimator_.feature_importances_ > 0])
                fold_selected_features["XGBoost"].extend(tmp_sel_features)
                print(
                    f"XGBoost finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features with non-zero importance"
                )
                predictions_dict_late_fusion["XGBoost"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "SVM" in models and svm is not None:
                # __SVM__
                print("Fitting of SVM")
                fitted_model = clone(svm)
                if task_type in ("binary", "binary_xgboost"):
                    predictions = fitted_model.fit(X_tmp_std, y_tmp).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = fitted_model.fit(X_tmp_std, y_tmp).predict(X_test_tmp_std)
                print(f"SVM finished on {omic_name} ({X_tmp_std.shape[0]} samples)")
                predictions_dict_late_fusion["SVM"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

            if "RandomForest" in models and random_forest is not None:
                # __RandomForest__
                print("Fitting of RandomForest")
                fitted_model = clone(random_forest)
                if task_type in ("binary", "binary_xgboost"):
                    predictions = fitted_model.fit(X_tmp_std, y_tmp).predict_proba(X_test_tmp_std)[:, 1]
                else:
                    predictions = fitted_model.fit(X_tmp_std, y_tmp).predict(X_test_tmp_std)
                threshold = np.percentile(fitted_model.best_estimator_.feature_importances_, 75)
                tmp_sel_features = list(X_tmp_std.columns[fitted_model.best_estimator_.feature_importances_ > threshold])
                fold_selected_features["RandomForest"].extend(tmp_sel_features)
                print(
                    f"RandomForest finished on {omic_name} ({X_tmp_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features in top 25% importance"
                )
                predictions_dict_late_fusion["RandomForest"][omic_name].loc[test_idx_tmp, f"Fold n°{k}"] = predictions

        for model in filter(lambda x: "STABL" in x, models):
            X_train = X_tot.loc[train_idx, fold_selected_features[model]]
            X_test = X_tot.loc[test_idx, fold_selected_features[model]]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            if len(fold_selected_features[model]) > 0:
                # Standardization
                std_pipe = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy="median")),
                        ('std', StandardScaler())
                    ]
                )

                X_train = pd.DataFrame(
                    data=std_pipe.fit_transform(X_train),
                    index=X_train.index,
                    columns=X_train.columns
                )
                X_test = pd.DataFrame(
                    data=std_pipe.transform(X_test),
                    index=X_test.index,
                    columns=X_test.columns
                )

                # __Final Models__
                if task_type == "binary":
                    predictions = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()

                elif task_type == "binary_xgboost":
                    predictions = clone(xgb_estimator).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()

                elif task_type == "regression":
                    predictions = clone(linreg).fit(X_train, y_train).predict(X_test)

                else:
                    raise ValueError("task_type not recognized.")

                predictions_dict[model].loc[test_idx, f"Fold n°{k}"] = predictions

            else:
                if task_type == "binary":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [0.5] * len(test_idx)

                elif task_type == "binary_xgboost":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [0.5] * len(test_idx)

                elif task_type == "regression":
                    predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = [np.mean(y_train)] * len(test_idx)

                else:
                    raise ValueError("task_type not recognized.")
        # __late fusion__
        if late_fusion:
            preds_lf = late_fusion_cv(
                predictions_dict_late_fusion, y[test_idx], task_type,
                Path(save_path, "Training CV"), n_iter=n_iter_lf
            )
            for model in preds_lf:
                predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = preds_lf[model]
        else:
            for model in [m for m in models if "STABL" not in m and "EF" not in m]:
                layer_dfs = [predictions_dict_late_fusion[model][omic] for omic in data_dict.keys()]
                avg = pd.concat(layer_dfs, axis=1).mean(axis=1)
                predictions_dict[model].loc[test_idx, f'Fold n°{k}'] = avg.loc[test_idx].values

        # __EF Lasso__
        if early_fusion:
            X_train = X_tot.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X_tot.loc[test_idx]
            X_train_std = pd.DataFrame(
                data=preprocessing.fit_transform(X_train),
                columns=preprocessing.get_feature_names_out(),
                index=X_train.index
            )

            X_test_std = pd.DataFrame(
                data=preprocessing.transform(X_test),
                columns=preprocessing.get_feature_names_out(),
                index=X_test.index
            )
            groups = outer_groups[train_idx] if outer_groups is not None else None

            if "EF Lasso" in models:
                # __Lasso__
                print("Fitting of EF Lasso")
                fitted_model = clone(lasso)
                if task_type == "binary":
                    predictions = fitted_model.fit(X_train_std, y_train, groups=groups).predict_proba(X_test_std)[:, 1]
                else:
                    predictions = fitted_model.fit(X_train_std, y_train, groups=groups).predict(X_test_std)
                tmp_sel_features = list(X_train_std.columns[np.where(fitted_model.best_estimator_.coef_.flatten())])
                fold_selected_features["EF Lasso"] = tmp_sel_features
                print(
                    f"EF Lasso finished on all omics ({X_train_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict["EF Lasso"].loc[test_idx, f"Fold n°{k}"] = predictions

            if "EF ALasso" in models:
                # __ALasso__
                print("Fitting of EF ALasso")
                fitted_model = clone(alasso)
                if task_type == "binary":
                    predictions = fitted_model.fit(X_train_std, y_train, groups=groups).predict_proba(X_test_std)[:, 1]
                else:
                    predictions = fitted_model.fit(X_train_std, y_train, groups=groups).predict(X_test_std)
                tmp_sel_features = list(X_train_std.columns[np.where(fitted_model.best_estimator_.coef_.flatten())])
                fold_selected_features["EF ALasso"] = tmp_sel_features
                print(
                    f"EF ALasso finished on all omics ({X_train_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict["EF ALasso"].loc[test_idx, f"Fold n°{k}"] = predictions

            if "EF ElasticNet" in models:
                # __EN__
                print("Fitting of EF ElasticNet")
                fitted_model = clone(en)
                if task_type == "binary":
                    predictions = fitted_model.fit(X_train_std, y_train, groups=groups).predict_proba(X_test_std)[:, 1]
                else:
                    predictions = fitted_model.fit(X_train_std, y_train, groups=groups).predict(X_test_std)
                tmp_sel_features = list(X_train_std.columns[np.where(fitted_model.best_estimator_.coef_.flatten())])
                fold_selected_features["EF ElasticNet"] = tmp_sel_features
                print(
                    f"EF ElasticNet finished on all omics ({X_train_std.shape[0]} samples);"
                    f" {len(tmp_sel_features)} features selected"
                )
                predictions_dict["EF ElasticNet"].loc[test_idx, f"Fold n°{k}"] = predictions

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for model in models:
            print(f"This fold: {len(fold_selected_features[model])} features selected for {model}")
            selected_features_dict[model].append(fold_selected_features[model])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        if task_type == "binary":
            auc_row = {"fold": k}
            for model in models:
                cumulative_preds = predictions_dict[model].dropna(how="all").median(axis=1)
                valid_idx = cumulative_preds.index[cumulative_preds.notna()]
                if len(valid_idx) > 0 and len(y.loc[valid_idx].unique()) > 1:
                    try:
                        auc_row[model] = round(roc_auc_score(y.loc[valid_idx], cumulative_preds.loc[valid_idx]), 3)
                    except Exception:
                        auc_row[model] = None
            auc_progress_path = Path(save_path, "Training CV", "auc_progress.csv")
            auc_df = pd.read_csv(auc_progress_path) if auc_progress_path.exists() else pd.DataFrame()
            auc_df = pd.concat([auc_df, pd.DataFrame([auc_row])], ignore_index=True)
            auc_df.to_csv(auc_progress_path, index=False)

        k += 1

    # __SAVING_CORR_FILTER__
    for omic_name, rows in corr_filter_rows.items():
        corr_dir = Path(save_path, "Training CV", f"CorrelationFilter {omic_name}")
        os.makedirs(corr_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(Path(corr_dir, "components.csv"), index=False)

    # __SAVING_RESULTS__
    print("Saving results...")
    if y.name is None:
        y.name = "outcome"

    summary_res_path = Path(save_path, "Summary")
    cv_res_path = Path(save_path, "Training CV")

    jaccard_matrix_dict = dict()
    formatted_features_dict = dict()

    for model in models:

        jaccard_matrix_dict[model] = jaccard_matrix(selected_features_dict[model])

        formatted_features_dict[model] = pd.DataFrame(
            data={
                "Fold selected features": selected_features_dict[model],
                "Fold nb of features": [len(el) for el in selected_features_dict[model]]
            },
            index=[f"Fold {i}" for i in range(outer_splitter.get_n_splits(X=X_tot))]
        )

        from collections import Counter
        n_folds = outer_splitter.get_n_splits(X=X_tot)
        counts = Counter()
        for fold_features in selected_features_dict[model]:
            counts.update(fold_features)
        freq_rows = sorted(counts.items(), key=lambda x: -x[1])
        freq_df = pd.DataFrame(freq_rows, columns=["feature", "n_selections"])
        freq_df["fraction_folds"] = (freq_df["n_selections"] / n_folds).round(3)
        freq_df.to_csv(Path(cv_res_path, f"Selected Features {model}.csv"), index=False)
        if "STABL" in model:
            for omic_name, val in stabl_features_dict[model].items():
                os.makedirs(Path(cv_res_path, f"Stabl features {model}"), exist_ok=True)
                val.to_csv(Path(cv_res_path, f"Stabl features {model}", f"Stabl features {model} {omic_name}.csv"))

    predictions_dict = {model: predictions_dict[model].median(axis=1) for model in predictions_dict.keys()}

    table_of_scores = compute_scores_table(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        selected_features_dict=formatted_features_dict
    )

    p_values = compute_pvalues_table(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        selected_features_dict=formatted_features_dict
    )

    table_of_scores.to_csv(Path(summary_res_path, "Scores training CV.csv"))
    table_of_scores.to_csv(Path(cv_res_path, "Scores training CV.csv"))

    p_values_path = Path(cv_res_path, "p-values")
    os.makedirs(p_values_path, exist_ok=True)
    for m, p in p_values.items():
        p.to_csv(Path(p_values_path, f"{m}.csv"))

    save_plots(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        save_path=cv_res_path
    )

    return predictions_dict


def multi_omic_stabl(
        data_dict,
        y,
        estimators,
        task_type,
        save_path,
        models,
        stabl_params=None,
        groups=None,
        early_fusion=False,
        X_test=None,
        y_test=None,
        n_iter_lf=10000
):
    """
    Performs a cross validation on the data_dict using the models and saves the results in save_path.

    Parameters
    ----------
    data_dict: dict
        Dictionary containing the input omic-files.

    y: pd.Series
        pandas Series containing the outcomes for the use case. Note that y should contains the union of outcomes for
        the data_dict.

    estimators : dict[str, sklearn estimator]
        Dict of feature selectors to benchmark with their name as key. They must implement `fit`, `get_support(indices=True)`. It should contain :
        - "lasso" : Lasso in GridSearchCV
        - "alasso" : ALasso in GridSearchCV
        - "en" : ElasticNet in GridSearchCV
        - "sgl" : SGL in GridSearchCV
        - "stabl_lasso" : Stabl with Lasso as base estimator
        - "stabl_alasso" : Stabl with ALasso as base estimator
        - "stabl_en" : Stabl with ElasticNet as base estimator
        - "stabl_sgl" : Stabl with SGL as base estimator

    task_type: str
        Can either be "binary" for binary classification or "regression" for regression tasks.

    save_path: Path or str
        Where to save the results

    models: list of str
        List of models to use. Can contain :
        - "Lasso" : Lasso
        - "STABL Lasso" : Stabl with Lasso as base estimator
        - "ALasso" : ALasso
        - "STABL ALasso" : Stabl with ALasso as base estimator
        - "ElasticNet" : ElasticNet
        - "STABL ElasticNet" : Stabl with ElasticNet as base estimator
        - "SGL-90" : SGL with 0.90 as correlation threshold
        - "STABL SGL-90" : Stabl with SGL with 0.90 as correlation threshold as base estimator
        - "SGL-95" : SGL with 0.95 as correlation threshold
        - "STABL SGL-95" : Stabl with SGL with 0.95 as correlation threshold as base estimator

    stabl_params: dict, default=None
        Dictionary containing the parameters to use for STABL. It overrides default settings. It is used to change Stabl parameters between omics. 
        It is of the form :
        {
            "STABL model" : {
                "omic_name" : dict of lambda_grid
            }
        }

    groups: pd.Series, default=None
        If used, should be the same size as y and should indicate the groups of the samples.

    early_fusion: bool, default=False
        If True, it will perform early fusion for each estimator.

    X_test: dict or None, default=None
        Dictionary containing the test omic-files.

    y_test: pd.Series or None, default=None
        pandas Series containing the outcomes for the use case. Note that y should contains the union of outcomes for X_test.

    n_iter_lf: int, default=10000
        Number of iterations for the late fusion.

    Returns
    -------
    predictions_dict: dict
        Dictionary containing the predictions of each model for each sample of X_test.
    """

    if stabl_params is None:
        stabl_params = {}

    if early_fusion:
        models += ["EF " + model for model in models if "STABL" not in model]

    lasso = estimators["lasso"]
    alasso = estimators["alasso"]
    en = estimators["en"]
    xgb_estimator = estimators.get("xgboost", None)
    svm = estimators.get("svm", None)
    random_forest = estimators.get("random_forest", None)

    stabl = estimators["stabl_lasso"]
    stabl_alasso = estimators["stabl_alasso"]
    stabl_en = estimators["stabl_en"]
    stabl_xgb = estimators.get("stabl_xgboost", None)

    preprocessing = _make_preprocessing_pipeline()

    os.makedirs(Path(save_path, "Training-Validation"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"), exist_ok=True)

    # Initializing the df containing the data of all omics
    X_tot = pd.concat(data_dict.values(), axis="columns")
    if X_test is not None:
        X_test_tot = pd.concat(X_test.values(), axis="columns")
        X_test_tot = X_test_tot.loc[y_test.index]

    predictions_dict = dict()
    selected_features_dict = dict()

    for model in models:
        selected_features_dict[model] = []

    predictions_dict_train_late_fusion = dict()
    for model in models:
        if "STABL" not in model and "EF" not in model:
            predictions_dict_train_late_fusion[model] = pd.DataFrame(data=None, columns=data_dict.keys(), index=y.index)
    predictions_dict_test_late_fusion = dict()
    if X_test is not None and y_test is not None:
        for model in models:
            if "STABL" not in model and "EF" not in model:
                predictions_dict_test_late_fusion[model] = pd.DataFrame(data=None, columns=X_test.keys(), index=y_test.index)

    for omic_name, X_omic in data_dict.items():
        all_columns = X_omic.columns
        if X_test is not None:
            all_columns = all_columns.intersection(X_test[omic_name].columns)

        # if prepro is not None:
        #     preprocessing = prepro

        X_omic_std = pd.DataFrame(
            data=preprocessing.fit_transform(X_omic[all_columns]),
            index=X_omic.index,
            columns=preprocessing.get_feature_names_out()
        )
        y_omic = y.loc[X_omic_std.index]
        if X_test is not None:
            X_test_omic_std = pd.DataFrame(
                data=preprocessing.transform(X_test[omic_name][all_columns]),
                index=X_test[omic_name].index,
                columns=preprocessing.get_feature_names_out()
            )

        # __STABL__
        if "STABL Lasso" in models:
            # fit STABL Lasso
            print(f"Fitting of STABL Lasso on {omic_name}")
            if "STABL Lasso" in stabl_params and omic_name in stabl_params["STABL Lasso"]:
                stabl.set_params(lambda_grid=stabl_params["STABL Lasso"][omic_name])
            stabl.fit(X_omic_std, y_omic, groups=groups)
            tmp_sel_features = list(stabl.get_feature_names_out())
            selected_features_dict["STABL Lasso"].extend(tmp_sel_features)
            print(
                f"STABL Lasso finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            save_stabl_results(
                stabl=stabl,
                path=Path(save_path, "Training-Validation", f"STABL Lasso results on {omic_name}"),
                df_X=X_omic,
                y=y_omic,
                task_type=task_type,
                override=True
            )

        if "STABL ALasso" in models:
            # fit STABL ALasso
            print(f"Fitting of STABL ALasso on {omic_name}")
            if "STABL ALasso" in stabl_params and omic_name in stabl_params["STABL ALasso"]:
                stabl_alasso.set_params(lambda_grid=stabl_params["STABL ALasso"][omic_name])
            stabl_alasso.fit(X_omic_std, y_omic, groups=groups)
            tmp_sel_features = list(stabl_alasso.get_feature_names_out())
            selected_features_dict["STABL ALasso"].extend(tmp_sel_features)
            print(
                f"STABL ALasso finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            save_stabl_results(
                stabl=stabl_alasso,
                path=Path(save_path, "Training-Validation", f"STABL ALasso results on {omic_name}"),
                df_X=X_omic,
                y=y_omic,
                task_type=task_type,
                override=True
            )

        if "STABL ElasticNet" in models:
            # fit STABL ElasticNet
            print(f"Fitting of STABL ElasticNet on {omic_name}")
            if "STABL ElasticNet" in stabl_params and omic_name in stabl_params["STABL ElasticNet"]:
                stabl_en.set_params(lambda_grid=stabl_params["STABL ElasticNet"][omic_name])
            stabl_en.fit(X_omic_std, y_omic, groups=groups)
            tmp_sel_features = list(stabl_en.get_feature_names_out())
            selected_features_dict["STABL ElasticNet"].extend(tmp_sel_features)
            print(
                f"STABL ElasticNet finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            save_stabl_results(
                stabl=stabl_en,
                path=Path(save_path, "Training-Validation", f"STABL ElasticNet results on {omic_name}"),
                df_X=X_omic,
                y=y_omic,
                task_type=task_type,
                override=True
            )

        if "STABL XGBoost" in models and stabl_xgb is not None:
            # fit STABL XGBoost
            print(f"Fitting of STABL XGBoost on {omic_name}")
            if "STABL XGBoost" in stabl_params and omic_name in stabl_params["STABL XGBoost"]:
                stabl_xgb.set_params(lambda_grid=stabl_params["STABL XGBoost"][omic_name])
            stabl_xgb.fit(X_omic_std, y_omic, groups=groups)
            tmp_sel_features = list(stabl_xgb.get_feature_names_out())
            selected_features_dict["STABL XGBoost"].extend(tmp_sel_features)
            print(
                f"STABL XGBoost finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            save_stabl_results(
                stabl=stabl_xgb,
                path=Path(save_path, "Training-Validation", f"STABL XGBoost results on {omic_name}"),
                df_X=X_omic,
                y=y_omic,
                task_type=task_type,
                override=True
            )

        if "Lasso" in models:
            # __Lasso__
            print(f"Fitting of Lasso on {omic_name}")
            model = clone(lasso)
            try:
                model = model.fit(X_omic_std, y_omic, groups=groups)
            except:
                model = model.fit(X_omic_std, y_omic)

            model = model.best_estimator_

            if task_type == "binary":
                predictions = model.predict_proba(X_omic_std)[:, 1]
            else:
                predictions = model.predict(X_omic_std)

            tmp_sel_features = list(X_omic_std.columns[np.where(model.coef_.flatten())])
            selected_features_dict["Lasso"].extend(tmp_sel_features)
            print(
                f"Lasso finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )

            base_linear_model_coef = pd.DataFrame(
                {"Feature": tmp_sel_features,
                 "Associated weight": model.coef_.flatten()[np.where(model.coef_.flatten())]
                 }
            ).set_index("Feature")
            base_linear_model_coef.to_csv(Path(save_path, "Training-Validation", f"Lasso coefficients {omic_name}.csv"))

            predictions_dict_train_late_fusion["Lasso"].loc[X_omic.index, omic_name] = predictions
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_omic_std)[:, 1]
                else:
                    predictions = model.predict(X_test_omic_std)
                predictions_dict_test_late_fusion["Lasso"].loc[X_test_omic_std.index, omic_name] = predictions
                predictions_dict_test_late_fusion["Lasso"].fillna(np.median(predictions_dict_train_late_fusion["Lasso"]), inplace=True)

        if "ALasso" in models:
            # __ALasso__
            print(f"Fitting of ALasso on {omic_name}")
            model = clone(alasso)
            try:
                model = model.fit(X_omic_std, y_omic, groups=groups)
            except:
                model = model.fit(X_omic_std, y_omic)

            model = model.best_estimator_

            if task_type == "binary":
                predictions = model.predict_proba(X_omic_std)[:, 1]
            else:
                predictions = model.predict(X_omic_std)

            tmp_sel_features = list(X_omic_std.columns[np.where(model.coef_.flatten())])
            selected_features_dict["ALasso"].extend(tmp_sel_features)
            print(
                f"ALasso finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )

            base_linear_model_coef = pd.DataFrame(
                {"Feature": tmp_sel_features,
                 "Associated weight": model.coef_.flatten()[np.where(model.coef_.flatten())]
                 }
            ).set_index("Feature")
            base_linear_model_coef.to_csv(Path(save_path, "Training-Validation", f"ALasso coefficients {omic_name}.csv"))

            predictions_dict_train_late_fusion["ALasso"].loc[X_omic.index, omic_name] = predictions
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_omic_std)[:, 1]
                else:
                    predictions = model.predict(X_test_omic_std)
                predictions_dict_test_late_fusion["ALasso"].loc[X_test_omic_std.index, omic_name] = predictions
                predictions_dict_test_late_fusion["ALasso"].fillna(np.median(predictions_dict_train_late_fusion["ALasso"]), inplace=True)

        if "ElasticNet" in models:
            # __EN__
            print(f"Fitting of ElasticNet on {omic_name}")
            model = clone(en)
            try:
                model = model.fit(X_omic_std, y_omic, groups=groups)
            except:
                model = model.fit(X_omic_std, y_omic)

            model = model.best_estimator_

            if task_type == "binary":
                predictions = model.predict_proba(X_omic_std)[:, 1]
            else:
                predictions = model.predict(X_omic_std)

            tmp_sel_features = list(X_omic_std.columns[np.where(model.coef_.flatten())])
            selected_features_dict["ElasticNet"].extend(tmp_sel_features)
            print(
                f"ElasticNet finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )

            base_linear_model_coef = pd.DataFrame(
                {"Feature": tmp_sel_features,
                 "Associated weight": model.coef_.flatten()[np.where(model.coef_.flatten())]
                 }
            ).set_index("Feature")
            base_linear_model_coef.to_csv(
                Path(save_path, "Training-Validation", f"ElasticNet coefficients {omic_name}.csv")
            )

            predictions_dict_train_late_fusion["ElasticNet"].loc[X_omic.index, omic_name] = predictions
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_omic_std)[:, 1]
                else:
                    predictions = model.predict(X_test_omic_std)
                predictions_dict_test_late_fusion["ElasticNet"].loc[X_test_omic_std.index, omic_name] = predictions
                predictions_dict_test_late_fusion["ElasticNet"].fillna(
                    np.median(predictions_dict_train_late_fusion["ElasticNet"]), inplace=True
                )

        if "XGBoost" in models and xgb_estimator is not None:
            # __XGBoost__
            print(f"Fitting of XGBoost on {omic_name}")
            model = clone(xgb_estimator)
            try:
                model = model.fit(X_omic_std, y_omic, groups=groups)
            except:
                model = model.fit(X_omic_std, y_omic)

            model = model.best_estimator_

            if task_type == "binary":
                predictions = model.predict_proba(X_omic_std)[:, 1]
            else:
                predictions = model.predict(X_omic_std)

            tmp_sel_features = list(
                X_omic_std.columns[model.feature_importances_ > 0]
            )
            selected_features_dict["XGBoost"].extend(tmp_sel_features)
            print(
                f"XGBoost finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features with non-zero importance"
            )

            feature_importances = pd.DataFrame(
                {"Feature": X_omic_std.columns, "Importance": model.feature_importances_}
            ).query("Importance > 0").set_index("Feature")
            feature_importances.to_csv(
                Path(save_path, "Training-Validation", f"XGBoost feature importances {omic_name}.csv")
            )

            predictions_dict_train_late_fusion["XGBoost"].loc[X_omic.index, omic_name] = predictions
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_omic_std)[:, 1]
                else:
                    predictions = model.predict(X_test_omic_std)
                predictions_dict_test_late_fusion["XGBoost"].loc[X_test_omic_std.index, omic_name] = predictions
                predictions_dict_test_late_fusion["XGBoost"].fillna(
                    np.median(predictions_dict_train_late_fusion["XGBoost"]), inplace=True
                )

        if "SVM" in models and svm is not None:
            # __SVM__
            print(f"Fitting of SVM on {omic_name}")
            model = clone(svm)
            try:
                model = model.fit(X_omic_std, y_omic, groups=groups)
            except:
                model = model.fit(X_omic_std, y_omic)

            model = model.best_estimator_

            if task_type == "binary":
                predictions = model.predict_proba(X_omic_std)[:, 1]
            else:
                predictions = model.predict(X_omic_std)

            print(f"SVM finished on {omic_name} ({X_omic_std.shape[0]} samples)")

            predictions_dict_train_late_fusion["SVM"].loc[X_omic.index, omic_name] = predictions
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_omic_std)[:, 1]
                else:
                    predictions = model.predict(X_test_omic_std)
                predictions_dict_test_late_fusion["SVM"].loc[X_test_omic_std.index, omic_name] = predictions
                predictions_dict_test_late_fusion["SVM"].fillna(
                    np.median(predictions_dict_train_late_fusion["SVM"]), inplace=True
                )

        if "RandomForest" in models and random_forest is not None:
            # __RandomForest__
            print(f"Fitting of RandomForest on {omic_name}")
            model = clone(random_forest)
            try:
                model = model.fit(X_omic_std, y_omic, groups=groups)
            except:
                model = model.fit(X_omic_std, y_omic)

            model = model.best_estimator_

            if task_type == "binary":
                predictions = model.predict_proba(X_omic_std)[:, 1]
            else:
                predictions = model.predict(X_omic_std)

            threshold = np.percentile(model.feature_importances_, 75)
            tmp_sel_features = list(
                X_omic_std.columns[model.feature_importances_ > threshold]
            )
            selected_features_dict["RandomForest"].extend(tmp_sel_features)
            print(
                f"RandomForest finished on {omic_name} ({X_omic_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features in top 25% importance"
            )

            feature_importances = pd.DataFrame(
                {"Feature": X_omic_std.columns, "Importance": model.feature_importances_}
            ).query("Importance > 0").set_index("Feature").sort_values("Importance", ascending=False)
            feature_importances.to_csv(
                Path(save_path, "Training-Validation", f"RandomForest feature importances {omic_name}.csv")
            )

            predictions_dict_train_late_fusion["RandomForest"].loc[X_omic.index, omic_name] = predictions
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_omic_std)[:, 1]
                else:
                    predictions = model.predict(X_test_omic_std)
                predictions_dict_test_late_fusion["RandomForest"].loc[X_test_omic_std.index, omic_name] = predictions
                predictions_dict_test_late_fusion["RandomForest"].fillna(
                    np.median(predictions_dict_train_late_fusion["RandomForest"]), inplace=True
                )

    final_prepro = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("std", StandardScaler())]
    )

    for model in filter(lambda x: "STABL" in x, models):
        if len(selected_features_dict[model]) > 0:
            X_train = X_tot[selected_features_dict[model]]
            X_train_std = pd.DataFrame(
                data=final_prepro.fit_transform(X_train),
                index=X_tot.index,
                columns=final_prepro.get_feature_names_out()
            )

            if task_type == "binary":
                base_linear_model = logit

            elif task_type == "binary_xgboost":
                base_linear_model = xgb_estimator

            else:
                base_linear_model = linreg

            base_linear_model.fit(X_train_std, y)
            base_linear_model_coef = pd.DataFrame(
                {"Feature": selected_features_dict[model],
                 "Associated weight": base_linear_model.coef_.flatten()
                 }
            ).set_index("Feature")
            base_linear_model_coef.to_csv(Path(save_path, "Training-Validation", f"{model} coefficients.csv"))

            if X_test is not None:
                X_test_std = pd.DataFrame(
                    data=final_prepro.transform(X_test_tot[selected_features_dict[model]]),
                    index=X_test_tot.index,
                    columns=final_prepro.get_feature_names_out()
                )
                if task_type == "binary":
                    model_preds = base_linear_model.predict_proba(X_test_std)[:, 1]
                else:
                    model_preds = base_linear_model.predict(X_test_std)
        else:
            if X_test is not None:
                model_preds = np.zeros(X_test_tot.shape[0])
                if task_type == "binary":
                    model_preds[:] = 0.5
                elif task_type == "regression":
                    model_preds[:] = np.mean(y)
        if X_test is not None:
            predictions_dict[model] = pd.Series(
                model_preds,
                index=y_test.index,
                name=f"{model} predictions"
            )

    # __late fusion__
    preds_lf = late_fusion_validation(
        predictions_dict_train_late_fusion, predictions_dict_test_late_fusion, y,
        task_type, Path(save_path, "Training-Validation"), n_iter=n_iter_lf
    )
    if preds_lf is not None:
        for model in preds_lf:
            predictions_dict[model] = preds_lf[model]

    # __EF Lasso__
    if early_fusion:
        X_train_std = pd.DataFrame(
            data=preprocessing.fit_transform(X_tot),
            index=X_tot.index,
            columns=preprocessing.get_feature_names_out()
        )

        if X_test is not None:
            X_test_std = pd.DataFrame(
                data=preprocessing.transform(X_test_tot),
                index=X_test_tot.index,
                columns=preprocessing.get_feature_names_out()
            )

        if "EF Lasso" in models:
            # __Lasso__
            print("Fitting of EF Lasso")
            model = clone(lasso)
            try:
                model.fit(X_train_std, y, groups=groups)
            except:
                model.fit(X_train_std, y)
            tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
            selected_features_dict["EF Lasso"] = tmp_sel_features

            model_coef = pd.DataFrame(
                {"Feature": selected_features_dict["EF Lasso"],
                 "Associated weight": model.best_estimator_.coef_.flatten()[np.where(model.best_estimator_.coef_.flatten())]
                 }
            ).set_index("Feature")
            model_coef.to_csv(Path(save_path, "Training-Validation", "EF Lasso coefficients.csv"))

            print(
                f"EF Lasso finished on all omics ({X_train_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.predict(X_test_std)
                predictions_dict["EF Lasso"] = pd.Series(
                    predictions,
                    index=y_test.index,
                    name=f"EF Lasso predictions"
                )

        if "EF ALasso" in models:
            # __ALasso__
            print("Fitting of EF ALasso")
            model = clone(alasso)
            try:
                model.fit(X_train_std, y, groups=groups)
            except:
                model.fit(X_train_std, y)
            tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
            selected_features_dict["EF ALasso"] = tmp_sel_features

            model_coef = pd.DataFrame(
                {"Feature": selected_features_dict["EF ALasso"],
                 "Associated weight": model.best_estimator_.coef_.flatten()[np.where(model.best_estimator_.coef_.flatten())]
                 }
            ).set_index("Feature")
            model_coef.to_csv(Path(save_path, "Training-Validation", "EF ALasso coefficients.csv"))

            print(
                f"EF ALasso finished on all omics ({X_train_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.predict(X_test_std)
                predictions_dict["EF ALasso"] = pd.Series(
                    predictions,
                    index=y_test.index,
                    name=f"EF ALasso predictions"
                )

        if "EF ElasticNet" in models:
            # __EN__
            print("Fitting of EF ElasticNet")
            model = clone(en)
            try:
                model.fit(X_train_std, y, groups=groups)
            except:
                model.fit(X_train_std, y)
            tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
            selected_features_dict["EF ElasticNet"] = tmp_sel_features

            model_coef = pd.DataFrame(
                {"Feature": selected_features_dict["EF ElasticNet"],
                 "Associated weight": model.best_estimator_.coef_.flatten()[np.where(model.best_estimator_.coef_.flatten())]
                 }
            ).set_index("Feature")
            model_coef.to_csv(Path(save_path, "Training-Validation", "EF ElasticNet coefficients.csv"))

            print(
                f"EF ElasticNet finished on all omics ({X_train_std.shape[0]} samples);"
                f" {len(tmp_sel_features)} features selected"
            )
            if X_test is not None:
                if task_type == "binary":
                    predictions = model.predict_proba(X_test_std)[:, 1]
                else:
                    predictions = model.predict(X_test_std)
                predictions_dict["EF ElasticNet"] = pd.Series(
                    predictions,
                    index=y_test.index,
                    name=f"EF ElasticNet predictions"
                )

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for model in models:
        print(f"This fold: {len(selected_features_dict[model])} features selected for {model}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    if X_test is not None:
        table_of_scores = compute_scores_table(
            predictions_dict=predictions_dict,
            y=y_test,
            task_type=task_type
        )
        p_values = compute_pvalues_table(
            predictions_dict=predictions_dict,
            y=y_test,
            task_type=task_type
        )

        table_of_scores.to_csv(Path(save_path, "Training-Validation", "Scores on Validation.csv"))
        table_of_scores.to_csv(Path(save_path, "Summary", "Scores on Validation.csv"))

        p_values_path = Path(Path(save_path, "Training-Validation"), "p-values")
        os.makedirs(p_values_path, exist_ok=True)
        for m, p in p_values.items():
            p.to_csv(Path(p_values_path, f"{m}.csv"))

        save_plots(
            predictions_dict=predictions_dict,
            y=y_test,
            task_type=task_type,
            save_path=Path(save_path, "Training-Validation"),
        )

    return predictions_dict


def late_fusion_cv(
    predictions_lf_dict,
    y,
    task_type,
    save_path,
    n_iter=10000
):
    """
    Perform a late fusion of omics using the prediction of each model on the train set.
    It uses stacked_multi_omic to perform the late fusion.

    Parameters
    ----------
    predictions_lf_dict: dict[str, pd.DataFrame]
        Dictionary containing the prediction on each omic.

    y: pd.Series
        pandas Series containing the outcomes for the use case. Note that y should contains the union of outcomes for
        the data_dict.

    task_type: str
        Can either be "binary" for binary classification or "regression" for regression tasks.

    save_path: Path or str
        Where to save the results

    n_iter: int
        Number of iterations for the late fusion.
    Returns
    -------
    final_predictions_dict: dict of pd.DataFrame
        Dictionary containing the final predictions of each model after late fusion.
    """
    final_predictions_dict = {}
    os.makedirs(save_path, exist_ok=True)
    for model_name, predictions in (tmodel := tqdm(predictions_lf_dict.items(), total=len(predictions_lf_dict))):
        tmodel.set_description(f"Late Fusion {model_name}")

        model_lf_path = Path(save_path, model_name)
        os.makedirs(model_lf_path, exist_ok=True)

        preds_omics = pd.DataFrame(data=None, columns=predictions.keys())
        for omic_name, preds in predictions.items():
            preds_omics[omic_name] = preds.median(axis=1)
        stacked_df, weights = stacked_multi_omic(preds_omics, y, task_type, n_iter=n_iter)
        weights.to_csv(Path(model_lf_path, f"Associated weights {model_name}.csv"))
        stacked_df.to_csv(
            Path(model_lf_path, f"Stacked Generalization predictions {model_name}.csv"))
        final_predictions_dict[model_name] = stacked_df["Stacked Gen. Predictions"]
    return final_predictions_dict


def late_fusion_validation(
    predictions_train_dict,
    predictions_valid_dict,
    y,
    task_type,
    save_path,
    n_iter
):
    """Perform a late fusion of omics using the prediction of each model on the train set.
    It uses stacked_multi_omic to perform the late fusion.

    Parameters
    ----------
    predictions_train_dict : dict of pd.DataFrame
        Predictions of each model on each omic of the train set.
    predictions_valid_dict : dict of pd.DataFrame or None
        Predictions of each model on each omic of the test set.
    y : pd.Series
        Outcome of the train set.
    task_type : str
        Task type of the use case. Can be "binary" or "regression".
    save_path : str or Path
        Path where to save the results.
    n_iter : int
        Number of iterations for the late fusion.

    Returns
    -------
    train_predictions_dict: dict of pd.DataFrame
        Predictions of each model on the train set after late fusion.

    final_predictions_dict: dict of pd.DataFrame or None
        Predictions of each model on the test set after late fusion.
        None if predictions_valid_dict is None.
    """
    final_predictions_dict = {}
    os.makedirs(save_path, exist_ok=True)
    for model_name, predictions in (tmodel := tqdm(predictions_train_dict.items(), total=len(predictions_train_dict))):
        tmodel.set_description(f"Late Fusion {model_name}")

        model_lf_path = Path(save_path, model_name)
        os.makedirs(model_lf_path, exist_ok=True)
        stacked_df, weights = stacked_multi_omic(predictions, y, task_type, n_iter=n_iter)
        weights.to_csv(Path(model_lf_path, f"Associated weights {model_name}.csv"))
        stacked_df.to_csv(
            Path(model_lf_path, f"Stacked Generalization predictions train {model_name}.csv"))
        if predictions_valid_dict:
            valid_preds = predictions_valid_dict[model_name]
            final_predictions_dict[model_name] = ((valid_preds * weights["Associated weight"]).sum(axis=1) /
                                                  weights["Associated weight"].sum())
    if predictions_valid_dict:
        return final_predictions_dict
    else:
        return None
