import warnings
# Suppress XGBoost glibc FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*glibc.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
from .unionfind import UnionFind
import sys
from tqdm.autonotebook import tqdm
from .preprocessing import remove_low_info_samples
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor,XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn import clone
from scipy.optimize import nnls
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error, mean_absolute_error
from .utils import compute_CI
from .metrics import jaccard_matrix
from .visualization import boxplot_binary_predictions, plot_roc, scatterplot_regression_predictions
from pathlib import Path


logit = LogisticRegression(C=np.inf, class_weight="balanced", max_iter=int(1e6))
linreg = LinearRegression()
randomforest=RandomForestRegressor(n_estimators=200, max_depth=5, n_jobs=1)
randomforest_class=RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=1)
xgboost = XGBRegressor(n_jobs =1)
xgboost_class=XGBClassifier(n_jobs =1)

std_pipe = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ('std', StandardScaler())
    ]
)

fromPreprocessing = lambda data,prepro: pd.DataFrame(data=prepro.fit_transform(data),index=data.index,columns=prepro.get_feature_names_out())
fromPreprocessingRep = lambda data,prepro: pd.DataFrame(data=prepro.transform(data),index=data.index,columns=prepro.get_feature_names_out())

@ignore_warnings(category=ConvergenceWarning)
def single_omic_simple(
        data,
        y,
        outer_splitter,
        estimator,
        estimator_name,
        preprocessing,
        task_type,
        ef = False,
        outer_groups=None
):
    """
    Performs a cross validation on the data_dict using the models and saves the results in save_path.

    Parameters
    ----------
    data: pd.DataFrame
        pandas DataFrames containing the data

    y: pd.Series
        pandas Series containing the outcomes for the use case. Note that y should contain the union of outcomes for
        the data_dict.

    outer_splitter: sklearn.model_selection._split.BaseCrossValidator
        Outer cross validation splitter

    estimator : sklearn estimator
        One of the following, with the corresponding estimator_name:
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

    outer_groups: pd.Series, default=None
        If used, should be the same size as y and should indicate the groups of the samples.

    ef: bool, default=False
        If True, doesnt return the insample predictions for non-stabl models.

    Returns
    -------
    predictions: pandas DataFrame
        DataFrame containing the predictions of the model for each sample in Cross-Validation.
    """

    stablFlag = "stabl" in estimator_name
    n_folds = outer_splitter.get_n_splits(X=data, y=y, groups=outer_groups)
    foldIdx = [f"Fold_{i+1}" for i in range(n_folds)]

    predictions_list = []
    selected_features_list = []
    if stablFlag:
        stabl_features_list = []
        predictions_list_xgboost = []
        predictions_list_rf = []
    else:
        best_params = []
        if not ef:
            insamplePredictions_list = []
    # predictions = pd.DataFrame(index=y.index, columns=foldIdx,dtype=float)
    # selected_features= pd.DataFrame(data=False, columns=data.columns, index=foldIdx)
    # if stablFlag:
    #     stabl_features= pd.DataFrame( columns=["Threshold", "min FDP+"], index=foldIdx)
    # else:
    #     best_params = []
    #     if not ef:
    #         insamplePredictions = pd.DataFrame(index=y.index, columns=foldIdx, dtype = float)

    k = 1
    for train, test in (tbar := tqdm(
            outer_splitter.split(data, y, groups=outer_groups),
            total=n_folds,
            file=sys.stdout
    )):
        # train_idx, test_idx = y.iloc[train].index, y.iloc[test].index
        groups = outer_groups.iloc[train].values if outer_groups is not None else None


        X_train = data.iloc[train,:]
        X_test = data.iloc[test,:]
        y_train = y.iloc[train]

        X_train_std = fromPreprocessing(X_train,preprocessing)
        X_test_std = fromPreprocessingRep(X_test,preprocessing)

        fold_predictions = np.full(len(y), np.nan)
        fold_selected = np.zeros(len(data.columns), dtype=bool)
        
        if stablFlag:
            fold_predictions_xgboost = np.full(len(y), np.nan)
            fold_predictions_rf = np.full(len(y), np.nan)

        if estimator_name in ["lasso", "alasso","en"]: 
            model = clone(estimator)
            model.fit(X_train_std, y_train, groups=groups)
            if task_type == "binary":
                pred = model.predict_proba(X_test_std)[:, 1]
                if not ef:
                    insamplePreds = model.predict_proba(X_train_std)[:, 1]
            else:
                pred = model.predict(X_test_std)
                if not ef:
                    insamplePreds = model.predict(X_train_std)
            if not ef:
                # insamplePredictions.loc[train_idx,f'Fold_{k}'] = insamplePreds 
                insample_fold = np.full(len(y), np.nan)
                insample_fold[train] = insamplePreds
                insamplePredictions_list.append(insample_fold)
            tmp_sel_features = np.where(model.best_estimator_.coef_.flatten())[0]
            fold_selected[tmp_sel_features] = True
            fold_predictions[test] = pred
            # tmp_sel_features = list(X_train_std.columns[np.where(model.best_estimator_.coef_.flatten())])
            # fold_selected_features.extend(tmp_sel_features)
            # predictions.loc[test_idx, f"Fold_{k}"] = pred
            best_params.append(model.best_params_)

        # __STABL__
        if stablFlag:
            estimator.fit(X_train_std, y_train, groups=groups)
            tmp_sel_features = list(estimator.get_feature_names_out())
            # fold_selected_features.extend(tmp_sel_features)
            # stabl_features.loc[f'Fold_{k}', "min FDP+"] = estimator.min_fdr_
            # stabl_features.loc[f'Fold_{k}', "Threshold"] = estimator.fdr_min_threshold_
            tmp_sel_idx = [data.columns.get_loc(f) for f in tmp_sel_features]
            fold_selected[tmp_sel_idx] = True
            stabl_features_list.append({
            "Threshold": estimator.fdr_min_threshold_,
            "min FDP+": estimator.min_fdr_
            })

            X_train = X_train[tmp_sel_features]
            X_test = X_test[tmp_sel_features]

            if len(tmp_sel_features) > 0:
                # Standardization
                X_train = fromPreprocessing(X_train,std_pipe)
                X_test = fromPreprocessingRep(X_test,std_pipe)
                # __Final Models__
                if task_type == "binary":
                    pred = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()
                    pred_xgboost = clone(xgboost_class).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()
                    pred_rf = clone(randomforest_class).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()
                elif task_type == "regression":
                    pred = clone(linreg).fit(X_train, y_train).predict(X_test)
                    pred_xgboost = clone(xgboost).fit(X_train, y_train).predict(X_test)
                    pred_rf = clone(randomforest).fit(X_train, y_train).predict(X_test)
                else:
                    raise ValueError("task_type not recognized.")

                fold_predictions[test] = pred
                fold_predictions_xgboost[test] = pred_xgboost
                fold_predictions_rf[test] = pred_rf

            else:
                if task_type == "binary":
                    # predictions.loc[test_idx, f'Fold_{k}'] = [0.5] * len(test_idx)
                    fold_predictions[test] = 0.5
                    fold_predictions_xgboost[test] = 0.5
                    fold_predictions_rf[test] = 0.5
                elif task_type == "regression":
                    # predictions.loc[test_idx, f'Fold_{k}'] = [np.mean(y_train)] * len(test_idx)
                    mean_y = np.mean(y_train)
                    fold_predictions[test] = mean_y
                    fold_predictions_xgboost[test] = mean_y
                    fold_predictions_rf[test] = mean_y
                else:
                    raise ValueError("task_type not recognized.")

        # selected_features.loc[f'Fold_{k}', fold_selected_features] = True
        selected_features_list.append(fold_selected)
        predictions_list.append(fold_predictions)
        
        if stablFlag:
            predictions_list_xgboost.append(fold_predictions_xgboost)
            predictions_list_rf.append(fold_predictions_rf)
        
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        k += 1

    predictions = pd.DataFrame(
        data=np.column_stack(predictions_list),
        index=y.index,
        columns=foldIdx
        )
    selected_features = pd.DataFrame(
        data=np.vstack(selected_features_list),
        columns=data.columns,
        index=foldIdx
        )

    if stablFlag:
        stabl_features = pd.DataFrame(
            stabl_features_list, 
            columns=["Threshold", "min FDP+"],
            index=foldIdx)
        
        # Create DataFrames for all prediction variants
        predictions_xgboost = pd.DataFrame(
            data=np.column_stack(predictions_list_xgboost),
            index=y.index,
            columns=foldIdx
        )
        predictions_rf = pd.DataFrame(
            data=np.column_stack(predictions_list_rf),
            index=y.index,
            columns=foldIdx
        )
        
        return 1, predictions, selected_features, stabl_features, predictions_xgboost, predictions_rf
    else:
        if ef:
            return 2,predictions,selected_features,best_params
        insamplePredictions = pd.DataFrame(
            data=np.column_stack(insamplePredictions_list),
            index=y.index,
            columns=foldIdx
        )
        return 0,predictions,selected_features,best_params, insamplePredictions

def save_single_omic_results(y,results,savePath,taskType):
    match results[0]:
        case 0:
            preds,selectedFeats,bestParams,insamplePredictions = results[1:]
            stablFeats = None
            preds_xgboost = None
            preds_rf = None
        case 1:
            if len(results) == 6:
                preds,selectedFeats,stablFeats,preds_xgboost,preds_rf = results[1:]
            else:  # Old format
                preds,selectedFeats,stablFeats = results[1:]
                preds_xgboost = None
                preds_rf = None
            bestParams = None
            insamplePredictions = None
        case 2:
            preds,selectedFeats,bestParams = results[1:]
            stablFeats = None
            insamplePredictions = None
            preds_xgboost = None
            preds_rf = None
    preds.to_csv(Path(savePath,"cvPreds.csv"))
    selectedFeats.astype(int).to_csv(Path(savePath,"selectedFeats.csv"))
    if bestParams is not None:
        pd.DataFrame(bestParams).to_csv(Path(savePath,"bestParams.csv"))
    if insamplePredictions is not None:
        insamplePredictions.to_csv(Path(savePath,"insamplePreds.csv"))
    if stablFeats is not None:
        stablFeats.to_csv(Path(savePath,"stablFeats.csv"))
        
        if preds_xgboost is not None:
            preds_xgboost.to_csv(Path(savePath,"cvPreds_xgboost.csv"))
            scores_xgboost = simpleScores(preds_xgboost,y,selectedFeats,taskType)
            scores_xgboost.to_csv(Path(savePath,"cvScores_xgboost.csv"))
            if taskType == "binary":
                plot_roc(y,preds_xgboost.median(axis=1),show_fig=False,path=Path(savePath,"ROC_xgboost.png"),export_file=True)  
                boxplot_binary_predictions(y,preds_xgboost.median(axis=1),show_fig=False,path=Path(savePath,"predBoxplot_xgboost.png"),export_file=True)
            if taskType == "regression":
                scatterplot_regression_predictions(y,preds_xgboost.median(axis=1),show_fig=False,path=Path(savePath,"scatterplot_xgboost.png"),export_file=True)

        if preds_rf is not None:
            preds_rf.to_csv(Path(savePath,"cvPreds_rf.csv"))
            scores_rf = simpleScores(preds_rf,y,selectedFeats,taskType)
            scores_rf.to_csv(Path(savePath,"cvScores_rf.csv"))
            if taskType == "binary":
                plot_roc(y,preds_rf.median(axis=1),show_fig=False,path=Path(savePath,"ROC_rf.png"),export_file=True)  
                boxplot_binary_predictions(y,preds_rf.median(axis=1),show_fig=False,path=Path(savePath,"predBoxplot_rf.png"),export_file=True)
            if taskType == "regression":
                scatterplot_regression_predictions(y,preds_rf.median(axis=1),show_fig=False,path=Path(savePath,"scatterplot_rf.png"),export_file=True)

    featCount = selectedFeats.sum(axis=0).T.sort_values(ascending=False)
    featCount.to_csv(Path(savePath,"featCount.csv"))

    scores = simpleScores(results[1],y,results[2],taskType)
    scores.to_csv(Path(savePath,"cvScores.csv"))
    if taskType == "binary":
        plot_roc(y,results[1].median(axis=1),show_fig=False,path=Path(savePath,"ROC.png"),export_file=True)  
        boxplot_binary_predictions(y,results[1].median(axis=1),show_fig=False,path=Path(savePath,"predBoxplot.png"),export_file=True)
    elif taskType == "regression":
        scatterplot_regression_predictions(y,results[1].median(axis=1),show_fig=False,path=Path(savePath,"scatterplot.png"),export_file=True)

        
def late_fusion_combination_normal(
        y,
        oosPredictions,
        isPredictions
    ):
    """
    y : pd.Series
        pandas Series containing the outcomes for the underlying datasets.
    oosPredictions : list of pd.DataFrames
        for each omic, the out-of-sample predictions over each of the folds of the crossvalidation
    isPredictions : list of pd.DataFramesr
        for each omic, the in-sample predictions over each of the folds of the crossvalidation
    """
    folds= isPredictions[0].columns
    predictions = pd.DataFrame(index=y.index, columns=folds,dtype=float)
    trainingPredictionsPerFold = {fold: [df[fold] for df in isPredictions] for fold in folds}
    validationPredictionsPerFold = {fold: [pred[fold] for pred in oosPredictions] for fold in folds}
    for fold in folds:
        foldTrainData = pd.concat(trainingPredictionsPerFold[fold], axis=1).dropna(how="all", axis=0)
        foldValidData = pd.concat(validationPredictionsPerFold[fold], axis=1).dropna(how="all", axis=0)
        foldTrainData["intercept"] = 1
        foldValidData["intercept"] = 1
        foldTrainY = y.loc[foldTrainData.index].to_numpy().flatten()
        beta,_ = nnls(foldTrainData.to_numpy(),foldTrainY)
        prediction = foldValidData.to_numpy() @ beta
        predictions.loc[foldValidData.index,fold] = prediction
    return predictions


def late_fusion_combination_stabl(
        data,
        y,
        selected_features,
        splits,
        task_type
    ):
    """
    data : pd.DataFrame
        pandas DataFrame containing the original data
    selected_features : list of lists
        for each omic, the list of selected features over each of the folds of the crosvalidation
    """
    predictions = pd.DataFrame(index=y.index, columns=selected_features.index,dtype=float)

    for k in range(len(splits)):
        trainIdx, testIdx = splits[k]
        fold_selected_features = np.argwhere(selected_features.iloc[k,:]).flatten()
        X_train = data.iloc[trainIdx,fold_selected_features]
        X_test = data.iloc[testIdx,fold_selected_features]
        y_train = y.iloc[trainIdx]

        if len(fold_selected_features) > 0:
            # Standardization
            X_train = fromPreprocessing(X_train,std_pipe)
            X_test = fromPreprocessingRep(X_test,std_pipe)
            # __Final Models__
            if task_type == "binary":
                pred = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()
            elif task_type == "regression":
                pred = clone(linreg).fit(X_train, y_train).predict(X_test)
            else:
                raise ValueError("task_type not recognized.")

            predictions.iloc[testIdx, k] = pred

        else:
            if task_type == "binary":
                predictions.iloc[testIdx, k] = [0.5] * len(testIdx)
            elif task_type == "regression":
                predictions.iloc[testIdx, k] = [np.mean(y_train)] * len(testIdx)

            else:
                raise ValueError("task_type not recognized.")
    
    return predictions




def simpleScores(
        predictions,
        y,
        features,
        task_type
):
    predictions = predictions.median(axis=1)

    if task_type == "binary":
        scores_columns = ["ROC AUC", "Average Precision", "N features", "CVS"]
    elif task_type == "regression":
        scores_columns = ["R2", "RMSE", "MAE", "N features", "CVS"]
    
    table_of_scores = []

    for metric in scores_columns:
        if metric == "ROC AUC":
            model_roc = roc_auc_score(y, predictions)
            model_roc_CI = compute_CI(y, predictions, scoring="roc_auc")
            cell_value = [f"{model_roc:.3f}",f"{model_roc_CI[0]:.3f}", f"{model_roc_CI[1]:.3f}"]

        elif metric == "Average Precision":
            model_ap = average_precision_score(y, predictions)
            model_ap_CI = compute_CI(y, predictions, scoring="average_precision")
            cell_value = [f"{model_ap:.3f}", f"{model_ap_CI[0]:.3f}", f"{model_ap_CI[1]:.3f}"]

        elif metric == "N features":
            sel_features = np.sum(features, axis=1)
            median_features = np.median(sel_features)
            iqr_features = np.quantile(sel_features, [.25, .75])
            cell_value = [f"{median_features:.3f}", f"{iqr_features[0]:.3f}", f"{iqr_features[1]:.3f}"]

        elif metric == "CVS":
            jaccard_mat = jaccard_matrix(features, remove_diag=False)
            jaccard_val = jaccard_mat[np.triu_indices_from(jaccard_mat, k=1)]
            jaccard_median = np.median(jaccard_val)
            jaccard_iqr = np.quantile(jaccard_val, [.25, .75])
            cell_value = [f"{jaccard_median:.3f}",f"{jaccard_iqr[0]:.3f}", f"{jaccard_iqr[1]:.3f}"]

        elif metric == "R2":
            model_r2 = r2_score(y, predictions)
            model_r2_CI = compute_CI(y, predictions, scoring="r2")
            cell_value = [f"{model_r2:.3f}",f"{model_r2_CI[0]:.3f}", f"{model_r2_CI[1]:.3f}"]

        elif metric == "RMSE":
            model_rmse = np.sqrt(mean_squared_error(y, predictions))
            model_rmse_CI = compute_CI(y, predictions, scoring="rmse")
            cell_value = [f"{model_rmse:.3f}", f"{model_rmse_CI[0]:.3f}", f"{model_rmse_CI[1]:.3f}"]

        elif metric == "MAE":
            model_mae = mean_absolute_error(y, predictions)
            model_mae_CI = compute_CI(y, predictions, scoring="mae")
            cell_value = [f"{model_mae:.3f}", f"{model_mae_CI[0]:.3f}", f"{model_mae_CI[1]:.3f}"]

        table_of_scores.extend(cell_value)

    table_of_scores = pd.DataFrame(data=table_of_scores, index=[ a + b for a in scores_columns for b in ["", " LB", " UB"]])
    return table_of_scores