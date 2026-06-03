import warnings
# Suppress XGBoost glibc FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*glibc.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")

from .EMS import read_json,unroll_parameters,write_json
from .single_omic import simpleScores,late_fusion_combination_normal,late_fusion_combination_stabl
import os
import numpy as np
import pandas as pd
from pathlib import Path  
from .visualization import boxplot_binary_predictions, plot_roc
from string import Template

defaultScriptTemplate = Template("""#!/usr/bin/bash
#SBATCH --job-name=${name}_${variant}
#SBATCH --error=./logs/${name}_${variant}_%a.err
#SBATCH --output=./logs/${name}_${variant}_%a.out
#SBATCH --array=0-${rep}
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH -c ${cpu}
#SBATCH --mem=${mem}GB

ml python/3.12.1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

export LOKY_MAX_CPU_COUNT=$$SLURM_CPUS_PER_TASK

time python3 ./sendOut.py 0 $${SLURM_ARRAY_TASK_ID} ${variant}
""")

endScriptTemplate = Template("""#!/usr/bin/bash
#SBATCH --job-name=${name}_e
#SBATCH --error=./logs/${name}_e.err
#SBATCH --output=./logs/${name}_e.out
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=8GB

ml python/3.12.1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

export LOKY_MAX_CPU_COUNT=$$SLURM_CPUS_PER_TASK

time python3 ./sendOut.py 1
""")

def parse_params(paramsFile: str, highMem: bool = False) -> None:
    params = read_json(paramsFile)
    
    # Validate required structure
    required_keys = ["Experiment_Name", "datasets", "models", "general", "preprocessing", "stabl_general"]
    for key in required_keys:
        if key not in params:
            raise KeyError(f"Missing required key '{key}' in params.json")
    
    # Validate general section
    general_required = ["taskType", "varType", "innerCVvals", "max_iter", "useScoring", 
                       "useRandomSeed", "setSeedValue", "n_jobs", "n_jobs_nonstabl", 
                       "memLowGB", "memHighGB", "cpusLow", "cpusHigh"]
    for key in general_required:
        if key not in params["general"]:
            raise KeyError(f"Missing required key '{key}' in params['general']")
    
    # Validate preprocessing section
    preprocessing_required = ["varValues", "lifThresh"]
    for key in preprocessing_required:
        if key not in params["preprocessing"]:
            raise KeyError(f"Missing required key '{key}' in params['preprocessing']")
    
    # Validate stabl_general section
    stabl_required = ["n_bootstraps", "replace", "artificialTypes", "artificialProportions", 
                     "sampleFractions", "fdrThreshParams"]
    for key in stabl_required:
        if key not in params["stabl_general"]:
            raise KeyError(f"Missing required key '{key}' in params['stabl_general']")
    
    # Validate that at least one model is enabled
    enabled_models = [k for k, v in params["models"].items() if v]
    if not enabled_models:
        raise ValueError("At least one model must be enabled in params['models']")
    
    # Validate that datasets list is not empty
    if not params["datasets"]:
        raise ValueError("datasets list cannot be empty")
    
    paramList = unroll_parameters(params)
    os.makedirs("./temp/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./results/", exist_ok=True)

    lowCount = 0
    highCount = 0
    for param in paramList:
        a, b = param["shorthand"].split("_")
        os.makedirs(f"./results/{b}/{a}", exist_ok=True)
        write_json(param, f"./results/{b}/{a}/params.json")
        highCount += (b == "h")
        lowCount += (b == "l")

    if lowCount != 0:
        os.makedirs("./results/l/", exist_ok=True)
        script = defaultScriptTemplate.substitute(
            name=params["Experiment_Name"],
            variant="l",
            rep=str(lowCount - 1),
            cpu=params["general"]["cpusLow"],
            mem=params["general"]["memLowGB"]
        )
        with open('./temp/arrayLow.sh', 'w') as file:
            file.write(script)

    if highCount != 0:
        os.makedirs("./results/h/", exist_ok=True)
        script = defaultScriptTemplate.substitute(
            name=params["Experiment_Name"],
            variant="h",
            rep=str(highCount - 1),
            cpu=params["general"]["cpusHigh"],
            mem=params["general"]["memHighGB"]
        )
        with open('./temp/arrayHigh.sh', 'w') as file:
            file.write(script)

    end_script = endScriptTemplate.substitute(name=params["Experiment_Name"])
    with open('./temp/end.sh', 'w') as file:
        file.write(end_script)
    
def save_late_fusion_results(pathLF, p, featCount, lfScores, lfPreds, plot_results, taskType, y):
    os.makedirs(pathLF, exist_ok=True)
    featCount.to_csv(pathLF / "featCount.csv")
    lfScores.to_csv(pathLF / "cvScores.csv")
    lfPreds.to_csv(pathLF / "cvPreds.csv")
    write_json(p, pathLF / "params.json")
    if plot_results and taskType == "binary":
        plot_roc(y, lfPreds.median(axis=1), show_fig=False, path=pathLF / "ROC.png", export_file=True)
        boxplot_binary_predictions(y, lfPreds.median(axis=1), show_fig=False, path=pathLF / "predBoxplot.png", export_file=True)


def run_end(paramsFile: str,
            data: pd.DataFrame,
            y: pd.Series,
            taskType: str,
            plot_results: bool = True,
            verbose: bool = False
            ) -> None:
    """
    Processes experiment results, performs late fusion, computes scores, and saves outputs.
    """
    params = read_json(paramsFile)
    results_dir = Path("./results/")
    if not results_dir.exists():
        if verbose:
            print("Results directory does not exist.")
        return

    # Create new organized results structure
    organized_results_dir = Path("./results_organized/")
    organized_results_dir.mkdir(exist_ok=True)
    


    intensities = [d for d in os.listdir(results_dir) if (results_dir / d).is_dir()]
    ef = "EarlyFusion" in params.get("datasets", [])
    n = len(params.get("datasets", []))
    m = n - ef
    lf = m > 1
    
    # Separate score collections for different result types
    single_omic_scores = []
    early_fusion_scores = []
    late_fusion_scores = []

    def create_model_identifier(param, exp_id, intensity, dataset_type="single", variant=None):
        """Create a clear, parsable model identifier"""
        model_name = param['model']
        task_type = param.get('taskType', 'binary')
        
        if variant:
            identifier = f"{model_name}_{variant}_{dataset_type}_{exp_id}_{intensity}"
        else:
            identifier = f"{model_name}_{dataset_type}_{exp_id}_{intensity}"
        
        return identifier

    def process_late_fusion_group(lfPreds, selectedFeats, grp, pathR, verbose, variant_suffix=None):
        try:
            lfScores = simpleScores(lfPreds, y, selectedFeats, taskType)
            featCount = selectedFeats.sum(axis=0).T.sort_values(ascending=False)
            
            if variant_suffix:
                pathLF = pathR / f"lf_{str(grp[0])}_{variant_suffix}"
            else:
                pathLF = pathR / f"lf_{str(grp[0])}"
                
            p = read_json(pathR / str(grp[0]) / "params.json")
            p["dataset"] = "LateFusion"
            if variant_suffix:
                p["variant"] = variant_suffix
            save_late_fusion_results(pathLF, p, featCount, lfScores, lfPreds, plot_results, taskType, y)
            
            model_id = create_model_identifier(p, str(grp[0]), pathR.name, "late_fusion", variant_suffix)
            lfScores.columns = [model_id]
            late_fusion_scores.append(lfScores)
        except Exception as e:
            if verbose:
                print(f"Error in fusion group {grp}: {e}")

    for intensity in intensities:
        pathR = results_dir / intensity
        exps = []
        existingParams = []
        if verbose:
            print(f"Processing intensity: {intensity}")
        
        for exp in os.listdir(pathR):
            exp_path = pathR / exp
            cv_scores_path = exp_path / "cvScores.csv"
            params_path = exp_path / "params.json"
            
            if cv_scores_path.exists() and params_path.exists():
                try:
                    param = read_json(params_path)
                    existingParams.append(param)
                    exps.append(int(exp))
                    
                    dataset_name = param.get('dataset', '')
                    is_early_fusion = dataset_name == "EarlyFusion"
                    
                    model_id = create_model_identifier(param, exp, intensity, 
                                                    "early_fusion" if is_early_fusion else "single_omic")
                    sc = pd.read_csv(cv_scores_path, index_col=0, names=[model_id], header=0)
                    
                    if is_early_fusion:
                        early_fusion_scores.append(sc)
                    else:
                        single_omic_scores.append(sc)
                    
                    if "stabl" in param.get("model", ""):
                        variants = ["xgboost", "rf", "linear"]  
                        
                        for variant in variants:
                            if variant == "linear":
                                continue
                                
                            variant_scores_path = exp_path / f"cvScores_{variant}.csv"
                            if variant_scores_path.exists():
                                try:
                                    variant_model_id = create_model_identifier(param, exp, intensity, 
                                                                             "early_fusion" if is_early_fusion else "single_omic", 
                                                                             variant)
                                    sc_variant = pd.read_csv(variant_scores_path, index_col=0, 
                                                           names=[variant_model_id], header=0)
                                    
                                    if is_early_fusion:
                                        early_fusion_scores.append(sc_variant)
                                    else:
                                        single_omic_scores.append(sc_variant)
                                        
                                except Exception as e:
                                    if verbose:
                                        print(f"Failed to read {variant_scores_path}: {e}")
                except Exception as e:
                    if verbose:
                        print(f"Failed to read {cv_scores_path}: {e}")

        if not existingParams:
            if verbose:
                print(f"No valid experiments found for intensity {intensity}.")
            continue

        if lf:
            lfGroupTags = np.array([e.get("lfTag", None) for e in existingParams])
            unique_tags = np.unique(lfGroupTags[lfGroupTags != None])
            lfGroups = [np.argwhere(lfGroupTags == i).flatten() for i in unique_tags]
            lfGroupsSTABL = [
                [existingParams[ee]["shorthand"].split("_")[0] for ee in e
                    if existingParams[ee].get("dataset") != "EarlyFusion" and "stabl" in existingParams[ee].get("model", "")]
                for e in lfGroups
            ]
            lfGroupsNonSTABL = [
                [existingParams[ee]["shorthand"].split("_")[0] for ee in e
                    if existingParams[ee].get("dataset") != "EarlyFusion" and "stabl" not in existingParams[ee].get("model", "")]
                for e in lfGroups
            ]
            lfGroupsSTABL = [np.sort(np.array(e).astype(int)) for e in lfGroupsSTABL if len(e) > 1]
            lfGroupsNonSTABL = [np.sort(np.array(e).astype(int)) for e in lfGroupsNonSTABL if len(e) > 1]

            for grp in lfGroupsSTABL:
                if verbose:
                    print(f"Late fusion STABL group: {grp}")
                selectedFeats = pd.concat([pd.read_csv(pathR / str(e) / "selectedFeats.csv", index_col=0).astype(bool) for e in grp], axis=1)
                
                # Process each prediction variant for STABL models (including random forest)
                prediction_variants = {
                    "linear": "cvPreds.csv",
                    "xgboost": "cvPreds_xgboost.csv", 
                    "rf": "cvPreds_rf.csv"
                }
                
                for variant_name, pred_file in prediction_variants.items():
                    # Check if the prediction file exists for the first experiment in the group
                    pred_path = pathR / str(grp[0]) / pred_file
                    if pred_path.exists():
                        if verbose:
                            print(f"  Processing {variant_name} variant...")
                        
                        prd = pd.read_csv(pred_path, index_col=0)
                        prd = prd.loc[data.index]
                        splits = [[np.argwhere(prd[col].isna()).flatten(), np.argwhere(~prd[col].isna()).flatten()] for col in prd.columns]
                        lfPreds = late_fusion_combination_stabl(data, y, selectedFeats, splits, taskType)
                        
                        # Create a modified group name to include the variant
                        grp_with_variant = grp.copy()
                        process_late_fusion_group(lfPreds, selectedFeats, grp_with_variant, pathR, verbose, variant_suffix=variant_name)
                    else:
                        if verbose:
                            print(f"  Skipping {variant_name} variant - file not found: {pred_path}")

            for grp in lfGroupsNonSTABL:
                if verbose:
                    print(f"Late fusion Non-STABL group: {grp}")
                selectedFeats = pd.concat([pd.read_csv(pathR / str(e) / "selectedFeats.csv", index_col=0).astype(bool) for e in grp], axis=1)
                isPreds = [pd.read_csv(pathR / str(e) / "insamplePreds.csv", index_col=0) for e in grp]
                oosPreds = [pd.read_csv(pathR / str(e) / "cvPreds.csv", index_col=0) for e in grp]
                lfPreds = late_fusion_combination_normal(y, oosPreds, isPreds)
                process_late_fusion_group(lfPreds, selectedFeats, grp, pathR, verbose)


    # Save organized results
    def save_organized_scores(scores_list, output_dir, filename_prefix):
        """Save scores to organized directory structure"""
        if scores_list:
            combined_scores = pd.concat(scores_list, axis=1).T.astype(float)
            # Sort by the first metric (usually the primary metric)
            if len(combined_scores.columns) > 0:
                combined_scores = combined_scores.sort_values(by=combined_scores.columns[0], ascending=False)
            combined_scores.to_csv(output_dir / f"{filename_prefix}_cvScores.csv")
            return combined_scores
        return None
    
    # Save single-omic results
    single_omic_results = save_organized_scores(single_omic_scores, organized_results_dir, "single_omic")
    if single_omic_results is not None and verbose:
        print(f"Saved {len(single_omic_results)} single-omic results to {organized_results_dir}/single_omic_cvScores.csv")
    
    # Save early fusion results
    early_fusion_results = save_organized_scores(early_fusion_scores, organized_results_dir, "early_fusion")
    if early_fusion_results is not None and verbose:
        print(f"Saved {len(early_fusion_results)} early fusion results to {organized_results_dir}/early_fusion_cvScores.csv")
    
    # Save late fusion results
    late_fusion_results = save_organized_scores(late_fusion_scores, organized_results_dir, "late_fusion")
    if late_fusion_results is not None and verbose:
        print(f"Saved {len(late_fusion_results)} late fusion results to {organized_results_dir}/late_fusion_cvScores.csv")
    
    # Create a summary file with all results combined
    all_scores = []
    if single_omic_results is not None:
        all_scores.append(single_omic_results)
    if early_fusion_results is not None:
        all_scores.append(early_fusion_results)
    if late_fusion_results is not None:
        all_scores.append(late_fusion_results)
    
    combined_all = pd.concat(all_scores, axis=0)
    combined_all = combined_all.sort_values(by=combined_all.columns[0], ascending=False)
    combined_all.to_csv(organized_results_dir / "all_results_cvScores.csv")
    combined_all.to_csv("./results/cvScores.csv")
    if verbose:
        print(f"Saved combined results summary to {organized_results_dir}/all_results_cvScores.csv")
