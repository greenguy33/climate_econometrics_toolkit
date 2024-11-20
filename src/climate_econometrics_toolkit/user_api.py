from climate_econometrics_toolkit import interface_api as api
from climate_econometrics_toolkit.ClimateEconometricsModel import ClimateEconometricsModel
import climate_econometrics_toolkit.utils as utils
from climate_econometrics_toolkit import regression as regression
from climate_econometrics_toolkit import prediction as predict

import pandas as pd
import copy

model = ClimateEconometricsModel()
model_list = {}

def model_checks():
    checks = {
        "No dataset loaded." : model.dataset is not None,
        "No target variable set." : not pd.isnull(model.target_var),
		"No model covariates found." : model.covariates != [],
		"No time-based column set." : model.time_column is not None,
		"No panel column set." : model.panel_column is not None
    }
    for key, check in checks.items():
        if not check:
            print(f"{key} Please update your model.")
            return False
    return True

def evaluate_model():
    if model_checks():
        model_id, _, return_string = api.run_model_analysis(copy.deepcopy(model.dataset), model, save_to_cache=False)
        model.model_id = model_id
        if return_string != "": print(return_string)
        if model_id != None:
            print(f"Model ID: {model_id}")
            model_list[str(model_id)] = copy.deepcopy(model)
            return str(model_id)

def get_best_model(metric="r2"):
    if metric not in utils.supported_metrics:
        print(f"Metric must be one of {utils.supported_metrics}")
    else:
        if metric in ["r2","out_sample_mse","rmse"]:
            sorted_models = sorted(model_list.items(), key=lambda x : getattr(x[1], metric))
        elif metric == "out_sample_mse_reduction":
            sorted_models = sorted(model_list.items(), key=lambda x : getattr(x[1], metric), reverse=True)
        elif metric == "out_sample_pred_int_cov":
            sorted_models = sorted(model_list.items(), key=lambda x : abs(getattr(x[1], "out_sample_pred_int_cov")-.95))
        for (key, model) in sorted_models:
            print("Model ID", key)
            print(model.print())
            return key
        
def get_all_model_ids():
    return list(model_list.keys())

def get_model_by_id(model_id):
    return model_list[model_id]

def run_bayesian_regression(model):
    regression.run_bayesian_regression(model)

def run_block_bootstrap(model):
    regression.run_block_bootstrap(model)

def predict_from_gcms(model, gcms_to_use):
    predict.predict_from_gcms(model, gcms_to_use)

def load_dataset_from_file(datafile):
    model.data_file = datafile.split("/")[-1]
    model.dataset = pd.read_csv(datafile)

def view_current_model():
    model.print()

def basic_existence_check(node):
    if model.dataset is None:
        print("Please load a dataset before setting variables.")
        return False
    elif node not in model.dataset:
        print(f"Element {node} not found in data")
        return False
    return True

def set_target_variable(node, existence_check=True):
    if not existence_check or basic_existence_check(node):
        model.target_var = node
        model.model_vars = model.covariates + [model.target_var]

def set_time_column(node):
    if basic_existence_check(node):
        model.time_column = node

def set_panel_column(node):
    if basic_existence_check(node):
        model.panel_column = node

def add_transformation(node, transform):
    if transform not in utils.supported_functions:
        print(f"{transform}() not a supported function.")
    elif node not in model.covariates and node != model.target_var:
        print(f"{node} not in covariates list and is not target variable.")
    elif node in model.covariates:
        # remove_covariate(node)
        add_covariate(f"{transform}({node})", existence_check=False)
    elif node == model.target_var:
        model.target_var = f"{transform}({node})"

def add_transformations(node, transform_list):
    all_transformations_valid = True
    for transform in transform_list:
        if transform not in utils.supported_functions:
            all_transformations_valid = False
            print(f"{transform}() not a supported function.")
    if all_transformations_valid:
        if node not in model.covariates and node != model.target_var:
            print(f"{node} not in covariates list and is not target variable.")
        elif node in model.covariates:
            for transform in transform_list:
                remove_covariate(node)
                add_covariate(f"{transform}({node})", existence_check=False)
                node = f"{transform}({node})"
        elif node == model.target_var:
            for transform in transform_list:
                node = f"{transform}({node})"
            set_target_variable(node, existence_check=False)

def add_covariates(nodes):
    if all(basic_existence_check(node) for node in nodes):
        for node in nodes:
            if node not in model.covariates:
                model.covariates.append(node)
        model.model_vars = model.covariates + [model.target_var]

def add_covariate(node, existence_check=True):
    if not existence_check or basic_existence_check(node):
        if node not in model.covariates:
            model.covariates.append(node)
        model.model_vars = model.covariates + [model.target_var]

def add_fixed_effects(nodes):
    if all(basic_existence_check(node) for node in nodes):
        for fe in nodes:
            if fe not in model.fixed_effects:
                model.fixed_effects.append(fe)

def add_fixed_effect(node):
    if basic_existence_check(node):
        if node not in model.fixed_effects:
            model.fixed_effects.append(node)

def add_time_trend(node, exp):
    if basic_existence_check(node):
        time_trend = node + " " + str(exp)
        if time_trend not in model.time_trends:
            model.time_trends.append(time_trend)

def remove_covariate(node):
    # TODO: cannot easily remove a transformed node
    model.covariates = [var for var in model.covariates if var != node]

def remove_fixed_effect(node):
    model.fixed_effects = [var for var in model.fixed_effects if var != node]

def remove_time_trend(node, exp):
    time_trend = node + " " + str(exp)
    model.time_trends = [var for var in model.time_trends if var != time_trend]
