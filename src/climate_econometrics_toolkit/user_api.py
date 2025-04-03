from climate_econometrics_toolkit import interface_api as api
from climate_econometrics_toolkit.ClimateEconometricsModel import ClimateEconometricsModel
import climate_econometrics_toolkit.utils as utils
from climate_econometrics_toolkit import regression as regression
from climate_econometrics_toolkit import prediction as predict
from climate_econometrics_toolkit import user_prediction_functions as user_predict
from climate_econometrics_toolkit import user_prediction_functions as user_predict
from climate_econometrics_toolkit import stat_tests as stat_tests

import pandas as pd
import os
import copy
import time

model = ClimateEconometricsModel()

cet_home = os.getenv("CETHOME")

# TODO: assert types for user input to each method

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

def evaluate_model(std_error_type="nonrobust"):
    # TODO: check to see if this model is already in cache, if so return that model rather than re-evaluating the same model
    if model_checks():
        _, _, return_string = api.run_model_analysis(copy.deepcopy(model.dataset), std_error_type, model, save_to_cache=False)
        if return_string != "": print(return_string)
        if model != None:
            model.save_model_to_cache()
            print(f"Model ID: {model.model_id}")
            return str(model.model_id)
        

def build_model_from_cache(model_id):
    if model.data_file == None:
        print("You must load a dataset before accessing the cache")
        return None
    else:
        return pd.read_pickle((f"{cet_home}/model_cache/{model.data_file}/{model_id}/model.pkl"))
        

def get_all_models_from_cache():
    if model.data_file == None:
        print("You must load a dataset before accessing the cache")
        return None
    else:
        model_list = []
        model_ids = os.listdir(f"{cet_home}/model_cache/{model.data_file}")
        for model_id in model_ids:
            model_list.append(build_model_from_cache(model_id))
        return model_list


def get_best_model(metric="r2"):
    model_list = get_all_models_from_cache()
    if metric not in utils.supported_metrics:
        print(f"Metric must be one of {utils.supported_metrics}")
    else:
        if metric in ["r2","out_sample_mse","rmse"]:
            sorted_models = sorted(model_list, key=lambda x : getattr(x, metric))
        elif metric == "out_sample_mse_reduction":
            sorted_models = sorted(model_list, key=lambda x : getattr(x, metric), reverse=True)
        elif metric == "out_sample_pred_int_cov":
            sorted_models = sorted(model_list, key=lambda x : abs(getattr(x, "out_sample_pred_int_cov")-.95))
        print("Model ID", sorted_models[0].model_id)
        print(sorted_models[0].print())
        return sorted_models[0]

def get_all_model_ids():
    if model.data_file == None:
        print("You must load a dataset before accessing the cache")
        return None
    else:
        return list(os.listdir(f"{cet_home}/model_cache/{model.data_file}"))

def get_model_by_id(model_id):
    return build_model_from_cache(model_id)

def load_dataset_from_file(datafile):
    # resets model when new dataset is loaded
    reset_model()
    model.data_file = datafile.split("/")[-1]
    model.dataset = pd.read_csv(datafile)

def set_dataset(dataframe, dataset_name):
    # resets model when new dataset is loaded
    reset_model()
    model.data_file = dataset_name
    model.dataset = dataframe

def view_current_model():
    model.print()

def basic_existence_check(var):
    if model.dataset is None:
        print("Please load a dataset before setting variables.")
        return False
    elif var not in model.dataset:
        print(f"Element {var} not found in data")
        return False
    return True

def set_target_variable(var, existence_check=True):
    if not existence_check or basic_existence_check(var):
        model.target_var = var
        model.model_vars = model.covariates + [model.target_var]

def set_time_column(var):
    if basic_existence_check(var):
        model.time_column = var

def set_panel_column(var):
    if basic_existence_check(var):
        model.panel_column = var

def add_transformation(var, transformations, keep_original_var=True):
    if not isinstance(transformations, list):
        transformations = [transformations]
    all_transformations_valid = True
    for transform in transformations:
        if transform not in utils.supported_functions:
            all_transformations_valid = False
            print(f"{transform}() not a supported function.")
    if all_transformations_valid:
        if var not in model.covariates and var != model.target_var:
            print(f"{var} not in covariates list and is not target variable.")
        elif var in model.covariates:
            for transform in transformations:
                if not keep_original_var:
                    remove_covariates(var)
                var = f"{transform}({var})"
            add_covariates(f"{var}", existence_check=False)
        elif var == model.target_var:
            for transform in transformations:
                var = f"{transform}({var})"
            set_target_variable(var, existence_check=False)

def add_covariates(vars, existence_check=True):
    if not isinstance(vars, list):
        vars = [vars]
    if not existence_check or all(basic_existence_check(var) for var in vars):
        for var in vars:
            if var not in model.covariates:
                model.covariates.append(var)
        model.model_vars = model.covariates + [model.target_var]

def add_fixed_effects(vars):
    if not isinstance(vars, list):
        vars = [vars]
    if all(basic_existence_check(var) for var in vars):
        for fe in vars:
            if fe not in model.fixed_effects:
                model.fixed_effects.append(fe)

def add_time_trend(var, exp):
    if basic_existence_check(var):
        time_trend = var + " " + str(exp)
        if time_trend not in model.time_trends:
            model.time_trends.append(time_trend)

def remove_covariates(vars):
    if not isinstance(vars, list):
        vars = [vars]
    for var_to_remove in vars:
        model.covariates = [var for var in model.covariates if var != var_to_remove]
        model.model_vars = [var for var in model.model_vars if var != var_to_remove]

def remove_time_trend(var, exp):
    time_trend = var + " " + str(exp)
    model.time_trends = [var for var in model.time_trends if var != time_trend]

def remove_transformation(var, transformations):
    if not isinstance(transformations, list):
        transformations = [transformations]
    transformed_var = copy.deepcopy(var)
    for transform in transformations:
        transformed_var = f"{transform}({transformed_var})"
    if model.target_var == transformed_var:
        set_target_variable(var)
    elif transformed_var in model.covariates:
        model.covariates = [var for var in model.covariates if var != transformed_var]
        model.model_vars = [var for var in model.model_vars if var != transformed_var]
    else:
        print(f"Transformed var f{transformed_var} not found")

def remove_fixed_effect(fe):
    model.fixed_effects = [var for var in model.fixed_effects if var != fe]

def add_random_effect(var, group):
    if model.random_effects != [var, group]:
        if model.random_effects != None:
            print("Only one random effect is supported. Please remove the previous random effect before adding another.")
        else:
            model.random_effects = [var, group]
            if var in model.covariates:
                remove_covariates(var)

def remove_random_effect(add_to_covariate_list=True):
    if model.random_effects is not None:
        if add_to_covariate_list:
            add_covariates(model.random_effects[0])
        model.random_effects = None

def run_spatial_lag_regression(reg_type, geometry_column=None):
    model_id = time.time()
    regression.run_spatial_regression(model, reg_type, model_id, geometry_column)

def run_spatial_error_regression(reg_type, geometry_column=None):
    model_id = time.time()
    regression.run_spatial_regression(model, reg_type, model_id, geometry_column)

def run_quantile_regression(q):
    model_id = time.time()
    if isinstance(q, list):
        for val in q:
            regression.run_quantile_regression(model, model_id, val)
    else:
        regression.run_quantile_regression(model, model_id, q)


def run_stationarity_check():
    assert model.dataset is not None, "Dataset must be set before running stationarity check."
    assert model.target_var is not None, "Target variable must be set before running stationarity check."
    assert model.target_var is not None, "Covariates must be set before running stationarity check."
    assert model.time_column is not None, "Time column must be set before running stationarity check."
    assert model.panel_column is not None, "Panel column must be set before running stationarity check."
    return stat_tests.panel_unit_root_tests(model)


def run_cointegration_check():
    assert model.dataset is not None, "Dataset must be set before running cointegration check."
    assert model.target_var is not None, "Target variable must be set before running cointegration check."
    assert model.target_var is not None, "Covariates must be set before running cointegration check."
    assert model.time_column is not None, "Time column must be set before running cointegration check."
    assert model.panel_column is not None, "Panel column must be set before running cointegration check."
    return stat_tests.cointegration_tests(model)


def run_cross_sectional_dependence_check():
    assert model.dataset is not None, "Dataset must be set before running cointegration check."
    assert model.target_var is not None, "Target variable must be set before running cointegration check."
    assert model.target_var is not None, "Covariates must be set before running cointegration check."
    assert model.time_column is not None, "Time column must be set before running cointegration check."
    assert model.panel_column is not None, "Panel column must be set before running cointegration check."
    return stat_tests.cross_sectional_dependence_tests(model)


def run_bayesian_regression(model, num_samples=1000):
    # TODO: check to see if bayesian inference already ran for this model
    if isinstance(model, str):
        model = get_model_by_id(model)
    regression.run_bayesian_regression(model, num_samples)

def run_block_bootstrap(model, std_error_type, num_samples=1000):
    # TODO: check to see if bootstrap already ran for this model
    if isinstance(model, str):
        model = get_model_by_id(model)
    assert model != None, "NoneType passed as model object"
    regression.run_block_bootstrap(model, std_error_type, num_samples)

def extract_raster_data(raster_file, shape_file, weight_file=None):
    return predict.extract_raster_data(raster_file, shape_file, weight_file)

def aggregate_raster_data(data, shape_file, climate_var_name, aggregation_func, geo_identifier, subperiods_per_time_unit, months_to_use=None):
    return predict.aggregate_raster_data(data, shape_file, climate_var_name, aggregation_func, geo_identifier, subperiods_per_time_unit, months_to_use)

def predict_out_of_sample(model, data, transform_data=False, var_map=None):
    if isinstance(model, str):
        model = get_model_by_id(model)
    return predict.predict_out_of_sample(model, copy.deepcopy(data), transform_data, var_map)

def call_user_prediction_function(function_name, args):
    func = getattr(user_predict, function_name)
    return func(*args)

# TODO: document below this line

def transform_data(data, model, include_target_var=True, demean=False):
    return utils.transform_data(copy.deepcopy(data), model, include_target_var, demean)

def reset_model():
    global model
    model = ClimateEconometricsModel()