import pandas as pd
import numpy as np
import os
import copy
import time

from importlib.resources import files
from functools import reduce
from metpy.calc import heat_index
from metpy.units import units
import progressbar
import itertools as it
import pyreadr

from climate_econometrics_toolkit import interface_api as api
from climate_econometrics_toolkit.ClimateEconometricsModel import ClimateEconometricsModel
import climate_econometrics_toolkit.utils as utils
from climate_econometrics_toolkit import regression as regression
from climate_econometrics_toolkit import prediction as predict
from climate_econometrics_toolkit import raster_extraction as extract
from climate_econometrics_toolkit import user_prediction_functions as user_predict
from climate_econometrics_toolkit import stat_tests as stat_tests
from climate_econometrics_toolkit import degree_days as dd

current_model = ClimateEconometricsModel()

cet_home = os.getenv("CETHOME")

# TODO: assert correct for user input to each method
# TODO: the transformations user interface is clunky and unintuitive, ideally should be re-worked

def run_specification_search(model=None, metric="out_sample_mse_reduction", cv_folds=10):
    if model is None:
        model = copy.deepcopy(current_model)
    utils.assert_with_log(metric in utils.supported_metrics, f"Supplied metric must be one of: {utils.supported_metrics}")
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    utils.model_checks(model)

    target_var = model.target_var.split("(")[-1].split(")")[0]
    covariates = list(set([var.split("(")[-1].split(")")[0] for var in model.covariates]))
    dataset = model.dataset
    panel_column = model.panel_column
    time_column = model.time_column
    levels = ["level","fd"]
    logged_target_var = [False,True]
    fe = [["ISO3"], ["year"], None, ["ISO3","year"]]
    transformations = [None, "sq", "cu"]
    permutation_list = list(it.product(*[levels, logged_target_var, fe, transformations]))

    model_list = []
    for permutation in progressbar.progressbar(permutation_list):
        model = ClimateEconometricsModel()
        model.dataset = dataset
        model.data_file = "ds1"
        model.panel_column = panel_column
        model.time_column = time_column
        model.target_var = target_var
        # copy is important here to avoid altering the base covariates list when applying transformations
        model.covariates = copy.deepcopy(covariates)
        if permutation[1]:
            model.target_var = f"ln({target_var})"
        if permutation[-1] == "sq":
            new_covariate_list = []
            for cov in model.covariates:
                new_covariate_list.append(f"sq({cov})")
            model.covariates.extend(new_covariate_list)
        elif permutation[-1] == "cu":
            new_covariate_list = []
            for cov in model.covariates:
                new_covariate_list.append(f"sq({cov})")
                new_covariate_list.append(f"cu({cov})")
            model.covariates.extend(new_covariate_list)
        if permutation[0] == "fd":
            model.covariates = [f"fd({var})" for var in model.covariates]
            model.target_var = f"fd({model.target_var})"
        if permutation[2] != None:
            add_fixed_effects(permutation[2])
        model.model_vars.append(model.target_var)
        for var in model.covariates:
            model.model_vars.append(var)
        evaluate_model_with_OLS(model, cv_folds=cv_folds)
        model_list.append(model)

    best_model = get_best_model(model_list, metric)
    utils.print_with_log(f"Best model for metric {metric}: {best_model.to_string()}", "info")
    return best_model


def compute_degree_days(years, countries, threshold, mode="above", panel_column_name="ISO3", time_column_name="year", crop=None, second_threshold=None, computation="country"):
	utils.print_with_log(f"Computing degree days with mode '{mode}', using panel column '{panel_column_name}' and time_column '{time_column_name}'", "info")
	if crop is None:
		utils.print_with_log("No growing season specified with 'crop' argument: degree days will be computed for the entire year.", "info")
	utils.assert_with_log(mode in ["above","below","between"], "Mode most be either 'above', 'below', or 'between' the supplied threshold(s).")
	if mode == "between":
		utils.assert_with_log(second_threshold is not None, "Second threshold argument must be supplied to use mode 'between'.")
		utils.assert_with_log(second_threshold > threshold, "Second threshold argument must be greater than threshold argument.")
	elif second_threshold is not None:
		utils.print_with_log(f"Argument '{second_threshold}' to 'second_threshold' parameter is ignored with mode {mode}.", "warning")
	col_name = f"deg_days_{mode}_{str(threshold)}"
	if mode == "between":
		col_name += f"_{str(second_threshold)}"
	if computation == "country":
		return dd.compute_country_degree_days(years, countries, threshold, mode, panel_column_name, time_column_name, crop, second_threshold, col_name)
	return dd.compute_gridded_degree_days(years, countries, threshold, mode, panel_column_name, time_column_name, crop, second_threshold, col_name)


def add_degree_days_to_dataframe(dataframe, threshold, panel_column = "ISO3", time_column = "year", mode = "above", crop=None, second_threshold=None, computation="country"):
    utils.assert_with_log(panel_column in dataframe, f"Specified panel column {panel_column} not in supplied dataframe.")
    utils.assert_with_log(time_column in dataframe, f"Specified time column {time_column} not in supplied dataframe.")
    degree_days_df = compute_degree_days(set(dataframe[time_column]), set(dataframe[panel_column]), threshold, mode, crop=crop, second_threshold=second_threshold, computation=computation)
    utils.assert_with_log(len(degree_days_df) > 0, f"No daily temperature data available for supplied columns {panel_column}/{time_column}.")
    merge_strategy = "outer"
    utils.print_with_log(f"Merging supplied dataframe with degree days dataframe using merge strategy '{merge_strategy}'", "info")
    return pd.merge(dataframe, degree_days_df, on=[panel_column,time_column], how=merge_strategy)


def integrate(dataframes, time_aggregation=None, keep_na=False, panel_column="ISO3", time_column="year"):
    df_mod = []
    for df in dataframes:
        df_mod.append(df[[col for col in df.columns if not col.startswith("Unnamed")]])
    all_geolocations = set.intersection(*[set(df[panel_column]) for df in df_mod])
    utils.assert_with_log(len(all_geolocations) > 0, f"No overlap found in column {panel_column} between datasets")
    all_times = set.intersection(*[set(df[time_column]) for df in df_mod])
    utils.assert_with_log(len(all_times) > 0, f"No overlap found in column {time_column} between datasets")
    merge_method = "inner" if not keep_na else "outer"
    integrated_df = reduce(lambda left,right: pd.merge(left,right,on=[panel_column,time_column], how=merge_method), df_mod)
    utils.print_with_log(f"Merging supplied dataframes using merge strategy '{merge_method}' and keep_na set to {keep_na}", "info")
    if time_aggregation is not None:
        utils.assert_with_log(isinstance(time_aggregation, int), "Argument to parameter 'time_aggregation' must be an integer.")
        utils.print_with_log(f"Applying time-based aggregation using function 'mean' and chunk size {time_aggregation}", "info")
        integrated_df["groupid"] = integrated_df.groupby(panel_column).cumcount()//time_aggregation
        grouped_df = integrated_df.groupby([panel_column,"groupid"]).mean()
        integrated_df = grouped_df.reset_index().drop("groupid",axis=1)
        integrated_df[time_column] = integrated_df.apply(lambda x: str(x[time_column]-time_aggregation/2).split(".")[0] + "-" + str(x[time_column]+time_aggregation/2).split(".")[0], axis=1)
    return integrated_df.reset_index(drop=True)


def convert_between_administrative_levels(data, from_code, to_code):
    utils.assert_with_log(from_code in ["admin1","admin2"], "Argument 'from_code' must be one of: admin1, admin2.")
    utils.assert_with_log(to_code in ["admin1","country"], "Argument 'to_code' must be one of: admin1, country.")
    utils.assert_with_log(from_code != to_code, "Arguments 'from_code' and 'to_code' must be different.")
    admin1_file = files("climate_econometrics_toolkit.preprocessed_data.admin_codes").joinpath(f'geonames_admin1.csv')
    admin2_file = files("climate_econometrics_toolkit.preprocessed_data.admin_codes").joinpath(f'geonames_admin2.csv')
    admin1_data = pd.read_csv(admin1_file)
    admin2_data = pd.read_csv(admin2_file)
    admin1_dict = dict(zip(admin1_data["admin1_name"], admin1_data["ISO3"]))
    admin1_alt_dict = dict(zip(admin1_data["admin1_name_alt"], admin1_data["ISO3"]))
    admin1_dict = dict(list(admin1_dict.items()) + list(admin1_alt_dict.items()))
    admin2_data["country.admin1Id"] = admin2_data["country.admin1Id.admin2Id"].str.rsplit(".", n=1, expand=True)[0]
    admin1id_to_admin1_dict = dict(zip(admin1_data["country.admin1Id"], admin1_data["admin1_name"]))
    admin2_dict = dict(zip(admin2_data["admin2_name"], admin2_data["country.admin1Id"]))
    admin2_alt_dict = dict(zip(admin2_data["admin2_name_alt"], admin2_data["country.admin1Id"]))
    admin2_dict = dict(list(admin2_dict.items()) + list(admin2_alt_dict.items()))
    utils.print_with_log(f"Converting supplied data from code system {from_code} to code system {to_code}", "info")
    if from_code == "admin1" and to_code == "country":
        return pd.Series(list(map(lambda x: admin1_dict[x] if x in admin1_dict else None, data)))
    if from_code == "admin2" and to_code == "admin1":
        return pd.Series(list(map(lambda x: admin1id_to_admin1_dict[admin2_dict[x]] if x in admin2_dict and admin2_dict[x] in admin1id_to_admin1_dict else None, data)))
    if from_code == "admin2" and to_code == "country":
        return pd.Series(list(map(lambda x: admin1_dict[admin1id_to_admin1_dict[admin2_dict[x]]] if x in admin2_dict and admin2_dict[x] in admin1id_to_admin1_dict and admin1id_to_admin1_dict[admin2_dict[x]] in admin1_dict else None, data)))


def load_ncep_ncar_data(weight="areaweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    if weight == "areaweighted":
        weight = "unweighted"
    file = files("climate_econometrics_toolkit.preprocessed_data.weather_data").joinpath(f'NCEP_reanalaysis_climate_data_1948_2024_{weight}.csv')
    return pd.read_csv(file)


def load_temperature_humidity_index_data(weight="areaweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    if weight == "areaweighted":
        weight = "unweighted"
    file = files("climate_econometrics_toolkit.preprocessed_data.temperature_humidity_index").joinpath(f'temperature_humidity_index_{weight}_1948_2024.csv')
    return pd.read_csv(file)


def load_dewpoint_temperature_data(weight="areaweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    if weight == "areaweighted":
        weight = "unweighted"
    file = files("climate_econometrics_toolkit.preprocessed_data.dewpoint_temperature").joinpath(f'dewpoint_temperature_{weight}_1948_2024.csv')
    return pd.read_csv(file)


def load_ndvi_data(weight="areaweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    if weight == "areaweighted":
        weight = "unweighted"
    file = files("climate_econometrics_toolkit.preprocessed_data.NDVI").joinpath(f'pku_ndvi_data_aggregated_{weight}.csv')
    return pd.read_csv(file)


def load_emdat_data():
    file = files("climate_econometrics_toolkit.preprocessed_data").joinpath('EMDAT_natural_disasters_1960_2024.csv')
    return pd.read_csv(file)


def load_faostat_data():
    file = files("climate_econometrics_toolkit.preprocessed_data").joinpath('FAOSTAT_production_indices_1961_2023.csv')
    return pd.read_csv(file)


def load_usda_fda_data():
    file = files("climate_econometrics_toolkit.preprocessed_data").joinpath('USDA_FDA_global_TFP_1961_2021.csv')
    return pd.read_csv(file).reset_index(drop=True)


def load_worldbank_gdp_data():
    file = files("climate_econometrics_toolkit.preprocessed_data").joinpath('worldbank_global_GDP_1961_2023.csv')
    return pd.read_csv(file)


def load_spei_data(weight="areaweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    if weight == "areaweighted":
        weight = "unweighted"
    file = files("climate_econometrics_toolkit.preprocessed_data.SPEI").joinpath(f'spei_{weight}.csv')
    return pd.read_csv(file)


def load_cdc_unified_max_temperature_data(weight="areaweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    if weight == "areaweighted":
        weight = "unweighted"
    file = files(f"climate_econometrics_toolkit.preprocessed_data.daily_temp_max.{weight}").joinpath(f'temp.daily_max.daily.bycountry.{weight}.csv')
    return pd.read_csv(file)


def load_cdc_unified_min_temperature_data(weight="areaweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    if weight == "areaweighted":
        weight = "unweighted"
    file = files(f"climate_econometrics_toolkit.preprocessed_data.daily_temp_min.{weight}").joinpath(f'temp.daily_min.daily.bycountry.{weight}.csv')
    return pd.read_csv(file)


def get_temperature_humidity_index(temp_data, relative_humidity_data):
    # requires temp data in celsius and relative humidity data (percentage) between 0 and 100
    utils.print_with_log("Generating temperature/humidity index using supplied data", "info")
    return heat_index(
        temp_data * units.degC, 
        relative_humidity_data * units.percent,
        mask_undefined=False
    ).magnitude


def evaluate_model_with_OLS(model=None, std_error_type="nonrobust", cv_folds=10):
    if model is None:
        model = current_model
    utils.initial_checks()
    # TODO: check to see if this model is already in cache, if so return that model rather than re-evaluating the same model
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    _, _, return_string = api.run_model_analysis(copy.deepcopy(model.dataset), std_error_type, model, save_model_to_cache=True, save_result_to_file=False, cv_folds=cv_folds)
    if return_string != "": utils.print_with_log(return_string, "error")
    if model != None:
        utils.print_with_log(f"Model assigned ID: {model.model_id}", "info")
        return str(model.model_id)
        

def build_model_from_cache(model_id):
    utils.initial_checks()
    utils.assert_with_log(current_model.data_file is not None, "Attempted to access cache with no dataset loaded")
    utils.print_with_log(f"Loading model from cache with ID '{str(model_id)}'", "info")
    return pd.read_pickle((f"{cet_home}/model_cache/{current_model.data_file}/{str(model_id)}/model.pkl"))
        

def get_all_models_from_cache():
    utils.initial_checks()
    utils.assert_with_log(current_model.data_file is not None, "Attempted to access cache with no dataset loaded")
    model_list = []
    model_ids = os.listdir(f"{cet_home}/model_cache/{current_model.data_file}")
    for model_id in model_ids:
        model_list.append(build_model_from_cache(model_id))
    return model_list


def get_best_model(model_list=None, metric="r2"):
    if not model_list:
        model_list = get_all_models_from_cache()
    utils.assert_with_log(metric in utils.supported_metrics, f"Metric must be one of {utils.supported_metrics}")
    if metric in ["r2","out_sample_mse","rmse"]:
        sorted_models = sorted(model_list, key=lambda x : getattr(x, metric))
    elif metric == "out_sample_mse_reduction":
        sorted_models = sorted(model_list, key=lambda x : getattr(x, metric), reverse=True)
    elif metric == "out_sample_pred_int_cov":
        sorted_models = sorted(model_list, key=lambda x : abs(getattr(x, "out_sample_pred_int_cov")-.95))
    utils.print_with_log(f"Model with ID '{sorted_models[0].model_id}' is best for metric '{metric}'", "info")
    return sorted_models[0]


def get_all_model_ids():
    utils.initial_checks()
    if current_model.data_file is None:
        utils.print_with_log("You must load a dataset before accessing the cache", "error")
        return None
    else:
        return list(os.listdir(f"{cet_home}/model_cache/{current_model.data_file}"))
    

def get_model_by_id(model_id):
    return build_model_from_cache(model_id)


def load_dataset_from_file(datafile):
    utils.print_with_log(f"Loading dataset {datafile} as active dataset and resetting current model.", "info")
    # resets model when new dataset is loaded
    reset_model()
    current_model.data_file = datafile.split("/")[-1]
    current_model.full_data_path = datafile
    current_model.dataset = pd.read_csv(datafile)


def set_dataset(dataframe, dataset_name):
    utils.initial_checks()
    utils.print_with_log(f"Setting dataset '{dataset_name}' as active dataset and resetting current model.", "info")
    # resets model when new dataset is loaded
    reset_model()
    current_model.data_file = dataset_name
    save_path = f"{cet_home}/data/{dataset_name}.csv"
    utils.print_with_log(f"Dataset '{dataset_name}' saved to file path {save_path}", "info")
    dataframe.to_csv(save_path)
    current_model.full_data_path = save_path
    current_model.dataset = dataframe


def view_current_model():
    current_model.print()


def basic_existence_check(var):
    utils.assert_with_log(current_model.dataset is not None, "Attempting to set variables before loading dataset")
    utils.assert_with_log(var in current_model.dataset, f"Element {var} not found in data")


def set_target_variable(var, existence_check=True):
    if existence_check:
        basic_existence_check(var)
    current_model.model_id = None
    current_model.target_var = var
    current_model.model_vars = current_model.covariates + [current_model.target_var]


def set_time_column(var):
    basic_existence_check(var)
    current_model.model_id = None
    current_model.time_column = var


def set_panel_column(var):
    basic_existence_check(var)
    current_model.model_id = None
    current_model.panel_column = var


def add_transformation(var, transformations, keep_original_var=False):
    current_model.model_id = None
    if not isinstance(transformations, list):
        transformations = [transformations]
    all_transformations_valid = True
    for transform in transformations:
        if transform not in utils.supported_functions:
            all_transformations_valid = False
            utils.print_with_log(f"{transform}() not a supported function.","error")
    if all_transformations_valid:
        if var not in current_model.covariates and var != current_model.target_var:
            utils.print_with_log(f"{var} not in covariates list and is not target variable.", "error")
        elif var in current_model.covariates:
            for transform in transformations:
                if not keep_original_var:
                    remove_covariates(var)
                var = f"{transform}({var})"
            add_covariates(f"{var}", existence_check=False)
        elif var == current_model.target_var:
            for transform in transformations:
                var = f"{transform}({var})"
            set_target_variable(var, existence_check=False)


def add_covariates(vars, existence_check=True):
    if not isinstance(vars, list):
        vars = [vars]
    if existence_check:
        for var in vars:
            basic_existence_check(var)
    current_model.model_id = None
    for var in vars:
        if var not in current_model.covariates:
            current_model.covariates.append(var)
    current_model.model_vars = current_model.covariates + [current_model.target_var]


def add_fixed_effects(vars):
    if not isinstance(vars, list):
        vars = [vars]
    for var in vars:
        basic_existence_check(var)
    current_model.model_id = None
    for fe in vars:
        if fe not in current_model.fixed_effects:
            current_model.fixed_effects.append(fe)


def add_time_trend(var, exp):
    basic_existence_check(var)
    current_model.model_id = None
    time_trend = var + " " + str(exp)
    if time_trend not in current_model.time_trends:
        current_model.time_trends.append(time_trend)


def remove_covariates(vars):
    current_model.model_id = None
    if not isinstance(vars, list):
        vars = [vars]
    for var_to_remove in vars:
        current_model.covariates = [var for var in current_model.covariates if var != var_to_remove]
        current_model.model_vars = [var for var in current_model.model_vars if var != var_to_remove]


def remove_time_trend(var, exp):
    current_model.model_id = None
    time_trend = var + " " + str(exp)
    current_model.time_trends = [var for var in current_model.time_trends if var != time_trend]


def remove_transformation(var, transformations):
    current_model.model_id = None
    if not isinstance(transformations, list):
        transformations = [transformations]
    transformed_var = copy.deepcopy(var)
    for transform in transformations:
        transformed_var = f"{transform}({transformed_var})"
    if current_model.target_var == transformed_var:
        set_target_variable(var)
    elif transformed_var in current_model.covariates:
        current_model.covariates = [var for var in current_model.covariates if var != transformed_var]
        current_model.model_vars = [var for var in current_model.model_vars if var != transformed_var]
    else:
        utils.print_with_log(f"Transformed var f{transformed_var} not found", "error")


def remove_fixed_effect(fe):
    current_model.model_id = None
    current_model.fixed_effects = [var for var in current_model.fixed_effects if var != fe]


def add_random_effect(var, group):
    current_model.model_id = None
    if current_model.random_effects != [var, group]:
        if current_model.random_effects != None:
            utils.print_with_log("Attempted to set multiple random effects when only one is supported.", "error")
        else:
            current_model.random_effects = [var, group]
            if var in current_model.covariates:
                remove_covariates(var)


def remove_random_effect(add_to_covariate_list=True):
    current_model.model_id = None
    if current_model.random_effects is not None:
        if add_to_covariate_list:
            add_covariates(current_model.random_effects[0])
        current_model.random_effects = None


def run_spatial_regression(reg_type, model=None, k=5):
    if model is None:
        model = current_model
    utils.initial_checks()
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    if model.model_id is None:
        model.model_id = time.time()
        utils.print_with_log(f"Model ID {model.model_id} assigned to model.", "info")
    utils.model_checks(model)
    regression.run_spatial_regression(model, reg_type, model.model_id, k)
    return model.model_id


def run_quantile_regression(q, model=None, std_error_type="nonrobust"):
    if model is None:
        model = current_model
    utils.initial_checks()
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    if model.model_id is None:
        model.model_id = time.time()
        utils.print_with_log(f"Model ID {model.model_id} assigned to model.", "info")
    utils.model_checks(model)
    if isinstance(q, list):
        for val in q:
            regression.run_quantile_regression(model, std_error_type, model.model_id, val)
    else:
        regression.run_quantile_regression(model, std_error_type, model.model_id, q)
    return model.model_id


def run_adf_panel_unit_root_tests(model=None):
    if model is None:
        model = current_model
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    if model.model_id is None:
        model.model_id = time.time()
        utils.print_with_log(f"Model ID {model.model_id} assigned to model.", "info")
    utils.model_checks(model)
    utils.print_with_log(f"Running Panel Unit Root Tests against model with ID '{model.model_id}'", "info")
    return stat_tests.panel_unit_root_tests(model)


def run_engle_granger_cointegration_check(model=None):
    if model is None:
        model = current_model
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    if model.model_id is None:
        model.model_id = time.time()
        utils.print_with_log(f"Model ID {model.model_id} assigned to model.", "info")
    utils.model_checks(model)
    utils.print_with_log(f"Running Cointegration Tests against model with ID '{model.model_id}'", "info")
    return stat_tests.cointegration_tests(model)


def run_pesaran_cross_sectional_dependence_check(model=None):
    if model is None:
        model = current_model
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    if model.model_id is None:
        model.model_id = time.time()
        utils.print_with_log(f"Model ID {model.model_id} assigned to model.", "info")
    utils.model_checks(model)
    utils.print_with_log(f"Running Cross-Sectional Dependence Tests against model with ID '{model.model_id}'", "info")
    return stat_tests.cross_sectional_dependence_tests(model)


def run_bayesian_regression(model=None, num_samples=1000, overwrite_samples=False):
    if model is None:
        model = current_model
    utils.initial_checks()
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    if model.model_id is None:
        model.model_id = time.time()
        utils.print_with_log(f"Model ID {model.model_id} assigned to model.", "info")
    utils.model_checks(model)
    regression.run_bayesian_regression(model, num_samples, overwrite=overwrite_samples)
    return model.model_id


def run_block_bootstrap(model=None, std_error_type="nonrobust", num_samples=1000, overwrite_samples=False):
    if model is None:
        model = current_model
    utils.initial_checks()
    # TODO: check to see if bootstrap already ran for this model
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    if model.model_id is None:
        model.model_id = time.time()
        utils.print_with_log(f"Model ID {model.model_id} assigned to model.", "info")
    utils.model_checks(model)
    regression.run_block_bootstrap(model, std_error_type, num_samples, overwrite=overwrite_samples)
    return model.model_id


def extract_raster_data(raster_file, shape_file=None, weights=None, weight_file=None):
    utils.initial_checks()
    if shape_file is None:
        shape_file = str(files("climate_econometrics_toolkit.preprocessed_data.shape_files.country_shapes").joinpath('country.shp'))
        utils.print_with_log("No shape file specified for extraction. Using default country shape file.", "info")
    if weight_file is not None and weights is not None:
        utils.print_with_log(f"Both 'weights' and 'weight_file' arguments were supplied; defaulting to use in built-in weights: {weights}", "warning")
    if weights is not None:
        utils.assert_with_log(weights in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
        if weights == "cropweighted":
            weight_file = str(files("climate_econometrics_toolkit.preprocessed_data.raster_weights.cropland_weights").joinpath('cropland_weights_5m.tif'))
        elif weights == "maizeweighted":
            weight_file = str(files("climate_econometrics_toolkit.preprocessed_data.raster_weights.cropland_weights").joinpath('maize_weights_5m.tif'))
        elif weights == "wheatweighted":
            weight_file = str(files("climate_econometrics_toolkit.preprocessed_data.raster_weights.cropland_weights").joinpath('wheat_weights_5m.tif'))
        elif weights == "soybeanweighted":
            weight_file = str(files("climate_econometrics_toolkit.preprocessed_data.raster_weights.cropland_weights").joinpath('soybean_weights_5m.tif'))
        elif weights == "riceweighted":
            weight_file = str(files("climate_econometrics_toolkit.preprocessed_data.raster_weights.cropland_weights").joinpath('rice_weights_5m.tif'))
        elif weights == "popweighted":
            weight_file = str(files("climate_econometrics_toolkit.preprocessed_data.raster_weights.population_weights").joinpath('population_weights_5m.tif'))
    utils.print_with_log(f"Extracting raster data using raster_file {raster_file}; shape_file {shape_file}; weights file {weight_file}.", "info")
    return extract.extract_raster_data(raster_file, shape_file, weight_file)


def aggregate_raster_data_to_year_level(data, climate_var_name, aggregation_func, subperiods_per_year, starting_year, shape_file=None, geo_identifier=None, subperiods_to_use=None, crop=None):
    if shape_file is not None:
        utils.assert_with_log(geo_identifier is not None, "If shape_file is specified, geo_identifier must also be specified.")
    if shape_file is None and geo_identifier is not None:
        utils.print_with_log(f"Specified geo_identifier {geo_identifier} is ignored with use of default country shapes file", "warning")
    if shape_file is None:
        shape_file = str(files("climate_econometrics_toolkit.preprocessed_data.shape_files.country_shapes").joinpath('country.shp'))
        geo_identifier = "GMI_CNTRY"
        utils.print_with_log("No shape file specified for aggregation. Using default country shape file.", "info")
    utils.print_with_log(f"Aggregating raster data using function {aggregation_func}", "info")
    return extract.aggregate_yearly_raster_data(data, shape_file, climate_var_name, aggregation_func, geo_identifier, subperiods_per_year, starting_year, subperiods_to_use, crop)


def aggregate_raster_data_to_month_level(data, climate_var_name, aggregation_func, subperiods_per_day, starting_year, starting_month, shape_file=None, geo_identifier=None):
    if shape_file is not None:
        utils.assert_with_log(geo_identifier is not None, "If shape_file is specified, geo_identifier must also be specified.")
    if shape_file is None and geo_identifier is not None:
        utils.print_with_log(f"Specified geo_identifier {geo_identifier} is ignored with use of default country shapes file", "warning")
    if shape_file is None:
        shape_file = str(files("climate_econometrics_toolkit.preprocessed_data.shape_files.country_shapes").joinpath('country.shp'))
        geo_identifier = "GMI_CNTRY"
        utils.print_with_log("No shape file specified for aggregation. Using default country shape file.", "info")
    utils.print_with_log(f"Aggregating raster data using function {aggregation_func}", "info")
    return extract.aggregate_monthly_raster_data(data, shape_file, climate_var_name, aggregation_func, geo_identifier, subperiods_per_day, starting_year, starting_month)


def predict_out_of_sample(model, data, transform_data=False, var_map=None):
    utils.initial_checks()
    if isinstance(model, str) or isinstance(model, float):
        model = get_model_by_id(model)
    utils.print_with_log(f"Generating out-of-sample predictions for Model with ID '{model.model_id}' using supplied data", "info")
    return predict.predict_out_of_sample(model, copy.deepcopy(data), transform_data, var_map)


def call_user_prediction_function(function_name, args):
    utils.print_with_log(f"Called user prediction function '{function_name}'.", "info")
    func = getattr(user_predict, function_name)
    return func(*args)


def transform_data(model, include_target_var=True, demean=False):
    utils.print_with_log(f"Running data transformation with the following settings: include_target_var: {include_target_var}, demean: {demean}", "info")
    return utils.transform_data(copy.deepcopy(model.dataset), model, include_target_var, demean)


def export_data(model, format="csv"):
    if not model.model_id is None:
        dataset_id = str(model.model_id)
    else:
        dataset_id = str(time.time())
    utils.assert_with_log(format in ["csv","stata","rdata"], "Format must be one of: 'csv', 'stata', 'rdata'.")
    transformed_data = transform_data(model)
    if format == "csv":
        transformed_data.to_csv(f"{cet_home}/data/{dataset_id}.csv")
        utils.print_with_log(f"Dataset exported to '{cet_home}/data/{dataset_id}.csv", "info")
    elif format == "stata":
        transformed_data.to_stata(f"{cet_home}/data/{dataset_id}.dta")
        utils.print_with_log(f"Dataset exported to '{cet_home}/data/{dataset_id}.dta", "info")
    elif format == "rdata":
        pyreadr.pyreadr.write_rdata(f"{cet_home}/data/{dataset_id}.rdata", transformed_data, dataset_id)
        utils.print_with_log(f"Dataset exported to '{cet_home}/data/{dataset_id}.rdata", "info")


def reset_model():
    global current_model
    current_model = ClimateEconometricsModel()