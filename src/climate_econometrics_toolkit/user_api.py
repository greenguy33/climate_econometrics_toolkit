import pandas as pd
import numpy as np
import os
import copy
import time
import logging

from importlib.resources import files
from functools import reduce
from metpy.calc import heat_index
from metpy.units import units
import progressbar
import itertools as it

from climate_econometrics_toolkit import interface_api as api
from climate_econometrics_toolkit.ClimateEconometricsModel import ClimateEconometricsModel
import climate_econometrics_toolkit.utils as utils
from climate_econometrics_toolkit import regression as regression
from climate_econometrics_toolkit import prediction as predict
from climate_econometrics_toolkit import raster_extraction as extract
from climate_econometrics_toolkit import user_prediction_functions as user_predict
from climate_econometrics_toolkit import stat_tests as stat_tests

model = ClimateEconometricsModel()

cet_home = os.getenv("CETHOME")

# TODO: utils.assert_with_log(types for user input to each method
# TODO: the transformations user interface is clunky and unintuitive, ideally should be re-worked
# TODO: there is inconsistency about when methods are accepting a model argument vs. when they are using the pre-loaded model
# TODO: what if you want to load and run a model from the cache using evaluate_model, spatial regression or quantile regression?

def model_checks():
    checks = {
        "No dataset loaded." : model.dataset is not None,
        "No target variable set." : not pd.isnull(model.target_var),
         # At least one covariate is required even if random effects are applied
		"No model covariates found." : model.covariates != [],
		"No time-based column set." : model.time_column is not None,
		"No panel column set." : model.panel_column is not None
    }
    for key, check in checks.items():
        if not check:
            utils.print_with_log(f"{key} Please update your model.", "error")
            return False
    return True


def run_specification_search(metric="out_sample_mse_reduction"):
    utils.assert_with_log(metric in utils.supported_metrics, f"Supplied metric must be one of: {utils.supported_metrics}")
    if model_checks():

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
            reset_model()
            model.dataset = dataset
            model.datafile = "ds1"
            set_panel_column(panel_column)
            set_time_column(time_column)
            model.target_var = target_var
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
            evaluate_model()
            model_list.append(copy.deepcopy(model))

        best_model = get_best_model(model_list, metric)
        utils.print_with_log(f"Best model for metric {metric}: {best_model.to_string()}", "info")
        return best_model


def compute_degree_days(years, countries, threshold, mode="above", panel_column_name="ISO3", time_column_name="year", crop=None, second_threshold=None):
    utils.print_with_log(f"Computing degree days with mode '{mode}', threshold '{threshold}', using panel column '{panel_column_name}' and time_column '{time_column_name}'", "info")
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
    if crop is not None:
        col_name += f"_{crop}_growing_season"
        # if crop specified, get growing season dates for specified crop
        country_start_days, country_end_days = utils.get_growing_season_data_by_crop(crop)
    res = {panel_column_name:[], time_column_name:[], col_name:[]}
    years_missing_data, countries_missing_temp_data, countries_missing_crop_data = set(), set(), set()
    for year in years:
        try:
            file = files(f"climate_econometrics_toolkit.preprocessed_data.daily_temp.unweighted").joinpath(f'temp.daily.bycountry.unweighted.{year}.csv')
            daily_temp_data = pd.read_csv(file)
            for country in countries:
                if country in daily_temp_data:
                    if crop is None:
                        # if no crop specified, compute degree days for entire year
                        daily_temps = daily_temp_data[country]
                    else:
                        # if crop specified, extract only crop growing days
                        try:
                            if country_end_days[country] < country_start_days[country]:
                                daily_temps = pd.concat([daily_temp_data[country].iloc[:int(country_end_days[country])+1],daily_temp_data[country].iloc[int(country_start_days[country]):]])
                            else:
                                daily_temps = daily_temp_data[country].iloc[int(country_start_days[country]):int(country_end_days[country])+1]
                        except (KeyError,ValueError):
                            # except case where no crop growing season data exists
                            daily_temps = None
                    if daily_temps is not None:
                        if mode == "above":
                            degree_days = int(np.sum([val-threshold for val in daily_temps if val > threshold]))
                        elif mode == "below":
                            degree_days = int(np.sum([threshold-val for val in daily_temps if val < threshold]))
                        elif mode == "between":
                            degree_days = int(np.sum([val-threshold for val in daily_temps if val > threshold and val < second_threshold]))
                    else:
                        degree_days = pd.NA
                        countries_missing_crop_data.add(country)
                    res[panel_column_name].append(country)
                    res[time_column_name].append(year)
                    res[col_name].append(degree_days)
                else:
                    countries_missing_temp_data.add(country)
        except FileNotFoundError:
            years_missing_data.add(year)
    if len(countries_missing_temp_data) > 0:
        utils.print_with_log(f"No daily temperature data available for countries: {sorted(countries_missing_temp_data)}", "warning")
    if len(countries_missing_crop_data) > 0:
        utils.print_with_log(f"No {crop} growing season data available for countries: {sorted(countries_missing_crop_data)}", "warning")
    if len(years_missing_data) > 0:
        utils.print_with_log(f"No daily temperature data available for years: {sorted(years_missing_data)}", "warning")        
    return pd.DataFrame.from_dict(res)


def add_degree_days_to_dataframe(dataframe, threshold, panel_column = "ISO3", time_column = "year", mode = "above", crop=None, second_threshold=None):
    utils.assert_with_log(panel_column in dataframe, f"Specified panel column {panel_column} not in supplied dataframe.")
    utils.assert_with_log(time_column in dataframe, f"Specified time column {time_column} not in supplied dataframe.")
    degree_days_df = compute_degree_days(set(dataframe[time_column]), set(dataframe[panel_column]), threshold, mode, crop=crop, second_threshold=second_threshold)
    utils.assert_with_log(len(degree_days_df) > 0, f"No daily temperature data available for supplied columns {panel_column}/{time_column}.")
    merge_strategy = "outer"
    utils.print_with_log(f"Merging supplied dataframe with degree days dataframe using threshold '{threshold}' and merge strategy '{merge_strategy}'", "info")
    return pd.merge(dataframe, degree_days_df, on=[panel_column,time_column], how=merge_strategy)


def integrate(dataframes, keep_na=False, panel_column="ISO3", time_column="year"):
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


def load_climate_data(weight="unweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    file = files("climate_econometrics_toolkit.preprocessed_data.weather_data").joinpath(f'NCEP_reanalaysis_climate_data_1948_2024_{weight}.csv')
    return pd.read_csv(file)


def load_temperature_humidity_index_data(weight="unweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    file = files("climate_econometrics_toolkit.preprocessed_data.temperature_humidity_index").joinpath(f'temperature_humidity_index_{weight}_1948_2024.csv')
    return pd.read_csv(file)


def load_ndvi_data(weight="unweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
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
    return pd.read_csv(file)


def load_worldbank_gdp_data():
    file = files("climate_econometrics_toolkit.preprocessed_data").joinpath('worldbank_global_GDP_1961_2023.csv')
    return pd.read_csv(file)


def load_spei_data(weight="unweighted"):
    utils.assert_with_log(weight in utils.supported_weights, f"Weight argument must be one of: {utils.supported_weights}.")
    file = files("climate_econometrics_toolkit.preprocessed_data.SPEI").joinpath(f'spei_{weight}.csv')
    return pd.read_csv(file)


def get_temperature_humidity_index(temp_data, relative_humidity_data):
    # requires temp data in celsius and relative humidity data (percentage) between 0 and 100
    utils.print_with_log("Generating temperature/humidity index using supplied data", "info")
    return heat_index(
        temp_data * units.degC, 
        relative_humidity_data * units.percent,
        mask_undefined=False
    ).magnitude


def evaluate_model(std_error_type="nonrobust"):
    utils.initial_checks()
    # TODO: check to see if this model is already in cache, if so return that model rather than re-evaluating the same model
    if model_checks():
        _, _, return_string = api.run_model_analysis(copy.deepcopy(model.dataset), std_error_type, model, save_model_to_cache=True, save_result_to_file=False)
        if return_string != "": utils.print_with_log(return_string, "error")
        if model != None:
            utils.print_with_log(f"Model assigned ID: {model.model_id}", "info")
            return str(model.model_id)
        

def build_model_from_cache(model_id):
    utils.initial_checks()
    utils.assert_with_log(model.data_file == None, "Attempted to access cache with no dataset loaded")
    utils.print_with_log(f"Loading model from cache with ID {model_id}", "info")
    return pd.read_pickle((f"{cet_home}/model_cache/{model.data_file}/{model_id}/model.pkl"))
        

def get_all_models_from_cache():
    utils.initial_checks()
    utils.assert_with_log(model.data_file == None, "Attempted to access cache with no dataset loaded")
    model_list = []
    model_ids = os.listdir(f"{cet_home}/model_cache/{model.data_file}")
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
    utils.print_with_log(f"Model with ID {sorted_models[0].model_id} is best for metric {metric}", "info")
    return sorted_models[0]


def get_all_model_ids():
    utils.initial_checks()
    if model.data_file == None:
        utils.print_with_log("You must load a dataset before accessing the cache", "error")
        return None
    else:
        return list(os.listdir(f"{cet_home}/model_cache/{model.data_file}"))
    

def get_model_by_id(model_id):
    return build_model_from_cache(model_id)


def load_dataset_from_file(datafile):
    utils.print_with_log(f"Loading dataset {datafile} as active dataset and resetting current model.", "info")
    # resets model when new dataset is loaded
    reset_model()
    model.data_file = datafile.split("/")[-1]
    model.full_data_path = datafile
    model.dataset = pd.read_csv(datafile)


def set_dataset(dataframe, dataset_name):
    utils.initial_checks()
    utils.print_with_log(f"Setting dataset '{dataset_name}' as active dataset and resetting current model.", "info")
    # resets model when new dataset is loaded
    reset_model()
    model.data_file = dataset_name
    save_path = f"{cet_home}/data/{dataset_name}.csv"
    utils.print_with_log(f"Dataset '{dataset_name}' saved to file path {save_path}", "info")
    dataframe.to_csv(save_path)
    model.full_data_path = save_path
    model.dataset = dataframe


def view_current_model():
    model.print()


def basic_existence_check(var):
    if model.dataset is None:
        utils.print_with_log("Attempting to set variables before loading dataset", "error")
        return False
    elif var not in model.dataset:
        utils.print_with_log(f"Element {var} not found in data", "error")
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


def add_transformation(var, transformations, keep_original_var=False):
    if not isinstance(transformations, list):
        transformations = [transformations]
    all_transformations_valid = True
    for transform in transformations:
        if transform not in utils.supported_functions:
            all_transformations_valid = False
            utils.print_with_log(f"{transform}() not a supported function.","error")
    if all_transformations_valid:
        if var not in model.covariates and var != model.target_var:
            utils.print_with_log(f"{var} not in covariates list and is not target variable.", "error")
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
        utils.print_with_log(f"Transformed var f{transformed_var} not found", "error")


def remove_fixed_effect(fe):
    model.fixed_effects = [var for var in model.fixed_effects if var != fe]


def add_random_effect(var, group):
    if model.random_effects != [var, group]:
        if model.random_effects != None:
            utils.print_with_log("Attempted to set multiple random effects when only one is supported.", "error")
        else:
            model.random_effects = [var, group]
            if var in model.covariates:
                remove_covariates(var)


def remove_random_effect(add_to_covariate_list=True):
    if model.random_effects is not None:
        if add_to_covariate_list:
            add_covariates(model.random_effects[0])
        model.random_effects = None


def run_spatial_regression(reg_type, std_error_type="nonrobust", geometry_column=None, k=5, num_lags=1):
    utils.initial_checks()
    model_id = time.time()
    model.model_id = model_id
    regression.run_spatial_regression(model, std_error_type, reg_type, model_id, geometry_column, k, num_lags)


def run_quantile_regression(q, std_error_type="nonrobust"):
    utils.initial_checks()
    model_id = time.time()
    model.model_id = model_id
    if isinstance(q, list):
        for val in q:
            regression.run_quantile_regression(model, std_error_type, model_id, val)
    else:
        regression.run_quantile_regression(model, std_error_type, model_id, q)


def run_adf_panel_unit_root_tests():
    utils.assert_with_log(model.dataset is not None, "Dataset must be set before running panel unit root tests.")
    utils.assert_with_log(model.target_var is not None, "Target variable must be set before running panel unit root tests.")
    utils.assert_with_log(model.target_var is not None, "Covariates must be set before running panel unit root tests.")
    utils.assert_with_log(model.time_column is not None, "Time column must be set before running panel unit root tests.")
    utils.assert_with_log(model.panel_column is not None, "Panel column must be set before running panel unit root tests.")
    utils.print_with_log(f"Runing Panel Unit Root Tests against current model", "info")
    return stat_tests.panel_unit_root_tests(model)


def run_engle_granger_cointegration_check():
    utils.assert_with_log(model.dataset is not None, "Dataset must be set before running cointegration check.")
    utils.assert_with_log(model.target_var is not None, "Target variable must be set before running cointegration check.")
    utils.assert_with_log(model.target_var is not None, "Covariates must be set before running cointegration check.")
    utils.assert_with_log(model.time_column is not None, "Time column must be set before running cointegration check.")
    utils.assert_with_log(model.panel_column is not None, "Panel column must be set before running cointegration check.")
    utils.print_with_log(f"Runing Cointegration Tests against current model", "info")
    return stat_tests.cointegration_tests(model)


def run_pesaran_cross_sectional_dependence_check():
    utils.assert_with_log(model.dataset is not None, "Dataset must be set before running cointegration check.")
    utils.assert_with_log(model.target_var is not None, "Target variable must be set before running cointegration check.")
    utils.assert_with_log(model.target_var is not None, "Covariates must be set before running cointegration check.")
    utils.assert_with_log(model.time_column is not None, "Time column must be set before running cointegration check.")
    utils.assert_with_log(model.panel_column is not None, "Panel column must be set before running cointegration check.")
    utils.print_with_log(f"Runing Cross-Sectional Dependence Tests against current model", "info")
    return stat_tests.cross_sectional_dependence_tests(model)


def run_bayesian_regression(model, num_samples=1000):
    utils.initial_checks()
    # TODO: check to see if bayesian inference already ran for this model
    if isinstance(model, str):
        model = get_model_by_id(model)
    regression.run_bayesian_regression(model, num_samples)


def run_block_bootstrap(model, std_error_type, num_samples=1000):
    utils.initial_checks()
    # TODO: check to see if bootstrap already ran for this model
    if isinstance(model, str):
        model = get_model_by_id(model)
    utils.assert_with_log(model != None, "NoneType passed as model object")
    regression.run_block_bootstrap(model, std_error_type, num_samples)


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


def aggregate_raster_data(data, climate_var_name, aggregation_func, subperiods_per_year, starting_year, shape_file=None, geo_identifier=None, subperiods_to_use=None, crop=None):
    if shape_file is not None:
        utils.assert_with_log(geo_identifier is not None, "If shape_file is specified, geo_identifier must also be specified.")
    if shape_file is None and geo_identifier is not None:
        utils.print_with_log(f"Specified geo_identifier {geo_identifier} is ignored with use of default country shapes file", "warning")
    if shape_file is None:
        shape_file = str(files("climate_econometrics_toolkit.preprocessed_data.shape_files.country_shapes").joinpath('country.shp'))
        geo_identifier = "GMI_CNTRY"
        utils.print_with_log("No shape file specified for aggregation. Using default country shape file.", "info")
    utils.print_with_log(f"Aggregating raster data using function {aggregation_func}", "info")
    return extract.aggregate_raster_data(data, shape_file, climate_var_name, aggregation_func, geo_identifier, subperiods_per_year, starting_year, subperiods_to_use, crop)


def predict_out_of_sample(model, data, transform_data=False, var_map=None):
    utils.initial_checks()
    if isinstance(model, str):
        model = get_model_by_id(model)
    utils.print_with_log(f"Generating out-of-sample predictions for Model with ID {model.model_id} using supplied data", "info")
    return predict.predict_out_of_sample(model, copy.deepcopy(data), transform_data, var_map)


def call_user_prediction_function(function_name, args):
    utils.print_with_log(f"User prediction function '{function_name}' called with args: {args}", "info")
    func = getattr(user_predict, function_name)
    return func(*args)


def transform_data(data, model, include_target_var=True, demean=False):
    utils.print_with_log(f"Runing data transformation with the following settings: include_target_var: {include_target_var}, demean: {demean}", "info")
    return utils.transform_data(copy.deepcopy(data), model, include_target_var, demean)


def reset_model():
    global model
    model = ClimateEconometricsModel()