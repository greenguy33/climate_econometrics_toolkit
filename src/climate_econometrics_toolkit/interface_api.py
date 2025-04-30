import pandas as pd
import shutil
import os
import threading

import climate_econometrics_toolkit.evaluate_model as ce_eval
import climate_econometrics_toolkit.model_builder as mb
import climate_econometrics_toolkit.utils as utils
import climate_econometrics_toolkit.regression as regression
import climate_econometrics_toolkit.prediction as predict
import climate_econometrics_toolkit.user_prediction_functions as user_predict
import climate_econometrics_toolkit.stat_tests as stat_tests
from climate_econometrics_toolkit import raster_extraction as extract


pd.set_option('display.min_rows', 100)
pd.set_option('display.max_rows', 100)

cet_home = os.getenv("CETHOME")

std_error_name_map = {
	"Nonrobust":"nonrobust",
	"White-Huber":"whitehuber",
	"Driscoll-Kraay":"driscollkraay",
	"Newey-West":"neweywest",
	"Time-clustered":"clusteredtime",
	"Space-clustered":"clusteredspace"
}

# TODO: refactor code into API and interface directories

def run_model_analysis(data, std_error_type, model, save_model_to_cache=True, save_result_to_file=True):
	regression_result = None
	return_string = ""
	data.sort_values([model.panel_column, model.time_column]).reset_index(drop=True)
	data.columns = data.columns.str.replace(' ', '_') 
	if len(set(data.columns)) != len(data.columns): 
		return_string += "\nTwo column names in dataset collide when spaces are removed. Please correct."
		model = None
	else:
		model = ce_eval.evaluate_model(data, std_error_type, model)
		if save_model_to_cache:
			model.save_model_to_cache()
		if save_result_to_file:
			model.save_result_to_file()
		regression_result = model.regression_result
		# TODO: don't print out fixed effect/time trend coefficients
		try:
			print(regression_result.summary2().tables[1])
		except:
			try:
				print(regression_result.summary())
			except:
				print(regression_result.summary)
		model.save_OLS_regression_script(std_error_type)
	return model, regression_result, return_string


def build_model_object_from_canvas(input_list, data_file, panel_column, time_column):
	return mb.parse_model_input(input_list, data_file, panel_column, time_column)


def evaluate_model(data_file, std_error_type, model, panel_column, time_column):
	data = pd.read_csv(data_file)
	model, unused_nodes = build_model_object_from_canvas(model, data_file, panel_column, time_column)
	model.dataset = data
	if len(unused_nodes) > 0:
		return_string += "\nWARNING: The following nodes are unused in the regression. " + str(unused_nodes)
	# TODO: check to see if this model is already in cache, if so return that model rather than re-evaluating the same model
	return run_model_analysis(data, std_error_name_map[std_error_type], model)


def clear_model_cache(dataset):
	if dataset == None:
		shutil.rmtree(f"{cet_home}/model_cache/")
		os.makedirs(f"{cet_home}/model_cache/")
	else:
		if os.path.isdir(f"{cet_home}/model_cache/{dataset}"):
			shutil.rmtree(f"{cet_home}/model_cache/{dataset}")


def run_bayesian_regression(model, use_threading=True):
	# TODO: check to see if bayesian inference already ran for this model
	regression.run_bayesian_regression(model, 1000, use_threading=use_threading)


def run_block_bootstrap(model, std_error_type, use_threading=True):
	# TODO: check to see if bootstrap already ran for this model
	regression.run_block_bootstrap(model, std_error_name_map[std_error_type], 1000, use_threading=use_threading)


def run_spatial_regression(model, reg_type, model_id, geometry_column, std_error_type, k, num_lags):
	regression.run_spatial_regression(model, std_error_type, reg_type, model_id, geometry_column, k, num_lags)


def run_quantile_regression(model, model_id, q, std_error_type):
	if isinstance(q, list):
		for val in q:
			regression.run_quantile_regression(model, std_error_type, model_id, val)
	else:
		regression.run_quantile_regression(model, std_error_type, model_id, q)


def run_panel_unit_root_tests(model, model_id):
	stat_tests.panel_unit_root_tests(model).to_csv(f"{cet_home}/statistical_tests_output/panel_unit_root_tests/{model_id}.csv")


def run_cointegration_tests(model, model_id):
	stat_tests.cointegration_tests(model).to_csv(f"{cet_home}/statistical_tests_output/cointegration_tests/{model_id}.csv")


def run_cross_sectional_dependence_tests(model, model_id):
	stat_tests.cross_sectional_dependence_tests(model).to_csv(f"{cet_home}/statistical_tests_output/cross_sectional_dependence_tests/{model_id}.csv")


def extract_raster_data(raster_file, shape_file, weights_file=None):
	return extract.extract_raster_data(raster_file, shape_file, weights_file)


def aggregate_raster_data(data, shape_file, climate_var_name, aggregation_func, geo_identifier, subperiods_per_year, starting_year, crop):
	return extract.aggregate_raster_data(data, shape_file, climate_var_name, aggregation_func, geo_identifier, subperiods_per_year, starting_year, subperiods_to_use=None, crop=crop)


def predict_out_of_sample(model, out_sample_data_file, function_name, use_threading=True):
	if use_threading:
		thread = threading.Thread(target=predict_function,name="prediction_thread",args=(model,out_sample_data_file,function_name))
		thread.daemon = True
		thread.start()
	else:
		predict_function(model, out_sample_data_file)
	

def predict_function(model, out_sample_data_files, function_name):
	for out_sample_data_file in out_sample_data_files:
		out_sample_data = pd.read_csv(out_sample_data_file)
		predictions = predict.predict_out_of_sample(model, out_sample_data, True, None)
		if function_name != "None":
			func = getattr(user_predict, function_name)
			predictions = func(model, predictions)
		data_file_short = out_sample_data_file.split("/")[-1].rpartition('.')[0]
		filename = f"{cet_home}/predictions/predictions_{model.model_id}_{data_file_short}"
		if function_name != "None":
			filename += f"_{function_name}"
		predictions.to_csv(filename+".csv")


def start_interface():
	utils.start_user_interface()
