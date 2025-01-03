import pandas as pd
import shutil
import os
import time

import climate_econometrics_toolkit.evaluate_model as ce_eval
import climate_econometrics_toolkit.model_builder as mb
import climate_econometrics_toolkit.utils as utils
import climate_econometrics_toolkit.regression as regression
import climate_econometrics_toolkit.prediction as predict

pd.set_option('display.min_rows', 100)
pd.set_option('display.max_rows', 100)

cet_home = os.getenv("CETHOME")

# TODO: facilitate loading a model directly from the model cache into the API
# To do this you might need to add a "get model id" button in the interface which the user can copy/paste in their code
# TODO: refactor code into API and interface directories

def run_model_analysis(data, model, save_to_cache=True):
	model_id = None
	regression_result = None
	return_string = ""
	data.sort_values([model.panel_column, model.time_column]).reset_index(drop=True)
	data.columns = data.columns.str.replace(' ', '_') 
	if len(set(data.columns)) != len(data.columns): 
		return_string += "\nTwo column names in dataset collide when spaces are removed. Please correct."
	else:
		model = ce_eval.evaluate_model(data, model)
		model_id = time.time()
		if save_to_cache:
			model.save_model_to_cache(model_id)
		regression_result = model.regression_result
		# TODO: don't print out fixed effect/time trend coefficients
		print(regression_result.summary2().tables[1])
	return model_id, regression_result, return_string


def evaluate_model(data_file, model, panel_column, time_column):
	# model = mb.parse_cxl(model)
	data = pd.read_csv(data_file)
	model, unused_nodes = mb.parse_model_input(model, data_file, panel_column, time_column)
	model.dataset = data
	if len(unused_nodes) > 0:
		return_string += "\nWARNING: The following nodes are unused in the regression. " + str(unused_nodes)
	model_id, regression_result, return_string = run_model_analysis(data, model)
	return model_id, regression_result, return_string


def get_best_model_for_dataset(filename):
	max_mse_red, model_id = None, None
	out_sample_mses = {}
	cache_dir = os.listdir(f"{cet_home}/model_cache/")
	if filename not in cache_dir:
		return None, None
	for file in os.listdir(f"{cet_home}/model_cache/{filename}"):
		out_sample_mses[file] = float(utils.get_attribute_from_model_file(filename, "out_sample_mse_reduction", file))
	if len(out_sample_mses) > 0:
		max_mse_red = max(out_sample_mses.values())
		model_id = [file for file in out_sample_mses if out_sample_mses[file] == max_mse_red][0]
	return max_mse_red, model_id


def clear_model_cache(dataset):
	if dataset == None:
		shutil.rmtree(f"{cet_home}/model_cache/")
		os.makedirs(f"{cet_home}/model_cache/")
	else:
		if os.path.isdir(f"{cet_home}/model_cache/{dataset}"):
			shutil.rmtree(f"{cet_home}/model_cache/{dataset}")


def run_bayesian_regression(data_file, model_id, use_threading=False):
	data_file_short = data_file.split("/")[-1]
	model, panel_column, time_column = utils.construct_model_input_from_cache(data_file_short, model_id)
	model, _ = mb.parse_model_input(model, data_file, panel_column, time_column)
	model.dataset = pd.read_csv(data_file).sort_values([model.time_column, model.panel_column]).reset_index(drop=True)
	model.model_id = model_id
	regression.run_bayesian_regression(model, use_threading)


def run_block_bootstrap(data_file, model_id, use_threading=False):
	data_file_short = data_file.split("/")[-1]
	model, panel_column, time_column = utils.construct_model_input_from_cache(data_file_short, model_id)
	model, _ = mb.parse_model_input(model, data_file, panel_column, time_column)
	model.dataset = pd.read_csv(data_file).sort_values([model.time_column, model.panel_column]).reset_index(drop=True)
	model.model_id = model_id
	regression.run_block_bootstrap(model, use_threading)


def extract_raster_data(raster_file, shape_file, weights_file=None):
	return predict.extract_raster_data(raster_file, shape_file, weights_file)


def aggregate_raster_data(data, shape_file, climate_var_name, aggregation_func, timescale, geo_identifier):
	return predict.aggregate_raster_data(data, shape_file, climate_var_name, aggregation_func, geo_identifier, timescale, months_to_use=None)


def start_interface():
	utils.start_user_interface()
