import pandas as pd
import shutil
import os

import climate_econometrics_toolkit.evaluate_model as ce_eval
import climate_econometrics_toolkit.model_builder as mb
import climate_econometrics_toolkit.climate_econometrics_utils as utils
import climate_econometrics_toolkit.climate_econometrics_regression as regression


def evaluate_model(data_file, model):
	# model = mb.parse_cxl(model)
	return_string = ""
	model_id = None
	regression_result = None
	# try:
	model, unused_nodes = mb.parse_model_input(model, data_file)
	if len(unused_nodes) > 0:
		return_string += "\nWARNING: The following nodes are unused in the regression. " + str(unused_nodes)
	data = pd.read_csv(data_file)
	data.columns = data.columns.str.replace(' ', '_') 
	if len(set(data.columns)) != len(data.columns): 
		return_string += "\nTwo column names in dataset collide when spaces are removed. Please correct."
	else:
		model = ce_eval.evaluate_model(data, model)
		# return_string += "\n" + utils.compare_to_last_model(model, data_file)
		model_id = model.save_model_to_cache()
		regression_result = model.regression_result
	# except BaseException as e:
	# 	return_string += "\nERROR: " + str(e)
	return model_id, regression_result, return_string


def get_best_model_for_dataset(filename):
	# TODO: make this path more flexible
	max_mse_red, model_id = None, None
	out_sample_mses = {}
	cache_dir = os.listdir("model_cache/")
	if filename not in cache_dir:
		return None, None
	for file in os.listdir(f"model_cache/{filename}"):
		out_sample_mses[file] = float(utils.get_attribute_from_model_file(filename, "out_sample_mse_reduction", file))
	if len(out_sample_mses) > 0:
		max_mse_red = max(out_sample_mses.values())
		model_id = [file for file in out_sample_mses if out_sample_mses[file] == max_mse_red][0]
	return max_mse_red, model_id
		

def clear_model_cache(dataset):
	if dataset == None:
		# TODO: make this path more flexible
		shutil.rmtree("model_cache/")
		os.makedirs("model_cache/")
	else:
		if os.path.isdir(f"model_cache/{dataset}"):
			shutil.rmtree(f"model_cache/{dataset}")


def run_bayesian_regression(data, file):
    data = pd.read_csv(data)
    model = mb.parse_cxl(file)
    transformed_data = utils.transform_data(data, model).dropna().reset_index(drop=True)
    regression.run_bayesian_regression(transformed_data, model)