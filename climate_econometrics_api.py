import pandas as pd
import shutil
import os

import climate_econometrics_toolkit.evaluate_model as ce_eval
import climate_econometrics_toolkit.model_builder as mb
import climate_econometrics_toolkit.climate_econometrics_utils as utils
import climate_econometrics_toolkit.climate_econometrics_regression as regression


def evaluate_model(data_file, model_file):
	model = mb.parse_cxl(model_file)
	data = pd.read_csv(data_file)
	model = ce_eval.evaluate_model(data, model)
	utils.compare_to_last_model(model)
	model.save_model_to_cache()


def get_best_model():
	# TODO: make this path more flexible
	out_sample_mses = {}
	cache_files = os.listdir("model_cache/")
	if len(cache_files) == 0:
		return None
	for file in cache_files:
		latest_model = pd.read_csv(f"model_cache/{file}/model.csv")
		out_sample_mses[file] = float(latest_model["attribute_value"][latest_model['model_attribute']=='out_sample_mse'].values[0])
	min_mse = min(out_sample_mses.values())
	file = [file for file in out_sample_mses if out_sample_mses[file] == min_mse][0]
	return pd.read_csv(f"model_cache/{file}/model.csv")
			

def clear_model_cache():
	# TODO: make this path more flexible
	shutil.rmtree("model_cache/")
	os.makedirs("model_cache/")


def run_bayesian_regression(data, file):
    data = pd.read_csv(data)
    model = mb.parse_cxl(file)
    transformed_data = utils.transform_data(data, model).dropna().reset_index(drop=True)
    regression.run_bayesian_regression(transformed_data, model)