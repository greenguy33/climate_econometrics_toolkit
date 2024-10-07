import numpy as np
from statsmodels.tsa.statespace.tools import diff
import pandas as pd
import statsmodels.api as sm
import os

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

supported_functions = ["fd","sq","ln"]
supported_effects = ["fe", "ie"]


def add_transformation_to_data(data, function):
	if function[0:2] == "sq":
		data[function] = np.square(data[function[3:-1]])
	elif function[0:2] == "fd":
		data[function] = diff(data[function[3:-1]])
	elif function[0:2] == "ln":
		data[function] = np.log(data[function[3:-1]])
	return data


def add_fixed_effect_to_data(node, data):
	for element in sorted(list(set(data[node])))[1:]:
		data[f"fe_{element}_{node}"] = np.where(data[node] == element, 1, 0)
	return data


def add_incremental_effects_to_data(node, data):
	ie_level = 1
	node_split = node.split(" ")
	if len(node_split) > 1:
		ie_level = int(node_split[1].strip())
	for element in sorted(list(set(data[node_split[0]]))):
		data[f"ie_{element}_{node_split[0]}_1"] = np.where(data[node_split[0]] == element, 1, 0)
		data[f"ie_{element}_{node_split[0]}_1"] = np.where(data[node_split[0]] == element, data[f"ie_{element}_{node_split[0]}_1"].cumsum(), 0)
		for i in range(1, ie_level+1):
			data[f"ie_{element}_{node_split[0]}_{i}"] = np.power(data[f"ie_{element}_{node_split[0]}_1"], i)
	return data


def transform_data(data, model):
	transformations = []
	for node in model.model_vars:
		if node[0:2] not in supported_functions and node[0:2] not in supported_effects:
			assert node in data, f"Element {node} not found in data"
		elif node[0:2] in supported_functions:
			function_split = node.split("(")
			data_node = function_split[-1].replace(")","")
			assert data_node in data, f"Element {data_node} not found in data"
			for function in reversed(function_split[:-1]):
				assert function in supported_functions, f"Invalid function call {function}"
				transformations.append(function + f"({data_node})")
				data = add_transformation_to_data(data, transformations[-1])
				data_node = transformations[-1]
	for fe in model.fixed_effects:
		data = add_fixed_effect_to_data(fe, data)
	for ie in model.incremental_effects:
		data = add_incremental_effects_to_data(ie, data)
	return data


def get_model_vars(data, model, demeaned):
	model_vars = [var for var in model.covariates]
	if demeaned:
		# exclude fixed effects from demeaned data
		for effect_col in [col for col in data if any(col.startswith(val) for val in supported_effects) and not col.startswith("fe")]:
			model_vars.append(effect_col)
	else:
		for effect_col in [col for col in data if any(col.startswith(val) for val in supported_effects)]:
			model_vars.append(effect_col)
	return model_vars


def get_last_model_out_sample_mse():
	# TODO: make this path more flexible
	cache_files = os.listdir("model_cache/")
	if len(cache_files) == 0:
		return None
	latest_model_dir = max([float(file) for file in cache_files])
	latest_model = pd.read_csv(f"model_cache/{latest_model_dir}/model.csv")
	return float(latest_model["attribute_value"][latest_model['model_attribute']=='out_sample_mse'].values[0])


def compare_to_last_model(model):
	last_model_osmse = get_last_model_out_sample_mse()
	if last_model_osmse == None:
		return(f"This model has out-of-sample MSE of {str(model.out_sample_mse)[:7]}. There is no model in the cache to compare to this model.")
	elif last_model_osmse < model.out_sample_mse:
		return(f"This model has HIGHER OUT-OF-SAMPLE MSE {str(model.out_sample_mse)[:7]} than the last model {str(last_model_osmse)[:7]}")
	elif last_model_osmse > model.out_sample_mse:
		return(f"This model has LOWER OUT-OF-SAMPLE MSE {str(model.out_sample_mse)[:7]} than the last model {str(last_model_osmse)[:7]}")
	elif last_model_osmse == model.out_sample_mse:
		return(f"This model has THE SAME OUT-OF-SAMPLE MSE {str(model.out_sample_mse)[:7]} as the last model {str(last_model_osmse)[:7]}")