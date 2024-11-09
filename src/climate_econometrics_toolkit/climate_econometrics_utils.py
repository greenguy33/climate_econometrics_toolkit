import numpy as np
from statsmodels.tsa.tsatools import add_lag
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
# import pyfixest as pf
import dateutil.parser as parser
import copy
import ast

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

supported_functions = ["fd","sq","cu","ln","lag1","lag2","lag3"]
supported_effects = ["fe", "tt1", "tt2", "tt3"]
# TODO: understand how changing this can lead to undesirable results (e.g. in the Burke model)
random_state = 123


def initial_checks():
	if not os.path.isdir("model_cache"):
		os.makedirs("model_cache")
	if not os.path.isdir("bayes_samples"):
		os.makedirs("bayes_samples")


def add_transformation_to_data(data, model, function):
	function_split = function.split("(")
	data_col = "(".join(function_split[1:])[:-1]
	if function_split[0] == "sq":
		data[function] = np.square(data[data_col])
	elif function_split[0] == "fd":
		# TODO: this won't work if the data isn't sorted by year or if there are missing year values
		# add something like this for missing year values:
		# data["T5_mean_diff"] = data.groupby("ID")["T5_mean"].diff()
		# t5_mean_diff = []
		# last_year = 0
		# last_region = ""
		# last_row = None
		# for row in data.itertuples():
		# 	this_year = row.yearn
		# 	this_region = row.ID
		# 	t5_mean_diff.append(row.T5_mean_diff)
		# 	if this_year - last_year > 1 and row.ID == last_region:
		# 		print(last_row.ID, last_row.year, last_row.country, last_row.T5_mean, last_row.T5_mean_diff)
		# 		print(row.ID, row.year, row.country, row.T5_mean, row.T5_mean_diff)
		# 		t5_mean_diff[-1] = np.NaN
		# 	last_year = this_year
		# 	last_region = this_region
		# 	last_row = row
		data[function] = data.groupby(model.panel_column)[data_col].diff()
	elif function_split[0] == "ln":
		data[function] = np.log(data[data_col])
	elif function_split[0].startswith("lag"):
		# TODO: this won't work if the data isn't sorted by year or if there are missing year values
		num_lags = int(function_split[0][3])
		data[function] = data.groupby(model.panel_column)[data_col].shift(num_lags)
	elif function_split[0] == "cu":
		data[function] = np.power(data[data_col], 3)
	return data


def add_fixed_effect_to_data(node, data):
	for element in sorted(list(set(data[node])))[1:]:
		data[f"fe_{element}_{node}"] = np.where(data[node] == element, 1, 0)
	return data


def add_time_trends_to_data(node, data, time_column):
	# TODO: This only supports time effects by year. Add support for monthly/weekly
	parsed_dates = [parser.parse(str(date)) for date in data[time_column]]
	min_year = min(parsed_dates[index].year for index in range(len(parsed_dates)))
	ie_level = 1
	node_split = node.split(" ")
	if len(node_split) > 1:
		ie_level = int(node_split[1].strip())
	for element in sorted(list(set(data[node_split[0]]))):
		data[f"tt1_{element}_{node_split[0]}"] = np.where(data[node_split[0]] == element, data[time_column] - min_year, 0)
		for i in range(1, ie_level+1):
			data[f"tt{i}_{element}_{node_split[0]}"] = np.power(data[f"tt1_{element}_{node_split[0]}"], i)
	return data


def is_inf(data):
	try:
		return np.isinf(data)
	except TypeError:
		return False


def remove_nan_rows(data, no_nan_cols):
	missing_indices = []
	for index, row in enumerate(data.iterrows()):
		if any(pd.isna(row[1][col]) or is_inf(row[1][col]) for col in no_nan_cols):
			missing_indices.append(index)
	data = data.drop(missing_indices).reset_index(drop=True)
	return data


# def demean_fixed_effects(data, model):
# 	fixed_effects = []
# 	for fe in model.fixed_effects:
# 		if not np.issubdtype(data[fe].dtype, np.number):
# 			enc = OrdinalEncoder()
# 			ordered_list = list(dict.fromkeys(data[fe]))
# 			enc.fit(np.array(ordered_list).reshape(-1,1))
# 			data[f"encoded_{fe}"] = [int(val) for val in enc.transform(np.array(data[fe]).reshape(-1,1))]
# 			fixed_effects.append(f"encoded_{fe}")
# 		else:
# 			fixed_effects.append(fe)
# 	vars_to_demean = copy.deepcopy(model.model_vars)
# 	vars_to_demean.extend([col for col in data.columns if col.startswith("tt_")])
# 	centered_data = pf.estimation.demean(
# 		np.array(data[vars_to_demean]), 
# 		np.array(data[fixed_effects]), 
# 		np.ones(len(data))
# 	)[0]
# 	centered_data = pd.DataFrame(centered_data, columns=vars_to_demean)
# 	for fe in model.fixed_effects:
# 		centered_data = pd.concat([data[fe], centered_data], axis=1).reset_index(drop=True)
# 	return centered_data


def transform_data(data, model, demean=False):
	transformations = []
	for node in model.model_vars:
		function_split = node.split("(")
		if function_split[0] not in supported_functions and function_split[0] not in supported_effects:
			assert node in data, f"Element {node} not found in data"
		elif function_split[0] in supported_functions:
			data_node = function_split[-1].replace(")","")
			assert data_node in data, f"Element {data_node} not found in data"
			for function in reversed(function_split[:-1]):
				assert function in supported_functions, f"Invalid function call {function}"
				transformations.append(function + f"({data_node})")
				data = add_transformation_to_data(data, model, transformations[-1])
				data_node = transformations[-1]
	for ie in model.time_trends:
		data = add_time_trends_to_data(ie, data, model.time_column)
	data = remove_nan_rows(data, model.covariates + model.fixed_effects + [model.target_var])
	# if not demean:
	for fe in model.fixed_effects:
		data = add_fixed_effect_to_data(fe, data)
	# else:
	# 	if len(model.fixed_effects) > 0:
	# 		data = demean_fixed_effects(data, model)
	return data


def get_model_vars(data, model, demeaned=False):
	model_vars = [var for var in model.covariates]
	if demeaned:
		# exclude fixed effects from demeaned data
		for effect_col in [col for col in data if any(col.startswith(val) for val in supported_effects) and not col.startswith("fe_")]:
			model_vars.append(effect_col)
	else:
		for effect_col in [col for col in data if any(col.startswith(val) for val in supported_effects)]:
			model_vars.append(effect_col)
	return model_vars


def get_attribute_from_model_file(dataset, attribute, model_id):
	model = pd.read_csv(f"model_cache/{dataset}/{model_id}/model.csv")
	return model["attribute_value"][model['model_attribute']==attribute].values[0]


def get_last_model_out_sample_mse(data_file):
	if not os.path.isdir(f"model_cache/{data_file}"):
		return None
	dataset_cache_files = [float(file) for file in os.listdir(f"model_cache/{data_file}")]
	if len(dataset_cache_files) == 0:
		return None
	return float(get_attribute_from_model_file(data_file, "out_sample_mse_reduction", max(dataset_cache_files)))


def construct_model_input_from_cache(data_file, model_id):
	target_var = get_attribute_from_model_file(data_file, "target_var", model_id)
	covariates = get_attribute_from_model_file(data_file, "covariates", model_id)
	covariate_list = ast.literal_eval(covariates)
	panel_column = get_attribute_from_model_file(data_file, "panel_column", model_id)
	time_column = get_attribute_from_model_file(data_file, "time_column", model_id)
	fixed_effects = get_attribute_from_model_file(data_file, "fixed_effects", model_id)
	fixed_effect_list = [f"fe({val})" for val in ast.literal_eval(fixed_effects)]
	time_trends = ast.literal_eval(get_attribute_from_model_file(data_file, "time_trends", model_id))
	time_trend_list = []
	for tt in time_trends:
		tt_split = tt.split(" ")
		time_trend_list.append(f"tt{tt_split[1]}({tt_split[0]})")
	covariate_list.extend(fixed_effect_list)
	covariate_list.extend(time_trend_list)
	target_var_list = [target_var] * len(covariate_list)
	return [covariate_list, target_var_list], panel_column, time_column


# def compare_to_last_model(model, data_file):
# 	last_model_osmse = get_last_model_out_sample_mse(data_file)
# 	if last_model_osmse == None:
# 		return(f"This model has out-of-sample MSE of {str(model.out_sample_mse_reduction)[:7]}. There is no model in the cache to compare to this model.")
# 	elif last_model_osmse < model.out_sample_mse_reduction:
# 		return(f"This model has HIGHER OUT-OF-SAMPLE MSE {str(model.out_sample_mse_reduction)[:7]} than the last model {str(last_model_osmse)[:7]}")
# 	elif last_model_osmse > model.out_sample_mse_reduction:
# 		return(f"This model has LOWER OUT-OF-SAMPLE MSE {str(model.out_sample_mse_reduction)[:7]} than the last model {str(last_model_osmse)[:7]}")
# 	elif last_model_osmse == model.out_sample_mse_reduction:
# 		return(f"This model has THE SAME OUT-OF-SAMPLE MSE {str(model.out_sample_mse_reduction)[:7]} as the last model {str(last_model_osmse)[:7]}")