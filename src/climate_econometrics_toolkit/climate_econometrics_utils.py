import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
import pyfixest as pf
import dateutil.parser as parser
import copy
import ast
import tkinter as tk

from climate_econometrics_toolkit.TkInterfaceUtils import TkInterfaceUtils
from climate_econometrics_toolkit.DragAndDropInterface import DragAndDropInterface
from climate_econometrics_toolkit.RegressionPlot import RegressionPlot
from climate_econometrics_toolkit.ResultPlot import ResultPlot
from climate_econometrics_toolkit.StatPlot import StatPlot

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


def demean_fixed_effects(data, model):
	data.to_csv("data_to_demean.csv")
	fixed_effects = []
	for fe in model.fixed_effects:
		if not np.issubdtype(data[fe].dtype, np.number):
			enc = OrdinalEncoder()
			ordered_list = list(dict.fromkeys(data[fe]))
			enc.fit(np.array(ordered_list).reshape(-1,1))
			data[f"encoded_{fe}"] = [int(val) for val in enc.transform(np.array(data[fe]).reshape(-1,1))]
			fixed_effects.append(f"encoded_{fe}")
		else:
			fixed_effects.append(fe)
	vars_to_demean = copy.deepcopy(model.model_vars)
	# vars_to_demean.extend(time_trend_columns)
	centered_data = pf.estimation.demean(
		np.array(data[vars_to_demean]), 
		np.array(data[fixed_effects]), 
		np.ones(len(data))
	)[0]
	centered_data = pd.DataFrame(centered_data, columns=vars_to_demean)
	for column in [col for col in data.columns if not col.startswith("fe_")]:
		if column not in centered_data:
			centered_data = pd.concat([centered_data, data[column]], axis=1).reset_index(drop=True)
	return centered_data


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
	# Note: removing nan's before demeaning fixed effects may slightly impact the results compared to other statistical packages.
	# This is done because the demeaning package does not handle NaNs.
	data = remove_nan_rows(data, model.covariates + model.fixed_effects + [model.target_var])
	if not demean:
		for fe in model.fixed_effects:
			data = add_fixed_effect_to_data(fe, data)
	else:
		if len(model.fixed_effects) > 0:
			data = demean_fixed_effects(data, model)
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

def start_user_interface():
	initial_checks()
	window = tk.Tk()
	window.title("Climate Econometrics Modeling Toolkit")

	window.rowconfigure(0, minsize=100, weight=1)
	window.rowconfigure(1, minsize=100, weight=1)
	window.columnconfigure(1, minsize=800, weight=1)

	canvas = tk.Canvas(
		window, 
		width=800, 
		height=800, 
		highlightthickness=5,
		highlightbackground="black",
		highlightcolor="red"
	)

	lefthand_bar = tk.Frame(window, relief=tk.RAISED, bd=2)

	regression_plot_frame = tk.Frame(lefthand_bar, relief=tk.RAISED, bd=2)
	result_plot_frame = tk.Frame(window, relief=tk.RAISED, bd=2)

	dnd = DragAndDropInterface(canvas, window)
	regression_plot = RegressionPlot(regression_plot_frame)
	result_plot = ResultPlot(result_plot_frame)

	mse_canvas = tk.Canvas(lefthand_bar, width=100, height=75)
	pred_int_canvas = tk.Canvas(lefthand_bar, width=100, height=75)
	stat_plot = StatPlot(mse_canvas, pred_int_canvas)
	tk_utils = TkInterfaceUtils(window, canvas, dnd, regression_plot, result_plot, stat_plot)
	window.protocol("WM_DELETE_WINDOW", tk_utils.on_close)

	# TODO: add frame for showing/changing the panel and time columns
	btn_load = tk.Button(lefthand_bar, text="Load Dataset", command=tk_utils.add_data_columns_from_file)
	btn_clear_canvas = tk.Button(lefthand_bar, text="Clear Canvas", command=tk_utils.clear_canvas)
	btn_evaluate = tk.Button(lefthand_bar, text="Evaluate Model", command=tk_utils.evaluate_model)
	btn_best_model = tk.Button(lefthand_bar, text="Restore Best Model", command=tk_utils.restore_best_model)
	btn_clear_model_cache = tk.Button(lefthand_bar, text="Clear Model Cache", command=tk_utils.clear_model_cache)
	btn_bayesian_regression = tk.Button(lefthand_bar, text="Run Bayesian Inference", command=tk_utils.run_bayesian_inference)
	result_text = tk.Text(lefthand_bar, height=2)

	btn_load.grid(row=0, column=0, sticky="nsew", padx=5, pady=5, columnspan=2)
	btn_clear_canvas.grid(row=1, column=0, sticky="nsew", padx=5, columnspan=2)
	btn_evaluate.grid(row=2, column=0, sticky="nsew", padx=5, columnspan=2)
	btn_best_model.grid(row=3, column=0, sticky="nsew", padx=5, columnspan=2)
	btn_clear_model_cache.grid(row=4, column=0, sticky="nsew", padx=5, columnspan=2)
	btn_bayesian_regression.grid(row=5, column=0, sticky="nsew", columnspan=2)
	result_text.grid(row=6, column=0, sticky="nsew", columnspan=2)
	mse_canvas.grid(row=7, column=0, sticky="nsew")
	pred_int_canvas.grid(row=7, column=1, sticky="nsew")
	regression_plot_frame.grid(row=8, column=0, sticky="ns", columnspan=2)
	lefthand_bar.grid(row=0, column=0, sticky="ns", rowspan=2)
	canvas.grid(row=0, column=1, sticky="nsew")
	result_plot_frame.grid(row=1, column=1, sticky="nsew")

	dnd.canvas_print_out = result_text
	window.mainloop()


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