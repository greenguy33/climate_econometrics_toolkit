import numpy as np
import pandas as pd
import os
import time
import logging
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pyfixest as pf
import dateutil.parser as parser
import copy
import ast
import tkinter as tk
from tkinter import ttk
from importlib.resources import files

from climate_econometrics_toolkit.TkInterfaceUtils import TkInterfaceUtils
from climate_econometrics_toolkit.DragAndDropInterface import DragAndDropInterface
from climate_econometrics_toolkit.RegressionPlot import RegressionPlot
from climate_econometrics_toolkit.ResultPlot import ResultPlot
from climate_econometrics_toolkit.StatPlot import StatPlot

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

cet_home = os.getenv("CETHOME")
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"{cet_home}/logs/{time.strftime('%m%d%y')}.log", level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

supported_functions = ["fd","sq","cu","ln","lag1","lag2","lag3","scale"]
supported_effects = ["fe", "tt1", "tt2", "tt3","re"]
# TODO: consider adjusted r2, as this accounts for different numbers of variables?
# last line of https://www.nature.com/articles/s43016-024-01040-8#Sec8
supported_metrics = ["out_sample_mse_reduction","out_sample_mse","out_sample_pred_int_cov","rmse","r2"]
supported_gcms = ["BCC-CSM2-MR","CanESM5","CNRM-CM6-1","HadGEM3-GC31-LL","IPSL-CM6A-LR","MIROC6","MRI-ESM2-0"]
supported_standard_errors = ["nonrobust", "whitehuber", "driscollkraay", "neweywest", "clusteredtime","clusteredspace"]
supported_weights = ["unweighted","popweighted","cropweighted","maizeweighted","riceweighted","soybeanweighted","wheatweighted"]
std_type_string = ",".join(supported_standard_errors)

std_error_name_map = {
	"nonrobust":"nonrobust",
	"whitehuber":"HC1",
	"neweywest":"HAC",
	"clusteredtime":"cluster",
	"clusteredspace":"cluster",
	"driscollkraay":"kernel"
}

std_error_args = {
	"neweywest":{"maxlags":3},
	"clusteredtime":{"groups":""},
	"clusteredspace":{"groups":""}
}
quantile_std_error_map = {
	"nonrobust":"iid",
	"greene":"robust"
}
spatial_std_error_map = {
	"nonrobust":None,
	"whitehuber":"white",
	"neweywest":"hac"
}

# TODO: understand how changing this can lead to undesirable results (e.g. in the Burke model)
random_state = 123

def initial_checks():
	if cet_home is None:
		print_with_log("Environmental Variable 'CETHOME' is not set. Defaulting to current directory as working directory.")
		os.environ["CETHOME"] = "."
	dirs_to_init = [
		"logs",
		"model_cache",
		"bayes_samples",
		"bootstrap_samples",
		"raster_output",
		"predictions",
		"OLS_output",
		"regression_scripts",
		"spatial_regression_output",
		"quantile_regression_output",
		"statistical_tests_output",
		"statistical_tests_output/panel_unit_root_tests/",
		"statistical_tests_output/cointegration_tests/",
		"statistical_tests_output/cross_sectional_dependence_tests/",
		"html",
		"prediction_intervals",
		"resampled_raster_files"
	]
	for dir in dirs_to_init:
		if not os.path.isdir(f"{cet_home}/{dir}"):
			os.makedirs(f"{cet_home}/{dir}")


def assert_with_log(clause, message):
	try:
		assert clause, message
	except AssertionError:
		logger.error(message)
		raise


def get_growing_season_data_by_crop(crop):
	assert_with_log(crop is None or crop in ["maize","rice","soybeans","wheat.spring","wheat.winter"], "Specified crop must be one of: 'maize','rice','soybeans','wheat.spring','wheat.winter'.")
	gs_file = files("climate_econometrics_toolkit.preprocessed_data.crop_growing_seasons").joinpath(f'{crop}.csv')
	growing_season_data = pd.read_csv(gs_file)
	country_start_days = dict(zip(growing_season_data["ISO3"],growing_season_data[f"day.{crop}.plant.start"]))
	country_end_days = dict(zip(growing_season_data["ISO3"],growing_season_data[f"day.{crop}.harvest.end"]))
	return country_start_days, country_end_days


def print_with_log(message, level="info"):
	assert level in ["warning","info","error"], "Invalid log-level passed. This is an internal application error."
	print(f"{level.upper()}: {message}")
	if level == "info":
		logger.info(message)
	elif level == "warning":
		logger.warning(message)
	elif level == "error":
		logger.error(message)


def add_transformation_to_data(data, model, function):
	# TODO: add interaction transformations
	function_split = function.split("(")
	data_col = "(".join(function_split[1:])[:-1]
	if function_split[0] == "sq":
		data[function] = np.square(data[data_col])
	elif function_split[0] == "fd":
		# TODO: this won't work if the data isn't sorted by year or if there are missing year values
		# TODO: this also breaks if there are multiple panel column/time column observations, like in the harveststat data
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
		# TODO: this also breaks if there are multiple panel column/time column observations, like in the harveststat data
		num_lags = int(function_split[0][3])
		data[function] = data.groupby(model.panel_column)[data_col].shift(num_lags)
	elif function_split[0] == "cu":
		data[function] = np.power(data[data_col], 3)
	elif function_split[0] == "scale":
		scaler = StandardScaler()
		data[function] = scaler.fit_transform(np.array(data[data_col]).reshape(-1,1)).flatten()
	return data


def add_dummy_variable_to_data(node, data, leave_out_first=True):
	if leave_out_first:
		for element in sorted(list(set(data[node])))[1:]:
			data[f"fe_{element}_{node}"] = np.where(data[node] == element, 1, 0)
	else:
		for element in sorted(list(set(data[node]))):
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


def demean_fixed_effects(data, model):
	fixed_effects = []
	for fe in model.fixed_effects:
		if not pd.api.types.is_integer_dtype(data[fe].dtype):
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


def transform_data(data, model, include_target_var=True, demean=False):
	data = copy.deepcopy(data)
	transformations = []
	vars_to_transform = copy.deepcopy(model.model_vars)
	if not include_target_var:
		vars_to_transform = copy.deepcopy(model.covariates)
	if model.random_effects is not None:
		vars_to_transform.append(model.random_effects[0])
	for node in vars_to_transform:
		if not node.startswith("re("):
			function_split = node.split("(")
			if function_split[0] not in supported_functions and function_split[0] not in supported_effects:
				assert_with_log(node in data, f"Element {node} not found in data")
			elif function_split[0] in supported_functions:
				data_node = function_split[-1].replace(")","")
				assert_with_log(data_node in data, f"Element {data_node} not found in data")
				for function in reversed(function_split[:-1]):
					assert_with_log(function in supported_functions, f"Invalid function call {function}")
					transformations.append(function + f"({data_node})")
					data = add_transformation_to_data(data, model, transformations[-1])
					data_node = transformations[-1]
	for ie in model.time_trends:
		data = add_time_trends_to_data(ie, data, model.time_column)
	# Note: removing nan's before demeaning fixed effects may slightly impact the results compared to other statistical packages.
	# This is done because the demeaning package does not handle NaNs.
	vars_to_include = model.covariates + model.fixed_effects
	if include_target_var:
		vars_to_include = vars_to_include + [model.target_var]
	if model.random_effects is not None:
		vars_to_include.append(model.random_effects[0])
	data = data.dropna(subset=vars_to_include).reset_index(drop=True)
	if not demean:
		for fe in model.fixed_effects:
			data = add_dummy_variable_to_data(fe, data)
	else:
		if len(model.fixed_effects) > 0:
			data = demean_fixed_effects(data, model)
	return data


def get_model_vars(data, model, exclude_fixed_effects=True):
	model_vars = [var for var in model.covariates]
	if exclude_fixed_effects:
		for effect_col in [col for col in data if any(col.startswith(val) for val in supported_effects) and not col.startswith("fe_")]:
			model_vars.append(effect_col)
	else:
		for effect_col in [col for col in data if any(col.startswith(val) for val in supported_effects)]:
			model_vars.append(effect_col)
	return model_vars


def construct_model_input_from_cache(data_file, model_id):
	model = pd.read_pickle(f"{cet_home}/model_cache/{data_file}/{model_id}/model.pkl")
	covariate_list = ast.literal_eval(model.covariates)
	fixed_effect_list = [f"fe({val})" for val in ast.literal_eval(model.fixed_effects)]
	time_trends = ast.literal_eval(model.time_trends)
	time_trend_list = []
	for tt in time_trends:
		tt_split = tt.split(" ")
		time_trend_list.append(f"tt{tt_split[1]}({tt_split[0]})")
	covariate_list.extend(fixed_effect_list)
	covariate_list.extend(time_trend_list)
	target_var_list = [model.target_var] * len(covariate_list)
	return [covariate_list, target_var_list], model.panel_column, model.time_column


def start_user_interface():
	initial_checks()
	logger.info('Started Interface')
	
	root = tk.Tk()
	root.title("Climate Econometrics Modeling Toolkit")

	window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
	window.pack(fill=tk.BOTH, expand=True)

	lefthand_bar = ttk.Frame(window, relief=tk.RAISED)
	window.add(lefthand_bar, weight=3)

	canvas_window = ttk.PanedWindow(window, orient=tk.VERTICAL)
	canvas_frame = ttk.Frame(canvas_window)
	canvas = tk.Canvas(
		canvas_frame, 
		highlightthickness=5,
		highlightbackground="black",
		highlightcolor="red"
	)
	window.add(canvas_window, weight=3)
	canvas_window.add(canvas_frame, weight=7)

	regression_plot_frame = ttk.Frame(lefthand_bar, relief=tk.RAISED)
	result_plot_frame = ttk.Frame(canvas_window, relief=tk.RAISED)
	canvas_window.add(result_plot_frame, weight=3)

	dnd = DragAndDropInterface(canvas, window)
	regression_plot = RegressionPlot(regression_plot_frame)
	result_plot = ResultPlot(result_plot_frame)

	metrics_row1 = ttk.Frame(lefthand_bar, relief=tk.RAISED)
	mse_canvas = tk.Canvas(metrics_row1, width=100, height=100)
	pred_int_canvas = tk.Canvas(metrics_row1, width=100, height=100)

	metrics_row2 = ttk.Frame(lefthand_bar, relief=tk.RAISED)
	r2_canvas = tk.Canvas(metrics_row2, width=100, height=100)
	rmse_canvas = tk.Canvas(metrics_row2, width=100, height=100)

	uncertainty_button_row = ttk.Frame(lefthand_bar, relief=tk.RAISED)
	evaluation_button_row = ttk.Frame(lefthand_bar, relief=tk.RAISED)

	stat_plot = StatPlot(mse_canvas, pred_int_canvas, r2_canvas, rmse_canvas)
	tk_utils = TkInterfaceUtils(window, canvas, dnd, regression_plot, result_plot, result_plot_frame, stat_plot)

	root.protocol("WM_DELETE_WINDOW", tk_utils.on_close)

	# TODO: add frame for showing/changing the panel and time columns

	step1_label = tk.Label(lefthand_bar, text="Step 1: Climate Data Aggregation", font=('Helvetica', 12, 'bold'))
	btn_extract = tk.Button(lefthand_bar, text="Extract Raster Data", command=lambda : tk_utils.extract_raster_data(window))
	step2_label = tk.Label(lefthand_bar, text="Step 2: Construct and Evaluate Model", font=('Helvetica', 12, 'bold'))
	btn_load = tk.Button(lefthand_bar, text="Load Dataset", command=tk_utils.add_data_columns_from_file)
	btn_clear_canvas = tk.Button(lefthand_bar, text="Clear Canvas", command=tk_utils.clear_canvas)
	btn_clear_model_cache = tk.Button(lefthand_bar, text="Clear Model Cache", command=tk_utils.clear_model_cache)
	btn_unit_root = tk.Button(evaluation_button_row, wraplength=120, text="Run ADF Panel Unit Root Tests", command=tk_utils.run_panel_unit_root_tests)
	btn_cointegration = tk.Button(evaluation_button_row, wraplength=120, text="Run Engel-Granger Cointegration Tests", command=tk_utils.run_cointegration_tests)
	btn_csd = tk.Button(evaluation_button_row, wraplength=120, text="Run Pesaran Cross-Sectional Dependence Tests", command=tk_utils.run_csd_tests)
	btn_evaluate = tk.Button(evaluation_button_row, wraplength=120, text="Evaluate Model with OLS", command=tk_utils.evaluate_model)
	btn_bootstrap = tk.Button(uncertainty_button_row, text="Run Block Bootstrap", command=tk_utils.run_block_bootstrap)
	btn_bayesian_regression = tk.Button(uncertainty_button_row, text="Run Bayesian Inference", command=tk_utils.run_bayesian_inference)
	btn_spatial_regression = tk.Button(uncertainty_button_row, text="Run Spatial Regression", command=tk_utils.run_spatial_regression)
	btn_quantile_regression = tk.Button(uncertainty_button_row, text="Run Quantile Regression", command=tk_utils.run_quantile_regression)
	step3_label = tk.Label(lefthand_bar, text="Step 3: Predict Impacts", font=('Helvetica', 12, 'bold'))
	btn_predict = tk.Button(lefthand_bar, text="Predict Out-of-Sample", command=tk_utils.predict_out_of_sample)
	result_text = tk.Text(lefthand_bar, height=2)

	step1_label.pack(fill=tk.BOTH, expand=True)
	btn_extract.pack(fill=tk.BOTH, expand=True)

	step2_label.pack(fill=tk.BOTH, expand=True)
	btn_load.pack(fill=tk.BOTH, expand=True)
	btn_clear_canvas.pack(fill=tk.BOTH, expand=True)
	btn_clear_model_cache.pack(fill=tk.BOTH, expand=True)

	evaluation_button_row.pack(fill=tk.BOTH, expand=True)
	btn_unit_root.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	btn_cointegration.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	btn_csd.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	btn_evaluate.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

	uncertainty_button_row.pack(fill=tk.BOTH, expand=True)
	btn_bootstrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	btn_bayesian_regression.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	btn_spatial_regression.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	btn_quantile_regression.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

	step3_label.pack(fill=tk.BOTH, expand=True)
	btn_predict.pack(fill=tk.BOTH, expand=True)

	result_text.pack(fill=tk.BOTH, expand=True)

	metrics_row1.pack(fill=tk.X)
	mse_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	r2_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	metrics_row2.pack(fill=tk.X)
	pred_int_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	rmse_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

	regression_plot_frame.pack(fill=tk.BOTH, expand=True)

	canvas.pack(fill=tk.BOTH, expand=True)

	dnd.canvas_print_out = result_text

	window.mainloop()
