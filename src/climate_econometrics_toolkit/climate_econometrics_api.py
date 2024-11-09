import pandas as pd
import shutil
import os
import tkinter as tk
import threading

import climate_econometrics_toolkit.evaluate_model as ce_eval
import climate_econometrics_toolkit.model_builder as mb
import climate_econometrics_toolkit.climate_econometrics_utils as utils
import climate_econometrics_toolkit.climate_econometrics_regression as regression

from climate_econometrics_toolkit.TkInterfaceUtils import TkInterfaceUtils
from climate_econometrics_toolkit.DragAndDropInterface import DragAndDropInterface
from climate_econometrics_toolkit.RegressionPlot import RegressionPlot
from climate_econometrics_toolkit.ResultPlot import ResultPlot
from climate_econometrics_toolkit.StatPlot import StatPlot

pd.set_option('display.min_rows', 100)
pd.set_option('display.max_rows', 100)

# TODO: improve python-facing API for users
# TODO: facilitate loading a model directly from the model cache into the API
# To do this you might need to add a "get model id" button in the interface which the user can copy/paste in their code
# TODO: refactor code into API and interface directories
def evaluate_model(data_file, model, panel_column, time_column):
	# model = mb.parse_cxl(model)
	return_string = ""
	model_id = None
	regression_result = None
	try:
		model, unused_nodes = mb.parse_model_input(model, data_file, panel_column, time_column)
		if len(unused_nodes) > 0:
			return_string += "\nWARNING: The following nodes are unused in the regression. " + str(unused_nodes)
		# TODO: find out why this sorting was causing nan's in the predictions
		data = pd.read_csv(data_file).sort_values([model.panel_column, model.time_column]).reset_index(drop=True)
		data.columns = data.columns.str.replace(' ', '_') 
		if len(set(data.columns)) != len(data.columns): 
			return_string += "\nTwo column names in dataset collide when spaces are removed. Please correct."
		else:
			model = ce_eval.evaluate_model(data, model)
			# return_string += "\n" + utils.compare_to_last_model(model, data_file)
			model_id = model.save_model_to_cache()
			regression_result = model.regression_result
	except BaseException as e:
		return_string += "\nERROR: " + str(e)
	print(regression_result.summary2().tables[1])
	return model_id, regression_result, return_string


def get_best_model_for_dataset(filename):
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
		shutil.rmtree("model_cache/")
		os.makedirs("model_cache/")
	else:
		if os.path.isdir(f"model_cache/{dataset}"):
			shutil.rmtree(f"model_cache/{dataset}")


def run_bayesian_regression(data_file, model_id, use_threading=False):
	data_file_short = data_file.split("/")[-1]
	model, panel_column, time_column = utils.construct_model_input_from_cache(data_file_short, model_id)
	model, _ = mb.parse_model_input(model, data_file, panel_column, time_column)
	data = pd.read_csv(data_file).sort_values([model.time_column, model.panel_column]).reset_index(drop=True)
	transformed_data = utils.transform_data(data, model).dropna().reset_index(drop=True)
	if use_threading:
		thread = threading.Thread(target=regression.run_bayesian_regression,name="bayes_sampling_thread",args=(transformed_data,model,model_id))
		thread.daemon = True
		thread.start()
	else:
		regression.run_bayesian_regression(transformed_data,model,model_id)


def start_interface():
	
	utils.initial_checks()
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
	regression_plot_frame = tk.Frame(window, relief=tk.RAISED, bd=2)
	result_plot_frame = tk.Frame(window, relief=tk.RAISED, bd=2)

	dnd = DragAndDropInterface(canvas, window)
	regression_plot = RegressionPlot(regression_plot_frame)
	result_plot = ResultPlot(result_plot_frame)

	lefthand_bar = tk.Frame(window, relief=tk.RAISED, bd=2)
	mse_canvas = tk.Canvas(lefthand_bar, width=100, height=200)
	pred_int_canvas = tk.Canvas(lefthand_bar, width=100, height=200)
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
	result_text = tk.Text(lefthand_bar, height=5)

	btn_load.grid(row=0, column=0, sticky="nsew", padx=5, pady=5, columnspan=2)
	btn_clear_canvas.grid(row=1, column=0, sticky="nsew", padx=5, columnspan=2)
	btn_evaluate.grid(row=2, column=0, sticky="nsew", padx=5, columnspan=2)
	btn_best_model.grid(row=3, column=0, sticky="nsew", padx=5, columnspan=2)
	btn_clear_model_cache.grid(row=4, column=0, sticky="nsew", padx=5, columnspan=2)
	btn_bayesian_regression.grid(row=5, column=0, sticky="nsew", columnspan=2)
	result_text.grid(row=6, column=0, sticky="nsew", columnspan=2)
	mse_canvas.grid(row=7, column=0, sticky="nsew")
	pred_int_canvas.grid(row=7, column=1, sticky="nsew")
	lefthand_bar.grid(row=0, column=0, sticky="ns")
	regression_plot_frame.grid(row=1, column=0, sticky="ns")
	canvas.grid(row=0, column=1, sticky="nsew")
	result_plot_frame.grid(row=1, column=1, sticky="nsew")

	dnd.canvas_print_out = result_text
	window.mainloop()
