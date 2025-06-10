import pandas as pd
import os
import numpy as np
import time
import logging

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.transforms as transform

import climate_econometrics_toolkit.interface_api as api
import climate_econometrics_toolkit.utils as utils
from climate_econometrics_toolkit.Popups import *

import xarray as xr
import threading

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

cet_home = os.getenv("CETHOME")

class TkInterfaceUtils():

	def __init__(self, window, canvas, dnd, regression_plot, result_plot, result_plot_frame, stat_plot):
		self.window = window
		self.canvas = canvas
		self.dnd = dnd
		self.regression_plot = regression_plot
		self.result_plot = result_plot
		self.result_plot_frame = result_plot_frame
		self.stat_plot = stat_plot
		self.panel_column = None
		self.time_column = None
		self.dataset = None

		# setup hover label on result plot
		self.hover_label_text = tk.StringVar()
		self.hover_label = tk.Label(self.result_plot_frame, textvariable=self.hover_label_text)
		self.hover_label.pack()
		self.hover_label.bind("<ButtonPress-1>", self.model_id_to_clipboard)


	def model_id_to_clipboard(self, event):
		df = pd.DataFrame([self.hover_label_text.get().split(" ")[2]])
		df.to_clipboard(index=False, header=False, sep=None, excel=False)
		self.hover_label_text.set("Copied!")


	def add_data_columns_from_file(self):

		if self.dnd.variables_displayed:
			self.update_interface_window_output("Please clear the canvas before loading another dataset.")
		else:
			filename = filedialog.askopenfilename(
				initialdir = "/",
				title = "Select a File",
				filetypes = (("CSV files",
							"*.csv*"),
							("all files",
							"*.*"))
			  )
			# filename = "data/GDP_climate_test_data.csv"

			self.dnd.data_source = filename.split("/")[-1]
			self.dnd.filename = filename
			data = pd.read_csv(filename)
			self.dataset = data
			columns = data.columns
			if len(columns) > 100:
				self.update_interface_window_output("ERROR: This dataset exceeds the maximum number of columns(100)")
			else:
				self.dnd.add_model_variables(columns)
				user_identified_columns = self.update_result_plot(self.dnd.data_source, "r2")
				if user_identified_columns == None:
					while self.time_column not in data:
						self.time_column = simpledialog.askstring(title="get_time_col", prompt="Provide the name of the time-based column:")
					while self.panel_column not in data:
						self.panel_column = simpledialog.askstring(title="get_panel_col", prompt="Provide the name of the panel column:")
				else:
					self.panel_column = user_identified_columns[0]
					self.time_column = user_identified_columns[1]
				self.dnd.panel_column = self.panel_column
				self.dnd.time_column = self.time_column
			utils.print_with_log(f"Loaded data from file {filename}", "info")


	def build_model_indices_lists(self):
		from_indices,to_indices = [],[]
		for element_id in self.canvas.find_all():
			element_tags = self.canvas.gettags(element_id)
			if self.dnd.tags_are_arrow(element_tags):
				from_indices.append(element_tags[0].split("boxed_text_")[1])
				to_indices.append(element_tags[1].split("boxed_text_")[1])
		return [from_indices, to_indices]
	

	def handle_click_on_result_plot(self, event):
		for index, circle in enumerate(self.result_plot.circles):
			if circle.contains_points([[event.x, event.y]]):
				self.restore_model(self.result_plot.models[index])
				break


	def handle_hover_on_result_plot(self, event):
		for index, circle in enumerate(self.result_plot.circles):
			if circle.contains_point((event.x, event.y)):
				self.hover_label_text.set("Model ID: " + self.result_plot.models[index] + " (click to copy to clipboard)")
				break


	def create_result_plot(self, metric):
		fig, axis = plt.subplots(1)
		axis.set_title(metric)
		axis.set_ylabel(metric + " value")
		xvals, yvals = [], []
		for index, val in enumerate(self.result_plot.plot_data):
			if not np.isnan(val):
				xvals.append(index)
				yvals.append(val)
		axis.plot(xvals, yvals, marker='o', color='r', zorder=1)
		for index, point in enumerate(self.result_plot.plot_data):
			if not np.isnan(point):
				circle = plt.Circle((0,0), 0.05, color='b', transform=(fig.dpi_scale_trans + transform.ScaledTranslation(index, point, axis.transData)), zorder=2)
				axis.add_patch(circle)
				self.result_plot.circles.append(circle)
		self.result_plot.plot_canvas = FigureCanvasTkAgg(fig, master=self.result_plot.plot_frame)
		self.result_plot.plot_canvas.draw()
		self.result_plot.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
		self.result_plot.plot_canvas.mpl_connect('button_press_event', self.handle_click_on_result_plot)
		self.result_plot.plot_canvas.mpl_connect('motion_notify_event', self.handle_hover_on_result_plot)


	def update_result_plot(self, dataset, metric):
		if os.path.isdir(f"{cet_home}/model_cache/{dataset}"):
			self.result_plot.clear_figure()
			sorted_cache_files = sorted({val:float(val) for val in os.listdir(f"{cet_home}/model_cache/{dataset}")}.items(), key=lambda item: item[1])
			for cache_file in sorted_cache_files:
				if os.path.exists(f"{cet_home}/model_cache/{dataset}/{cache_file[0]}/tkinter_canvas.pkl"):
					model = pd.read_pickle(f"{cet_home}/model_cache/{dataset}/{str(cache_file[0])}/model.pkl")
					self.result_plot.plot_data.append(getattr(model, metric))
					self.result_plot.models.append(cache_file[0])
			self.create_result_plot(metric)
			model = pd.read_pickle(f"{cet_home}/model_cache/{dataset}/{cache_file[0]}/model.pkl")
			return model.panel_column, model.time_column


	def get_regression_stats_from_model(self ,model_id):
		model = pd.read_pickle(f"{cet_home}/model_cache/{self.dnd.data_source}/{model_id}/model.pkl")
		return model.out_sample_mse_reduction, model.out_sample_pred_int_cov, model.r2, model.rmse
	

	def bind_stat_canvases_to_result_plot(self, mse_canvas, pred_int_canvas, r2_canvas, rmse_canvas):
		mse_canvas.bind("<ButtonPress-1>", lambda _, data=self.dnd.data_source, metric="out_sample_mse_reduction" : self.update_result_plot(data, metric))
		pred_int_canvas.bind("<ButtonPress-1>", lambda _, data=self.dnd.data_source, metric="out_sample_pred_int_cov" : self.update_result_plot(data, metric))
		r2_canvas.bind("<ButtonPress-1>", lambda _, data=self.dnd.data_source, metric="r2" : self.update_result_plot(data, metric))
		rmse_canvas.bind("<ButtonPress-1>", lambda _, data=self.dnd.data_source, metric="rmse": self.update_result_plot(data, metric))


	def evaluate_model(self):
		if self.dnd.variables_displayed:
			standard_error_popup = StandardErrorPopup(self.window)
			model, regression_result, print_string = api.evaluate_model(self.dnd.filename, standard_error_popup.std_error_type, self.build_model_indices_lists(), self.panel_column, self.time_column)
			self.update_interface_window_output(print_string)
			if model != None:
				utils.print_with_log(f"Evaluating Model with ID {model.model_id}", "info")
				self.update_interface_window_output(f"Model results saved to {cet_home}/OLS_output/{model.model_id}.csv. Regression script saved to {cet_home}/regression_scripts/{model.model_id}.csv")
				self.dnd.save_canvas_to_cache(str(model.model_id), self.panel_column, self.time_column)
				if model.random_effects is None:
					try:
						self.regression_plot.plot_new_regression_result(regression_result.summary2().tables[1], "nonrandom", self.dnd.data_source, model.model_id)
					except AttributeError:
						self.regression_plot.plot_new_regression_result(regression_result, "driscollkraay", self.dnd.data_source, model.model_id)
				else:
					self.regression_plot.plot_new_regression_result(regression_result, "random", self.dnd.data_source, model.model_id)
				self.update_result_plot(self.dnd.data_source, "r2")
				canvases = self.stat_plot.update_stat_plot(*self.get_regression_stats_from_model(model.model_id))
				self.bind_stat_canvases_to_result_plot(*canvases)
			self.dnd.current_model = model
			return model
		else:
			self.update_interface_window_output("Please load a dataset and create a model before evaluating model.")


	def restore_model(self, model_id):
		self.dnd.restore_canvas_from_cache(str(model_id))
		self.regression_plot.restore_regression_result(self.dnd.data_source, str(model_id))
		canvases = self.stat_plot.update_stat_plot(*self.get_regression_stats_from_model(model_id))
		self.bind_stat_canvases_to_result_plot(*canvases)
		utils.print_with_log(f"Model with ID {model_id} restored from cache.", "info")


	def run_bayesian_inference(self):
		if self.dnd.current_model is None:
			model = api.build_model_object_from_canvas(self.build_model_indices_lists(), self.dnd.filename, self.panel_column, self.time_column)[0]
			model.model_id = time.time()
		else:
			model = self.dnd.current_model
		utils.print_with_log(f"Running Bayesian Inference against Model with ID {model.model_id}", "info")
		self.update_interface_window_output(f"Bayesian inference will run in background...see command line for progress. Output will be available in {cet_home}/bayes_samples/{model.model_id}")
		api.run_bayesian_regression(model)


	def run_block_bootstrap(self):
		if self.dnd.current_model is None:
			model = api.build_model_object_from_canvas(self.build_model_indices_lists(), self.dnd.filename, self.panel_column, self.time_column)[0]
			model.model_id = time.time()
		else:
			model = self.dnd.current_model
		standard_error_popup = StandardErrorPopup(self.window)
		utils.print_with_log(f"Running Bootstrapping against Model with ID {model.model_id}", "info")
		self.update_interface_window_output(f"Bootstrapping will run in background...see command line for progress. Output will be available in {cet_home}/bootstrap_samples/coefficient_samples_{model.model_id}.csv")
		api.run_block_bootstrap(model, standard_error_popup.std_error_type)


	def run_spatial_regression(self):
		if self.dnd.current_model is None:
			model = api.build_model_object_from_canvas(self.build_model_indices_lists(), self.dnd.filename, self.panel_column, self.time_column)[0]
			model_id = time.time()
		else:
			model = self.dnd.current_model
			model_id = model.model_id
		spatial_regression_type_popup = SpatialRegressionTypePopup(self.window)
		utils.print_with_log(f"Running Spatial Regression against Model with ID {model_id}", "info")
		reg_type = spatial_regression_type_popup.reg_type.split(" ")[0]
		if spatial_regression_type_popup.k == "":
			k = 5
		else:
			k = int(spatial_regression_type_popup.k)
		api.run_spatial_regression(
			model, 
			reg_type, 
			model_id, 
			k		
		)
		self.update_interface_window_output(f"Spatial regression output is available in {cet_home}/spatial_regression_output/{model_id}")


	def run_quantile_regression(self):
		if self.dnd.current_model is None:
			model = api.build_model_object_from_canvas(self.build_model_indices_lists(), self.dnd.filename, self.panel_column, self.time_column)[0]
			model_id = time.time()
		else:
			model = self.dnd.current_model
			model_id = model.model_id
		quantile_popup = QuantileRegressionPopup(self.window)
		utils.print_with_log(f"Running Quantile Regression against Model with ID {model_id}", "info")
		quantiles = quantile_popup.quantiles.strip()
		if "," in quantiles:
			if quantiles[-1] == ",":
				quantiles = quantiles[:-1]
			quantiles = [float(val) for val in quantiles.split(",")]
		else:
			quantiles = float(quantiles)
		std_error_type = quantile_popup.std_error_type
		api.run_quantile_regression(model, model_id, quantiles, std_error_type)
		self.update_interface_window_output(f"Quantile regression output is available in {cet_home}/quantile_regression_output/{model_id}")


	def run_panel_unit_root_tests(self):
		model = api.build_model_object_from_canvas(self.build_model_indices_lists(), self.dnd.filename, self.panel_column, self.time_column)[0]
		model_id = time.time()
		utils.print_with_log(f"Running Panel Unit Root Tests against Model with ID {model_id}", "info")
		model.dataset = self.dataset
		api.run_panel_unit_root_tests(model, model_id)
		self.update_interface_window_output(f"Panel Unit Root test output is available in {cet_home}/statistical_tests_output/panel_unit_root_tests/{model_id}.csv")


	def run_cointegration_tests(self):
		model = api.build_model_object_from_canvas(self.build_model_indices_lists(), self.dnd.filename, self.panel_column, self.time_column)[0]
		model_id = time.time()
		utils.print_with_log(f"Running Cointegration Tests against Model with ID {model_id}", "info")
		model.dataset = self.dataset
		api.run_cointegration_tests(model, model_id)
		self.update_interface_window_output(f"Cointegration test output is available in {cet_home}/statistical_tests_output/cointegration_tests/{model_id}.csv")

	def run_csd_tests(self):
		model = api.build_model_object_from_canvas(self.build_model_indices_lists(), self.dnd.filename, self.panel_column, self.time_column)[0]
		model_id = time.time()
		utils.print_with_log(f"Running Cross-Sectional Dependence Tests against Model with ID {model_id}", "info")
		model.dataset = self.dataset
		api.run_cross_sectional_dependence_tests(model, model_id)
		self.update_interface_window_output(f"Cross-Sectional Dependence test output is available in {cet_home}/statistical_tests_output/cross_sectional_dependence_tests/{model_id}.csv")


	def extract_raster_data(self, window):
		raster_extract_popup = RasterExtractionPopup(window)
		raster_files = raster_extract_popup.raster_file
		shape_file = raster_extract_popup.shape_file

		if raster_files is None or shape_file is None:
			self.update_interface_window_output("Both a raster file and a shape file must be selected.")
		else:
			subperiods_per_year = int(raster_extract_popup.time_interval)
			starting_year = int(raster_extract_popup.starting_year)
			weights_file = raster_extract_popup.weight_file
			aggregation_func = raster_extract_popup.func
			crop = raster_extract_popup.crop
			if crop == "":
				crop = None
			geoid_popup = GeoIdentifierSelectionPopup(window, shape_file)
			geo_identifier = geoid_popup.geo_identifier
			utils.print_with_log(f"Extracting raster data using function {aggregation_func}; raster_file(s) {raster_files}; shape_file {shape_file}; weights file {weights_file}; geo-identifier {geo_identifier}; crop filter {crop}.", "info")
			self.update_interface_window_output(f"Raster aggregation will run in background. When complete file will be saved to {cet_home}/raster_output. Check command line for errors.")
			thread = threading.Thread(target=self.raster_aggregation,name="raster_agg_thread",args=(raster_files, shape_file, aggregation_func, weights_file, subperiods_per_year, starting_year, crop, geo_identifier))
			thread.daemon = True
			thread.start()


	def integrate_raster_datasets(self, raster_datasets, geo_id):
		# remove values of panel and time variables that aren't shared between all datasets
		# TODO: replace with merge function from pandas
		common_time_vals = set()
		common_geo_vals = set()
		for dataset in raster_datasets:
			if len(common_time_vals) == 0:
				common_time_vals = set(dataset["time"])
			else:
				for time_val in common_time_vals:
					if time_val not in set(dataset["time"]):
						common_time_vals.remove(time_val)
			if len(common_geo_vals) == 0:
				common_geo_vals = set(dataset[geo_id])
			else:
				for geo_val in common_geo_vals:
					if geo_val not in set(dataset[geo_id]):
						common_geo_vals.remove(geo_val)
		for dataset in raster_datasets:
			dataset = dataset[dataset["time"].isin(common_time_vals)]
			dataset = dataset[dataset[geo_id].isin(common_geo_vals)]
		df = pd.DataFrame()
		df[geo_id] = raster_datasets[0][geo_id]
		df["time"] = raster_datasets[0]["time"]
		for dataset in raster_datasets:
			df[dataset.columns[2]] = dataset[dataset.columns[2]]
		df.to_csv(f"{cet_home}/raster_output/integrated_dataset_with_{len(raster_datasets)}_input_files.csv")


	def raster_aggregation(self, raster_files, shape_file, aggregation_func, weights_file, subperiods_per_year, starting_year, crop, geo_identifier):
		raster_datasets = []
		# TODO: subperiods per year can accept list that corresponds with each raster file?
		for raster_file in raster_files:
			raster = xr.open_dataset(raster_file)
			climate_var_name = list(raster.data_vars)[-1]
			out = api.extract_raster_data(raster_file, shape_file, weights_file)
			raster_datasets.append(api.aggregate_raster_data(out, shape_file, climate_var_name, aggregation_func.lower(), geo_identifier, subperiods_per_year, starting_year, crop))
		if len(raster_files) == 1:
			raster_file_short = raster_file.split("/")[-1].rpartition('.')[0]
			raster_datasets[0].to_csv(f"{cet_home}/raster_output/{raster_file_short}.csv")
		else:
			self.integrate_raster_datasets(raster_datasets, geo_identifier)
		utils.print_with_log(f"Raster aggregation completed; output is available in {cet_home}/raster_output/{raster_file_short}.csv", "info")


	def predict_out_of_sample(self):
		if self.dnd.current_model is None:
			self.update_interface_window_output(f"Please evaluate your model or select an existing model before running prediction.")
		else:
			out_sample_data_files = filedialog.askopenfilenames(
				initialdir = "/",
				title = "Select One or More File(s) with Data to Predict",
				filetypes = (("CSV files",
							"*.csv*"),
							("all files",
							"*.*"))
			)
			if len(out_sample_data_files) > 0:
				prediction_function_popup = PredictionFunctionPopup(self.window)
				function = prediction_function_popup.function
				self.update_interface_window_output(f"Prediction will run in background...see command line for progress. Output will be available in {cet_home}/predictions")
				api.predict_out_of_sample(self.dnd.current_model, out_sample_data_files, function)
				utils.print_with_log("Out-of-sample predictions generated for Model with ID {self.dnd.current_model.model_id} for data file(s) {out_sample_data_files}", "info")


	def export_data(self):
		format = ExportDataFormatPopup(self.window).export_format
		model = api.build_model_object_from_canvas(self.build_model_indices_lists(), self.dnd.filename, self.panel_column, self.time_column)[0]
		api.export_data(self.dataset, model, format)
	
	
	def clear_canvas(self):
		self.dnd.clear_canvas()
		self.regression_plot.clear_figure()
		self.result_plot.clear_figure()
		self.stat_plot.clear_stat_plot()
		self.panel_column = None
		self.time_column = None
		self.hover_label_text.set("")
		utils.print_with_log("Canvas cleared.", "info")


	def clear_model_cache(self):
		api.clear_model_cache(self.dnd.data_source)
		self.result_plot.clear_figure()
		self.update_interface_window_output("Model cache cleared")
		self.hover_label_text.set("")
		utils.print_with_log("Model cache cleared.", "info")


	def update_interface_window_output(self, output_text):
		utils.print_with_log(output_text, "info")
		self.dnd.canvas_print_out.delete(1.0, tk.END)
		self.dnd.canvas_print_out.insert(tk.END, output_text)
