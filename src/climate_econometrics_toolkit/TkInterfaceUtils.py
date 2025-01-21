import pandas as pd
import os

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.transforms as transform

import climate_econometrics_toolkit.interface_api as api
from climate_econometrics_toolkit.RasterExtractionPopup import RasterExtractionPopup
from climate_econometrics_toolkit.PredictionFunctionPopup import PredictionFunctionPopup

import xarray as xr
import geopandas as gpd
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
		axis.plot(self.result_plot.plot_data, marker='o', color='r', zorder=1)
		for index, point in enumerate(self.result_plot.plot_data):
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
			# TODO: Improve the text displayed
			model, regression_result, print_string = api.evaluate_model(self.dnd.filename, self.build_model_indices_lists(), self.panel_column, self.time_column)
			self.update_interface_window_output(print_string)
			if model != None:
				self.update_interface_window_output(
					f"Model results saved to {cet_home}/model_results/{model.model_id}.csv\nRegression script saved to {cet_home}/regression_scripts/{model.model_id}.csv"
				)
				self.dnd.save_canvas_to_cache(str(model.model_id), self.panel_column, self.time_column)
				self.regression_plot.plot_new_regression_result(regression_result.summary2().tables[1], self.dnd.data_source, model.model_id)
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


	def run_bayesian_inference(self):
		if self.dnd.current_model is None:
			self.update_interface_window_output("Please evaluate your model or select an existing model before running Bayesian inference.")
		else:
			self.update_interface_window_output("Bayesian inference will run in background...see command line for progress. Output will be available in {cet_home}/bayes_samples")
			api.run_bayesian_regression(self.dnd.current_model)


	def run_block_bootstrap(self):
		if self.dnd.current_model is None:
			self.update_interface_window_output("Please evaluate your model or select an existing model before running bootstrapping.")
		else:
			self.update_interface_window_output("Bootstrapping will run in background...see command line for progress. Output will be available in {cet_home}/bootstrap_samples")
			api.run_block_bootstrap(self.dnd.current_model)


	def extract_raster_data(self, window):
		raster_extract_popup = RasterExtractionPopup(window)
		raster_files = raster_extract_popup.raster_file
		shape_file = raster_extract_popup.shape_file

		if raster_files is None or shape_file is None:
			self.update_interface_window_output("Both a raster file and a shape file must be selected.")
		else:
			time_interval = int(raster_extract_popup.time_interval)
			weights_file = raster_extract_popup.weight_file
			aggregation_func = raster_extract_popup.func
			self.update_interface_window_output(f"Raster aggregation will run in background. When complete file will be saved to {cet_home}/raster_output. Check command line for errors.")
			thread = threading.Thread(target=self.raster_aggregation,name="bootstrap_thread",args=(raster_files, shape_file, aggregation_func, weights_file, time_interval))
			thread.daemon = True
			thread.start()


	def integrate_raster_datasets(self, raster_datasets, geo_id):
		# remove values of panel and time variables that aren't shared between all datasets
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


	def raster_aggregation(self, raster_files, shape_file, aggregation_func, weights_file, time_interval):
		raster_datasets = []
		# TODO: for multiple rasters, since time index is set to 0 for all files, this will erroneously align time spans even if they are not aligned in original files
		for raster_file in raster_files:
			raster = xr.open_dataset(raster_file)
			geo_identifier = gpd.read_file(shape_file).columns[0]
			climate_var_name = list(raster.data_vars)[-1]
			out = api.extract_raster_data(raster_file, shape_file, weights_file)
			raster_datasets.append(api.aggregate_raster_data(out, shape_file, climate_var_name, aggregation_func.lower(), time_interval, geo_identifier))
		if len(raster_files) == 1:
			raster_file_short = raster_file.split("/")[-1].rpartition('.')[0]
			raster_datasets[0].to_csv(f"{cet_home}/raster_output/{raster_file_short}.csv")
		else:
			self.integrate_raster_datasets(raster_datasets, geo_identifier)


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


	def clear_canvas(self):
		self.dnd.clear_canvas()
		self.regression_plot.clear_figure()
		self.result_plot.clear_figure()
		self.stat_plot.clear_stat_plot()
		self.panel_column = None
		self.time_column = None
		self.hover_label_text.set("")


	def clear_model_cache(self):
		api.clear_model_cache(self.dnd.data_source)
		self.result_plot.clear_figure()
		self.update_interface_window_output("Model cache cleared")
		self.hover_label_text.set("")


	def on_close(self):
		self.window.quit()
		self.window.destroy()


	def update_interface_window_output(self, output_text):
		self.dnd.canvas_print_out.delete(1.0, tk.END)
		self.dnd.canvas_print_out.insert(tk.END, output_text)
