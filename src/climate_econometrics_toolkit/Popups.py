import tkinter as tk
from tkinter import filedialog

from climate_econometrics_toolkit import user_prediction_functions as user_predict


class StandardErrorPopup(tk.Toplevel):

	std_error_type = None

	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("500x300")
		popup.transient(window)

		def on_close():
			self.std_error_type = std_error_type.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		std_error_list = ["Nonrobust","White-Huber","Driscoll-Kraay","Newey-West","Time-clustered","Space-clustered"]

		std_error_type = tk.StringVar(value=std_error_list[0])

		radio_button_label = tk.Label(popup, text="Choose a type of standard error to be estimated with the regression:")
		radio_button_label.grid(row=0, column=0, padx=5, pady=1)
		for index, method in enumerate(std_error_list):
			button = tk.Radiobutton(popup, text=method, variable=std_error_type, value=method)
			button.grid(row=index+1, column=0, padx=5, pady=1)
		
		window.wait_window(popup)
		

class SpatialRegressionTypePopup(tk.Toplevel):

	reg_type = None
	geometry_column = None

	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("500x300")
		popup.transient(window)

		def on_close():
			self.reg_type = reg_type.get()
			self.geometry_column = geometry_column.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		reg_type_list = ["lag (implementation: spreg.GM_Lag)","error (implementation: spreg.GM_Error)"]

		reg_type = tk.StringVar(value=reg_type_list[0])
		
		text_entry_label = tk.Label(popup, text="Enter the column in your dataset containing the geometric coordinates of the panel variable.\nIf your dataset uses ISO3 country codes, no geometry column is required.")

		geometry_column = tk.StringVar()
		geometry_column_entry = tk.Entry(popup, textvariable=geometry_column)

		radio_button_label = tk.Label(popup, text="Choose a type of spatial regression model to run:")
		radio_button_label.grid(row=0, column=0, padx=5, pady=1)
		for index, method in enumerate(reg_type_list):
			button = tk.Radiobutton(popup, text=method, variable=reg_type_list, value=method)
			button.grid(row=index+1, column=0, padx=5, pady=1)
		text_entry_label.grid(row=3, column=0, padx=5, pady=1, columnspan=2)
		geometry_column_entry.grid(row=4, column=0, padx=5, pady=1, columnspan=2)
		
		window.wait_window(popup)


class PredictionFunctionPopup(tk.Toplevel):

	function = None

	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("500x300")
		popup.transient(window)

		def on_close():
			self.function = function.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		method_list = ["None"]
		method_list_from_user_predict = [method for method in dir(user_predict) if not method.startswith("__") and not method == "np" and not method == "pd"]
		method_list.extend(method_list_from_user_predict)

		function = tk.StringVar(value=method_list[0])

		radio_button_label = tk.Label(popup, text="Optionally choose a function to apply to the predictions.")
		radio_button_label.grid(row=0, column=0, padx=5, pady=1)
		for index, method in enumerate(method_list):
			button = tk.Radiobutton(popup, text=method, variable=function, value=method)
			button.grid(row=index+1, column=0, padx=5, pady=1)
		
		window.wait_window(popup)


class RasterExtractionPopup(tk.Toplevel):

	weight_file = None
	raster_file = None
	shape_file = None
	time_interval = None
	func = None

	def open_file(self, file, popup):
			if file == "raster":
				filepath = filedialog.askopenfilenames(
					initialdir = "/",
					title = "Select One or More Files",
					parent=popup
			  	)
				if filepath:
					self.raster_file = filepath
			else:
				filepath = filedialog.askopenfilename(
					initialdir = "/",
					title = "Select One File",
					parent=popup
			  	)
				if filepath:
					if file == "weights":
						self.weight_file = filepath
					elif file == "shapes":
						self.shape_file = filepath


	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("500x300")
		popup.transient(window)

		def on_close():
			self.time_interval = time_interval.get()
			self.func = function.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		raster_file_button = tk.Button(popup, text="Select One or More Raster File(s)", command=lambda : self.open_file("raster", popup))
		shape_file_button = tk.Button(popup, text="Select One Shape File", command=lambda : self.open_file("shapes", popup))
		weight_file_button = tk.Button(popup, text="Select One Weight File", command=lambda : self.open_file("weights", popup))

		function = tk.StringVar(value="Mean")
		mean_button = tk.Radiobutton(popup, text="Mean", variable=function, value="Mean")
		sum_button = tk.Radiobutton(popup, text="Sum", variable=function, value="Sum")

		time_interval = tk.StringVar()

		text_entry_label = tk.Label(popup, text="Enter the quantity of time periods to aggregate as an integer.\nFor example, if your raster data is monthly and\nyou want to aggregate to the yearly level, enter 12.")
		radio_button_label = tk.Label(popup, text="Select the aggregation function")

		time_entry = tk.Entry(popup, textvariable=time_interval)

		raster_file_button.grid(row=0, column=0, padx=5, pady=1, columnspan=2)
		shape_file_button.grid(row=1, column=0, padx=5, pady=1, columnspan=2)
		weight_file_button.grid(row=2, column=0, padx=5, pady=1, columnspan=2)

		text_entry_label.grid(row=3, column=0, padx=5, pady=1, columnspan=2)
		time_entry.grid(row=4, column=0, padx=5, pady=1, columnspan=2)

		radio_button_label.grid(row=5, column=0, padx=5, pady=1, columnspan=2)
		mean_button.grid(row=6, column=0, padx=5, pady=1)
		sum_button.grid(row=6, column=1, padx=5, pady=1)

		window.wait_window(popup)


class QuantileRegressionPopup(tk.Toplevel):

	quantiles = None

	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("500x300")
		popup.transient(window)

		def on_close():
			self.quantiles = quantiles.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)
		
		text_entry_label = tk.Label(popup, text="Enter quantiles to run (must be between 0 and 1).\nIf multiple quantiles, separate with commas: e.g. .1,.2,.3,etc.")

		quantiles = tk.StringVar()
		quantile_entry = tk.Entry(popup, textvariable=quantiles)

		text_entry_label.grid(row=1, column=0, padx=5, pady=1, columnspan=2)
		quantile_entry.grid(row=2, column=0, padx=5, pady=1, columnspan=2)
		
		window.wait_window(popup)