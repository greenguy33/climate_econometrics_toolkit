import tkinter as tk

import geopandas as gpd
from tkinter import filedialog
from inspect import getmembers, isfunction

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
	k = None

	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("600x200")
		popup.transient(window)

		def on_close():
			self.reg_type = reg_type.get()
			self.k = k.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		reg_type_list = ["lag (implementation: spreg.Panel_FE_Lag)","error (implementation: spreg.Panel_FE_Error)"]
		reg_type = tk.StringVar(value=reg_type_list[0])

		k_label = tk.Label(popup, text="Enter an integer for k in k-nearest-neighbors spatial weight matrix construction:")
		k = tk.StringVar(value="5")
		k_column_entry = tk.Entry(popup, textvariable=k)

		radio_button_label = tk.Label(popup, text="Choose a type of spatial regression model to run:")
		radio_button_label.grid(row=0, column=0, padx=5, pady=1)
		for index, method in enumerate(reg_type_list):
			button = tk.Radiobutton(popup, text=method, variable=reg_type, value=method)
			button.grid(row=index+1, column=0, padx=5, pady=1)

		k_label.grid(row=5, column=0, padx=5, pady=1, columnspan=2)
		k_column_entry.grid(row=6, column=0, padx=5, pady=1, columnspan=2)
		
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
		method_list_from_user_predict = [val[0] for val in (getmembers(user_predict, isfunction))]
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
	starting_year = None
	crop = None

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
		popup.geometry("500x500")
		popup.transient(window)

		def on_close():
			self.time_interval = time_interval.get()
			self.func = function.get()
			self.starting_year = starting_year.get()
			self.crop = crop.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		raster_file_button = tk.Button(popup, text="Select One or More Raster File(s)", command=lambda : self.open_file("raster", popup))
		shape_file_button = tk.Button(popup, text="Select One Shape File", command=lambda : self.open_file("shapes", popup))
		weight_file_button = tk.Button(popup, text="Select One Weight File (optional)", command=lambda : self.open_file("weights", popup))

		function = tk.StringVar(value="Mean")
		mean_button = tk.Radiobutton(popup, text="Mean", variable=function, value="Mean")
		sum_button = tk.Radiobutton(popup, text="Sum", variable=function, value="Sum")

		time_interval = tk.StringVar(value="Daily")
		daily_button = tk.Radiobutton(popup, text="Daily", variable=time_interval, value=365)
		monthly_button = tk.Radiobutton(popup, text="Monthly", variable=time_interval, value=12)

		crop = tk.StringVar(value="Daily")
		none_button = tk.Radiobutton(popup, text="None", variable=crop, value=None)
		wheat_winter_button = tk.Radiobutton(popup, text="Wheat (winter season)", variable=crop, value="wheat.winter")
		wheat_spring_button = tk.Radiobutton(popup, text="Wheat (spring season)", variable=crop, value="wheat.spring")
		soybeans_button = tk.Radiobutton(popup, text="Soybeans", variable=crop, value="soybeans")
		maize_button = tk.Radiobutton(popup, text="Maize", variable=crop, value="maize")
		rice_button = tk.Radiobutton(popup, text="Rice", variable=crop, value="rice")

		starting_year = tk.StringVar()

		time_period_radio_button_label = tk.Label(popup, text="Select the time interval of your raster data.\nOnly daily and monthly to yearly aggregation is currently supported.")
		year_entry_label = tk.Label(popup, text="Enter the first year for which data is present in the raster data.")
		function_radio_button_label = tk.Label(popup, text="Select the aggregation function")
		crop_radio_button_label = tk.Label(popup, text="Select the crop growing season to filter the raster data.")

		year_entry = tk.Entry(popup, textvariable=starting_year)

		raster_file_button.grid(row=0, column=0, padx=5, pady=1, columnspan=2)
		shape_file_button.grid(row=1, column=0, padx=5, pady=1, columnspan=2)
		weight_file_button.grid(row=2, column=0, padx=5, pady=1, columnspan=2)

		time_period_radio_button_label.grid(row=3, column=0, padx=5, pady=1, columnspan=2)
		daily_button.grid(row=4, column=0, padx=5, pady=1)
		monthly_button.grid(row=4, column=1, padx=5, pady=1)
		
		year_entry_label.grid(row=5, column=0, padx=5, pady=1, columnspan=2)
		year_entry.grid(row=6, column=1, padx=5, pady=1, columnspan=2)

		function_radio_button_label.grid(row=7, column=0, padx=5, pady=1, columnspan=2)
		mean_button.grid(row=8, column=0, padx=5, pady=1)
		sum_button.grid(row=8, column=1, padx=5, pady=1)

		crop_radio_button_label.grid(row=9, column=0, padx=5, pady=1, columnspan=2)
		none_button.grid(row=10, column=0, padx=5, pady=1)
		wheat_winter_button.grid(row=10, column=1, padx=5, pady=1)
		wheat_spring_button.grid(row=11, column=0, padx=5, pady=1)
		soybeans_button.grid(row=11, column=1, padx=5, pady=1)
		maize_button.grid(row=12, column=0, padx=5, pady=1)
		rice_button.grid(row=12, column=1, padx=5, pady=1)

		window.wait_window(popup)

class GeoIdentifierSelectionPopup(tk.Toplevel):

	geo_identifier = None

	def __init__(self, window, shape_file):

		popup = tk.Toplevel()
		popup.geometry("500x500")
		popup.transient(window)

		def on_close():
			self.geo_identifier = geo_identifier.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		geo_identifier = tk.StringVar(value=None)

		geo_id_radio_button_label = tk.Label(popup, text="Select the geo-identifier to use from your selected shapefile.\nIf you specified a crop growing season filter, select\nan ISO3 or GMI code.")
		
		shape_file_columns = gpd.read_file(shape_file).columns
		buttons = []
		for column in shape_file_columns:
			buttons.append(tk.Radiobutton(popup, text=column, variable=geo_identifier, value=column))

		geo_id_radio_button_label.grid(row=0, column=0, padx=5, pady=1)
		for index, button in enumerate(buttons):
			button.grid(row=index+1, column=0, padx=5, pady=1)

		window.wait_window(popup)


class QuantileRegressionPopup(tk.Toplevel):

	quantiles = None
	std_error_type = None

	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("500x300")
		popup.transient(window)

		def on_close():
			self.quantiles = quantiles.get()
			self.std_error_type = std_error_type.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)
		
		text_entry_label = tk.Label(popup, text="Enter quantiles to run (must be between 0 and 1).\nIf multiple quantiles, separate with commas: e.g. .1,.2,.3,etc.")

		quantiles = tk.StringVar()
		quantile_entry = tk.Entry(popup, textvariable=quantiles)

		text_entry_label.grid(row=1, column=0, padx=5, pady=1, columnspan=2)
		quantile_entry.grid(row=2, column=0, padx=5, pady=1, columnspan=2)

		std_error_type_list = ["nonrobust","greene"]

		std_error_label = tk.Label(popup, text="Select a type of standard error to use:")
		std_error_type = tk.StringVar(value=std_error_type_list[0])

		std_error_label.grid(row=3, column=0, padx=5, pady=1, columnspan=2)
		for index, method in enumerate(std_error_type_list):
			button = tk.Radiobutton(popup, text=method, variable=std_error_type, value=method)
			button.grid(row=index+4, column=0, padx=5, pady=1)
		
		window.wait_window(popup)


class ExportDataFormatPopup(tk.Toplevel):

	export_format = None

	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("500x300")
		popup.transient(window)

		def on_close():
			self.export_format = export_format.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		text_entry_label = tk.Label(popup, text="Select an export format:")
		format_list = ["CSV", "Stata (dta)", "Rdata"]
		export_format = tk.StringVar(value=format_list[0])
		
		text_entry_label.grid(row=0, column=0, padx=5, pady=1, columnspan=len(format_list))

		for index, format in enumerate(format_list):
			button = tk.Radiobutton(popup, text=format, variable=export_format, value=format)
			button.grid(row=1, column=index, padx=5, pady=1)
		
		window.wait_window(popup)
