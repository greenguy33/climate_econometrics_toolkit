from climate_econometrics_toolkit import utils

import tkinter as tk
from tkinter import filedialog

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