import tkinter as tk
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import climate_econometrics_toolkit.climate_econometrics_utils as utils

class ResultPlot():

    def __init__(self, plot_frame):
        self.plot_frame = plot_frame
        self.plot_canvas = None
        self.plot_data = []

    def clear_figure(self):
        if self.plot_canvas != None:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_data = []
        
    def create_figure(self):
        fig, axis = plt.subplots(1)
        axis.plot(self.plot_data, marker='o', color='r')
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_result_plot(self, dataset):
        if os.path.isdir(f"model_cache/{dataset}"):
            self.clear_figure()
            sorted_cache_files = sorted({val:float(val) for val in os.listdir(f"model_cache/{dataset}")}.items(), key=lambda item: item[1])
            for cache_file in sorted_cache_files:
                out_sample_mse = float(utils.get_attribute_from_model_file(dataset, "out_sample_mse_reduction", str(cache_file[0])))
                self.plot_data.append(out_sample_mse)
            self.create_figure()