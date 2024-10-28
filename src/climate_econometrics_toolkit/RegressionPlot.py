import tkinter as tk

import scipy.stats as stats
import numpy as np
import pickle as pkl
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class RegressionPlot():

    def __init__(self, plot_frame):
        self.plot_frame = plot_frame
        self.plot_canvas = None

    def add_normal_distribution_to_axis(self, coef_name, reg_result, index, axis):
    
        mean = reg_result["Coef."][index]
        sd = reg_result["Std.Err."][index]
        x = np.linspace(mean - 3*sd, mean + 3*sd, 100)
        axis.plot(x, stats.norm.pdf(x, mean, sd))
        axis.set_title(coef_name)
        return axis       

    def build_axes(self, reg_result):
        num_plots = len([val for val in reg_result.index if val != "const" and not val.startswith("fe_") and not val.startswith("tt_")])
        fig, axes = plt.subplots(1,num_plots,figsize=(5,5))
        axis_count = 0
        for index in range(len(reg_result.index)):
            coef_name = reg_result.index[index]
            if coef_name != "const" and not coef_name.startswith("fe_") and not coef_name.startswith("tt_"):
                if num_plots == 1:
                    axis = self.add_normal_distribution_to_axis(coef_name, reg_result, index, axes)
                else:
                    axis = self.add_normal_distribution_to_axis(coef_name, reg_result, index, axes[axis_count])
                    axis_count += 1 
        return fig, axes

    def plot_new_regression_result(self, reg_result, dataset, cache_dir):

        if self.plot_canvas != None:
            self.plot_canvas.get_tk_widget().destroy()

        fig, axes = self.build_axes(reg_result)

        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        with open (f'model_cache/{dataset}/{cache_dir}/regression_plot.pkl', 'wb') as buff:
            pkl.dump({"axes":axes,"fig":fig},buff)

    def clear_figure(self):
        if self.plot_canvas != None:
            self.plot_canvas.get_tk_widget().destroy()

    def restore_regression_result(self, dataset, cache_dir):

        if self.plot_canvas != None:
            self.clear_figure()

        cached_plot = pd.read_pickle(f'model_cache/{dataset}/{cache_dir}/regression_plot.pkl')
        fig = cached_plot["fig"]

        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)