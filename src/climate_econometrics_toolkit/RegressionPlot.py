import tkinter as tk

import scipy.stats as stats
import numpy as np
import pickle as pkl
import pandas as pd

import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

cet_home = os.getenv("CETHOME")

class RegressionPlot():

    def __init__(self, plot_frame):
        self.plot_frame = plot_frame
        self.plot_canvas = None

    def add_normal_distribution_to_axis(self, coef_name, reg_result, index, axis, num_plots):
        mean = float(reg_result["Coef."][index])
        sd = float(reg_result["Std.Err."][index])
        pval = reg_result.pvals[index]
        if not pd.isnull(pval):
            if pval < .00001:
                pval = "< .00001"
            elif pval < .0001:
                pval = "< .0001"
            elif pval < .001:
                pval = "< .001"
            elif pval < .01:
                pval = "< .01"
            elif pval < .05:
                pval = "< .05"
            else:
                pval = '%.2f' % pval
        else:
            pval = "NaN"
        axis.set_xlabel(f"P-value : {str(pval)}")
        x = np.linspace(mean - 3*sd, mean + 3*sd, 100)
        axis.plot(x, stats.norm.pdf(x, mean, sd))
        axis.set_title(coef_name)
        if num_plots < 4:
            x_axis_label_size = 8
        else:
            x_axis_label_size = 6
        axis.title.set_size(8)
        axis.get_yaxis().set_visible(False)
        axis.xaxis.set_tick_params(labelsize=x_axis_label_size)
        axis.set_xticks(np.linspace(min(x), max(x), 3))
        axis.set_xticklabels(['{:.2e}'.format(val) for val in np.linspace(min(x), max(x), 3)])
        return axis       

    def build_axes(self, reg_result):
        num_plots = len([
            val for val in reg_result.index if val != "const" 
            and val != "Intercept"
            and not val.startswith("fe_") 
            and not (val.startswith("tt") and val[3] == "_")
            and (reg_result["Std.Err."][val] != "")
            and (not pd.isnull(reg_result["Std.Err."][val]))
            and not (val.startswith("C("))
        ])
        if num_plots <= 4:
            fig, axes = plt.subplots(1,num_plots,figsize=(6,2))
        else:
            num_rows = int(num_plots/4)
            if num_plots % 4 != 0:
                num_rows += 1
            fig, axes = plt.subplots(num_rows,4,figsize=(6,5))
        axis_count = 0
        for index in range(len(reg_result.index)):
            coef_name = reg_result.index[index]
            if (
                coef_name != "const" 
                and coef_name != "Intercept"
                and not coef_name.startswith("fe_")
                and (not (coef_name.startswith("tt") and coef_name[3] == "_")) 
                and (reg_result["Std.Err."][index] != "")
                and (not pd.isnull(reg_result["Std.Err."][index]))
                and not (coef_name.startswith("C("))
            ):
                if num_plots == 1:
                    self.add_normal_distribution_to_axis(coef_name, reg_result, index, axes, num_plots)
                else:
                    if num_plots <= 4:
                        self.add_normal_distribution_to_axis(coef_name, reg_result, index, axes[axis_count], num_plots)
                    else:
                        col_num = int(axis_count/4)
                        self.add_normal_distribution_to_axis(coef_name, reg_result, index, axes[col_num][axis_count-(4*col_num)], num_plots)
                    axis_count += 1 
        return fig, axes

    def plot_new_regression_result(self, reg_result, model_type, dataset, cache_dir):

        if self.plot_canvas != None:
            self.plot_canvas.get_tk_widget().destroy()

        if model_type == "random":
            err_data = reg_result.bse.to_frame()
            mean_data = reg_result.params.to_frame()
            pvals = reg_result.pvalues.to_frame()
            reg_result = pd.concat([mean_data, err_data, pvals], axis=1)
            reg_result.columns = ["Coef.","Std.Err.","pvals"]
        elif model_type == "driscollkraay":
            reg_result = pd.concat([reg_result.params, reg_result.std_errors, reg_result.pvalues], axis=1)
            reg_result.columns = ["Coef.","Std.Err.","pvals"]
        elif "P>|t|" in reg_result:
            reg_result["pvals"] = reg_result["P>|t|"]
        else:
            reg_result["pvals"] = reg_result["P>|z|"]
        fig, axes = self.build_axes(reg_result)

        plt.tight_layout(h_pad = -.3)

        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        with open (f'{cet_home}/model_cache/{dataset}/{cache_dir}/regression_plot.pkl', 'wb') as buff:
            pkl.dump({"axes":axes,"fig":fig},buff)

    def clear_figure(self):
        if self.plot_canvas != None:
            self.plot_canvas.get_tk_widget().destroy()

    def restore_regression_result(self, dataset, cache_dir):

        if self.plot_canvas != None:
            self.clear_figure()

        cached_plot = pd.read_pickle(f'{cet_home}/model_cache/{dataset}/{cache_dir}/regression_plot.pkl')
        fig = cached_plot["fig"]

        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)