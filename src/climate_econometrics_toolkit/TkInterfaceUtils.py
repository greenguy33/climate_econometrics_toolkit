import pandas as pd
import os

import tkinter as tk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import climate_econometrics_toolkit.climate_econometrics_api as api
import climate_econometrics_toolkit.climate_econometrics_utils as utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class TkInterfaceUtils():

    def __init__(self, window, canvas, dnd, regression_plot, result_plot):
        self.window = window
        self.canvas = canvas
        self.dnd = dnd
        self.regression_plot = regression_plot
        self.result_plot = result_plot

    def add_data_columns_from_file(self):

        if self.dnd.variables_displayed:
            self.dnd.canvas_print_out.insert(tk.END, "\nPlease clear the canvas before loading another dataset.")
        else:
        #     filename = filedialog.askopenfilename(
        #         initialdir = "/",
        #         title = "Select a File",
        #         filetypes = (("CSV files",
        #                     "*.csv*"),
        #                     ("all files",
        #                     "*.*"))
        #        )
            filename = "/home/hayden-freedman/climate_econometrics_toolkit/data/GrowthClimateDataset.csv"

            self.dnd.data_source = filename.split("/")[-1]
            self.dnd.filename = filename
            data = pd.read_csv(filename)
            columns = data.columns
            self.dnd.add_model_variables(columns)

            self.update_result_plot(self.dnd.data_source)

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
        
    def create_result_plot(self):
        fig, axis = plt.subplots(1)
        axis.plot(self.result_plot.plot_data, marker='o', color='r')
        for index, point in enumerate(self.result_plot.plot_data):
            circle = plt.Circle((index,point), 0.05, color='b')
            axis.add_patch(circle)
            self.result_plot.circles.append(circle)
        self.result_plot.plot_canvas = FigureCanvasTkAgg(fig, master=self.result_plot.plot_frame)
        self.result_plot.plot_canvas.draw()
        self.result_plot.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.result_plot.plot_canvas.mpl_connect('button_press_event', self.handle_click_on_result_plot)

    def update_result_plot(self, dataset):
        if os.path.isdir(f"model_cache/{dataset}"):
            self.result_plot.clear_figure()
            sorted_cache_files = sorted({val:float(val) for val in os.listdir(f"model_cache/{dataset}")}.items(), key=lambda item: item[1])
            for cache_file in sorted_cache_files:
                out_sample_mse = float(utils.get_attribute_from_model_file(dataset, "out_sample_mse_reduction", str(cache_file[0])))
                self.result_plot.plot_data.append(out_sample_mse)
                self.result_plot.models.append(cache_file[0])
            self.create_result_plot()

    def evaluate_model(self):
        if self.dnd.variables_displayed:
            # TODO: Improve the text displayed
            model_id, regression_result, print_string = api.evaluate_model(self.dnd.filename, self.build_model_indices_lists())
            self.dnd.canvas_print_out.insert(tk.END, print_string)
            if model_id != None:
                best_model_mse = api.get_best_model_for_dataset(self.dnd.data_source)[0]
                self.dnd.canvas_print_out.insert(tk.END, f"\nThe best model in the cache has MSE reduction of {str(best_model_mse*100)[:5]}%")
                self.dnd.save_canvas_to_cache(str(model_id))
                self.regression_plot.plot_new_regression_result(regression_result.summary2().tables[1], self.dnd.data_source, model_id)
                self.update_result_plot(self.dnd.data_source)

    def restore_model(self, model_id):
        self.dnd.restore_canvas_from_cache(str(model_id))
        self.regression_plot.restore_regression_result(self.dnd.data_source, str(model_id))

    def restore_best_model(self):
        if self.dnd.data_source == None:
            self.dnd.canvas_print_out.insert(tk.END, f"\nPlease load a dataset before restoring a model from cache.") 
        else:
            min_mse, model_id = api.get_best_model_for_dataset(self.dnd.data_source)
            if model_id == None:
                self.dnd.canvas_print_out.insert(tk.END, f"\nThere is no cached model for this dataset.")
            else:
                self.restore_model(self.dnd, self.regression_plot, model_id)

    def run_bayesian_inference(self):
        api.run_bayesian_regression(self.dnd.filename, self.build_model_indices_lists())

    def clear_canvas(self):
        self.dnd.clear_canvas()
        self.regression_plot.clear_figure()
        self.result_plot.clear_figure()

    def clear_model_cache(self):
        api.clear_model_cache(self.dnd.data_source)
        self.result_plot.clear_figure()

    def on_close(self):
        self.window.quit()
        self.window.destroy()
