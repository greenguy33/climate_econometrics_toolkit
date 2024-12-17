import pandas as pd
import os

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.transforms as transform

import climate_econometrics_toolkit.interface_api as api
import climate_econometrics_toolkit.utils as utils
from climate_econometrics_toolkit.GcmSelectionPopup import GcmSelectionPopup

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

cet_home = os.getenv("CETHOME")

class TkInterfaceUtils():

    def __init__(self, window, canvas, dnd, regression_plot, result_plot, stat_plot):
        self.window = window
        self.canvas = canvas
        self.dnd = dnd
        self.regression_plot = regression_plot
        self.result_plot = result_plot
        self.stat_plot = stat_plot
        self.panel_column = None
        self.time_column = None

    def add_data_columns_from_file(self):

        if self.dnd.variables_displayed:
            self.dnd.canvas_print_out.insert(tk.END, "\nPlease clear the canvas before loading another dataset.")
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
                self.dnd.canvas_print_out.insert(tk.END, f"\nERROR: This dataset exceeds the maximum number of columns(100)")
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

    def update_result_plot(self, dataset, metric):
        if os.path.isdir(f"{cet_home}/model_cache/{dataset}"):
            self.result_plot.clear_figure()
            sorted_cache_files = sorted({val:float(val) for val in os.listdir(f"model_cache/{dataset}")}.items(), key=lambda item: item[1])
            for cache_file in sorted_cache_files:
                values = float(utils.get_attribute_from_model_file(dataset, metric, str(cache_file[0])))
                self.result_plot.plot_data.append(values)
                self.result_plot.models.append(cache_file[0])
            self.create_result_plot(metric)
            cached_canvas = pd.read_pickle(f"{cet_home}/model_cache/{dataset}/{cache_file[0]}/tkinter_canvas.pkl")
            return cached_canvas["panel_column"], cached_canvas["time_column"]

    def get_regression_stats_from_model(self ,model_id):
        out_sample_mse = float(utils.get_attribute_from_model_file(self.dnd.data_source, "out_sample_mse_reduction", str(model_id)))
        pred_int_cov = float(utils.get_attribute_from_model_file(self.dnd.data_source, "out_sample_pred_int_cov", str(model_id)))
        r2 = float(utils.get_attribute_from_model_file(self.dnd.data_source, "r2", str(model_id)))
        rmse = float(utils.get_attribute_from_model_file(self.dnd.data_source, "rmse", str(model_id)))
        return out_sample_mse, pred_int_cov, r2, rmse
    
    def bind_stat_canvases_to_result_plot(self, mse_canvas, pred_int_canvas, r2_canvas, rmse_canvas):
        mse_canvas.bind("<ButtonPress-1>", lambda x, data=self.dnd.data_source, metric="out_sample_mse_reduction" : self.update_result_plot(data, metric))
        pred_int_canvas.bind("<ButtonPress-1>", lambda x, data=self.dnd.data_source, metric="out_sample_pred_int_cov" : self.update_result_plot(data, metric))
        r2_canvas.bind("<ButtonPress-1>", lambda x, data=self.dnd.data_source, metric="r2" : self.update_result_plot(data, metric))
        rmse_canvas.bind("<ButtonPress-1>", lambda x, data=self.dnd.data_source, metric="rmse": self.update_result_plot(data, metric))

    def evaluate_model(self):
        if self.dnd.variables_displayed:
            # TODO: Improve the text displayed
            model_id, regression_result, print_string = api.evaluate_model(self.dnd.filename, self.build_model_indices_lists(), self.panel_column, self.time_column)
            self.dnd.canvas_print_out.insert(tk.END, print_string)
            if model_id != None:
                # best_model_mse = api.get_best_model_for_dataset(self.dnd.data_source)[0]
                # self.dnd.canvas_print_out.insert(tk.END, f"\nThe best model in the cache has MSE reduction of {str(best_model_mse*100)[:5]}%")
                self.dnd.save_canvas_to_cache(str(model_id), self.panel_column, self.time_column)
                self.regression_plot.plot_new_regression_result(regression_result.summary2().tables[1], self.dnd.data_source, model_id)
                self.update_result_plot(self.dnd.data_source, "r2")
                canvases = self.stat_plot.update_stat_plot(*self.get_regression_stats_from_model(model_id))
                self.bind_stat_canvases_to_result_plot(*canvases)
        else:
            self.dnd.canvas_print_out.insert(tk.END, "\nPlease load a dataset and create a model before evaluating model.")
        return model_id

    def restore_model(self, model_id):
        self.dnd.restore_canvas_from_cache(str(model_id))
        self.regression_plot.restore_regression_result(self.dnd.data_source, str(model_id))
        canvases = self.stat_plot.update_stat_plot(*self.get_regression_stats_from_model(model_id))
        self.bind_stat_canvases_to_result_plot(*canvases)

    def restore_best_model(self):
        if self.dnd.data_source == None:
            self.dnd.canvas_print_out.insert(tk.END, f"\nPlease load a dataset before restoring a model from cache.") 
        else:
            min_mse, model_id = api.get_best_model_for_dataset(self.dnd.data_source)
            if model_id == None:
                self.dnd.canvas_print_out.insert(tk.END, f"\nThere is no cached model for this dataset.")
            else:
                self.restore_model(model_id)

    def run_bayesian_inference(self):
        # TODO: I don't like how we need to evaluate the model to get the model id
        model_id = self.evaluate_model()
        api.run_bayesian_regression(self.dnd.filename, model_id, use_threading=True)

    def run_block_bootstrap(self):
        model_id = self.evaluate_model()
        # TODO: I don't like how we need to evaluate the model to get the model id
        api.run_block_bootstrap(self.dnd.filename, model_id, use_threading=True)

    def predict_from_gcms(self, window):
        gcm_popup = GcmSelectionPopup(window)
        model_id = self.evaluate_model()
        # # TODO: I don't like how we need to evaluate the model to get the model id
        api.predict_from_gcms(self.dnd.filename, model_id, list(gcm_popup.gcms_to_use), use_threading=True)

    def clear_canvas(self):
        self.dnd.clear_canvas()
        self.regression_plot.clear_figure()
        self.result_plot.clear_figure()
        self.stat_plot.clear_stat_plot()
        self.panel_column = None
        self.time_column = None

    def clear_model_cache(self):
        api.clear_model_cache(self.dnd.data_source)
        self.result_plot.clear_figure()
        self.dnd.canvas_print_out.insert(tk.END, "\nModel cache cleared")

    def on_close(self):
        self.window.quit()
        self.window.destroy()
