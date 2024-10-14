import pandas as pd
import os

import tkinter as tk
from tkinter import filedialog

import climate_econometrics_toolkit.climate_econometrics_api as api
from climate_econometrics_toolkit.DragAndDropInterface import DragAndDropInterface
from climate_econometrics_toolkit.RegressionPlot import RegressionPlot
from climate_econometrics_toolkit.ResultPlot import ResultPlot

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def add_data_columns_from_file():

    if dnd.variables_displayed:
        dnd.canvas_print_out.insert(tk.END, "\nPlease clear the canvas before loading another dataset.")
    else:
        # filename = filedialog.askopenfilename(
        #     initialdir = "/",
        #     title = "Select a File",
        #     filetypes = (("CSV files",
        #                 "*.csv*"),
        #                 ("all files",
        #                 "*.*"))
        #     )
        filename = "/home/hayden-freedman/climate_econometrics_toolkit/GrowthClimateDataset.csv"

        dnd.data_source = filename.split("/")[-1]
        data = pd.read_csv(filename)
        columns = data.columns
        dnd.add_model_variables(columns)

        result_plot.update_result_plot(dnd.data_source)

def evaluate_model():
    if dnd.variables_displayed:
        from_indices,to_indices = [],[]
        for element_id in canvas.find_all():
            element_tags = canvas.gettags(element_id)
            if dnd.tags_are_arrow(element_tags):
                from_indices.append(element_tags[0].split("boxed_text_")[1])
                to_indices.append(element_tags[1].split("boxed_text_")[1])
        model_id, regression_result, print_string = api.evaluate_model(dnd.data_source, [from_indices,to_indices])
        print(regression_result.summary2().tables[1])
        dnd.canvas_print_out.insert(tk.END, print_string)
        if model_id != None:
            best_model_mse = api.get_best_model_for_dataset(dnd.data_source)[0]
            dnd.canvas_print_out.insert(tk.END, f"\nThe best model in the cache has MSE reduction of {str(best_model_mse*100)[:5]}%")
            dnd.save_canvas_to_cache(str(model_id))
            regression_plot.plot_new_regression_result(regression_result.summary2().tables[1], dnd.data_source, model_id)
            result_plot.update_result_plot(dnd.data_source)

def restore_best_model():
    if dnd.data_source == None:
        dnd.canvas_print_out.insert(tk.END, f"\nPlease load a dataset before restoring a model from cache.") 
    else:
        min_mse, model_id = api.get_best_model_for_dataset(dnd.data_source)
        if model_id == None:
            dnd.canvas_print_out.insert(tk.END, f"\nThere is no cached model for this dataset.")
        else:
            dnd.restore_canvas_from_cache(str(model_id))
            regression_plot.restore_regression_result(dnd.data_source, str(model_id))

def clear_canvas():
    dnd.clear_canvas()
    regression_plot.clear_figure()
    result_plot.clear_figure()

def clear_model_cache():
    api.clear_model_cache(dnd.data_source)
    result_plot.clear_figure()

def on_close():
    window.quit()

window = tk.Tk()
window.title("Climate Econometrics Modeling Toolkit")

window.rowconfigure(0, minsize=100, weight=1)
window.rowconfigure(1, minsize=100, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

window.protocol("WM_DELETE_WINDOW", on_close)

canvas = tk.Canvas(
    window, 
    width=800, 
    height=800, 
    highlightthickness=5,
    highlightbackground="black",
    highlightcolor="red"
)
regression_plot_frame = tk.Frame(window, relief=tk.RAISED, bd=2)
result_plot_frame = tk.Frame(window, relief=tk.RAISED, bd=2)

dnd = DragAndDropInterface(canvas, window)
regression_plot = RegressionPlot(regression_plot_frame)
result_plot = ResultPlot(result_plot_frame)

lefthand_bar = tk.Frame(window, relief=tk.RAISED, bd=2)
btn_load = tk.Button(lefthand_bar, text="Load Dataset", command=add_data_columns_from_file)
btn_clear_canvas = tk.Button(lefthand_bar, text="Clear Canvas", command=clear_canvas)
btn_evaluate = tk.Button(lefthand_bar, text="Evaluate Model", command=evaluate_model)
btn_best_model = tk.Button(lefthand_bar, text="Restore Best Model", command=restore_best_model)
btn_clear_model_cache = tk.Button(lefthand_bar, text="Clear Model Cache", command=clear_model_cache)
result_text = tk.Text(lefthand_bar, height=10)

btn_load.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
btn_clear_canvas.grid(row=1, column=0, stick="nsew", padx=5)
btn_evaluate.grid(row=2, column=0, sticky="nsew", padx=5)
btn_best_model.grid(row=3, column=0, sticky="nsew", padx=5)
btn_clear_model_cache.grid(row=4, column=0, sticky="nsew", padx=5)
result_text.grid(row=5, column=0)

lefthand_bar.grid(row=0, column=0, sticky="ns")
regression_plot_frame.grid(row=1, column=0, sticky="ns")
canvas.grid(row=0, column=1, sticky="nsew")
result_plot_frame.grid(row=1, column=1, stick="nsew")

dnd.canvas_print_out = result_text
window.mainloop()