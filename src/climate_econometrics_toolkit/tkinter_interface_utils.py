import pandas as pd

import tkinter as tk
from tkinter import filedialog

import climate_econometrics_toolkit.climate_econometrics_api as api


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def add_data_columns_from_file(dnd, result_plot):

    if dnd.variables_displayed:
        dnd.canvas_print_out.insert(tk.END, "\nPlease clear the canvas before loading another dataset.")
    else:
        filename = filedialog.askopenfilename(
            initialdir = "/",
            title = "Select a File",
            filetypes = (("CSV files",
                        "*.csv*"),
                        ("all files",
                        "*.*"))
            )
        # filename = "/home/hayden-freedman/climate_econometrics_toolkit/GrowthClimateDataset.csv"

        dnd.data_source = filename.split("/")[-1]
        dnd.filename = filename
        data = pd.read_csv(filename)
        columns = data.columns
        dnd.add_model_variables(columns)

        result_plot.update_result_plot(dnd.data_source)

def build_model_indices_lists(dnd, canvas):
    from_indices,to_indices = [],[]
    for element_id in canvas.find_all():
        element_tags = canvas.gettags(element_id)
        if dnd.tags_are_arrow(element_tags):
            from_indices.append(element_tags[0].split("boxed_text_")[1])
            to_indices.append(element_tags[1].split("boxed_text_")[1])
    return [from_indices, to_indices]

def evaluate_model(dnd, canvas, regression_plot, result_plot):
    if dnd.variables_displayed:
        model_id, regression_result, print_string = api.evaluate_model(dnd.filename, build_model_indices_lists(dnd, canvas))
        dnd.canvas_print_out.insert(tk.END, print_string)
        if model_id != None:
            best_model_mse = api.get_best_model_for_dataset(dnd.data_source)[0]
            dnd.canvas_print_out.insert(tk.END, f"\nThe best model in the cache has MSE reduction of {str(best_model_mse*100)[:5]}%")
            dnd.save_canvas_to_cache(str(model_id))
            regression_plot.plot_new_regression_result(regression_result.summary2().tables[1], dnd.data_source, model_id)
            result_plot.update_result_plot(dnd.data_source)

def restore_best_model(dnd, regression_plot):
    if dnd.data_source == None:
        dnd.canvas_print_out.insert(tk.END, f"\nPlease load a dataset before restoring a model from cache.") 
    else:
        min_mse, model_id = api.get_best_model_for_dataset(dnd.data_source)
        if model_id == None:
            dnd.canvas_print_out.insert(tk.END, f"\nThere is no cached model for this dataset.")
        else:
            dnd.restore_canvas_from_cache(str(model_id))
            regression_plot.restore_regression_result(dnd.data_source, str(model_id))

def run_bayesian_inference(dnd, canvas):
    api.run_bayesian_regression(dnd.filename, build_model_indices_lists(dnd, canvas))

def clear_canvas(dnd, regression_plot, result_plot):
    dnd.clear_canvas()
    regression_plot.clear_figure()
    result_plot.clear_figure()

def clear_model_cache(dnd, result_plot):
    api.clear_model_cache(dnd.data_source)
    result_plot.clear_figure()

def on_close(window):
    window.quit()
    window.destroy()
