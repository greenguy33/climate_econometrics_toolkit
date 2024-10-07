import tkinter as tk
from tkinter import filedialog
import pandas as pd
import pickle as pkl
import os

import climate_econometrics_toolkit.climate_econometrics_api as api

class DragAndDropInterface():

    def __init__(self, canvas):
        self.canvas = canvas
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_object = None
        self.in_drag = False
        self.arrow_list = []
        self.variables_displayed = False
        self.data_source = None
        self.canvas_print_out = None

        self.canvas.bind("<ButtonPress-1>", self.handle_canvas_click)

    def save_canvas_to_cache(self, model_id):
        canvas_data = []
        for item in self.canvas.find_all():
            item_info = {
                "type":self.canvas.type(item),
                "coords":self.canvas.coords(item),
                "tags":self.canvas.gettags(item)
            }
            if self.canvas.type(item) == "text":
                item_info["text"] = self.canvas.itemcget(item, "text")
            canvas_data.append(item_info)
        with open (f'model_cache/{model_id}/tkinter_canvas.pkl', 'wb') as buff:
            pkl.dump({"data_source":self.data_source,"canvas_data":canvas_data},buff)

    def restore_canvas_from_cache(self, model_id):
        cached_canvas = pd.read_pickle(f'model_cache/{model_id}/tkinter_canvas.pkl')
        if cached_canvas["data_source"] != self.data_source:
            dnd.canvas_print_out.insert(tk.END, f"\nCached model is for a different data source. Please clear cache to use new dataset.")  
        else:
            self.clear_canvas()
            for item in cached_canvas["canvas_data"]:
                if item["type"] == "rectangle":
                    self.canvas.create_rectangle(*item["coords"], fill="orange", tags=item["tags"])
                elif item["type"] == "line":
                    self.canvas.create_line(*item["coords"], arrow=tk.LAST, tags=item["tags"])
                    self.arrow_list.append(self.get_arrow_source_and_target(item["tags"]))
                elif item["type"] == "text":
                    self.canvas.create_text(*item["coords"], text=item["text"], fill="white", tags=item["tags"])
            self.variables_displayed = True

    def tags_are_arrow(self, element_tags):
        if (
            len(element_tags) >= 2 and 
            element_tags[0].startswith("from_") and 
            element_tags[1].startswith("to_")
        ):
            return True
        else:
            return False

    def get_arrow_source_and_target(self, arrow_tags):
        arrow_source = [elem for elem in self.canvas.find_withtag(arrow_tags[0].split("from_")[1]) if self.canvas.type(elem) == "text"][0]
        arrow_target = [elem for elem in self.canvas.find_withtag(arrow_tags[1].split("to_")[1]) if self.canvas.type(elem) == "text"][0]
        return (arrow_source, arrow_target)

    def add_model_variables(self, variables):

        last_rectangle_right_side = 0
        row_count = 0
        for index, column in enumerate(variables):
            if last_rectangle_right_side > self.canvas.winfo_width() - 100:
                row_count += 1
                last_rectangle_right_side = 0

            column_box_tag = f"boxed_text_{column}".replace(" ","_")

            text = self.canvas.create_text(last_rectangle_right_side + len(column)*5+50, row_count * 50 + 20, text=column, fill="white", tags=column_box_tag)
            rect = self.canvas.create_rectangle(self.canvas.bbox(text), fill="orange", tags=column_box_tag)
            self.canvas.lower(rect)

            self.canvas.tag_bind(column_box_tag, "<B1-Motion>", self.on_drag)
            self.canvas.tag_bind(column_box_tag, "<ButtonRelease-1>", self.end_drag)

            last_rectangle_right_side = self.canvas.bbox(text)[2]

        self.variables_displayed = True

    def handle_canvas_click(self, event):
        clicked_object = canvas.find_overlapping(event.x, event.y, event.x, event.y)
        if len(clicked_object) == 0:
            self.reset_click()
        else:
            self.on_click(event)
        
    def reset_click(self):
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_object = None

    def end_drag(self, event):
        if self.in_drag:
            self.reset_click()
        self.in_drag = False

    def draw_arrow(self, event):
        source_object = self.drag_object
        target_object = self.canvas.find_closest(event.x, event.y)[0]
        if not self.tags_are_arrow(canvas.gettags(target_object)):
            if source_object != target_object and (source_object,target_object) not in self.arrow_list and (target_object,source_object) not in self.arrow_list:
                target_bb = canvas.bbox(target_object)
                source_bb = canvas.bbox(source_object)
                # TODO: ensure that an arrow cannot have another arrow as a source or target
                arrow = self.canvas.create_line(
                    (source_bb[0] + source_bb[2]) / 2,
                    (source_bb[1] + source_bb[3]) / 2,
                    (target_bb[0] + target_bb[2]) / 2,
                    (target_bb[1] + target_bb[3]) / 2,
                    arrow=tk.LAST,
                    tags=[
                        f"from_{self.canvas.gettags(source_object)[0]}",
                        f"to_{self.canvas.gettags(target_object)[0]}"
                    ]
                )
                canvas.tag_bind(f"from_{self.canvas.gettags(source_object)[0]}", "<ButtonPress-3>", self.delete_arrow)
                self.arrow_list.append((source_object,target_object))
            self.reset_click()

    def clear_canvas(self):
        self.reset_click()
        self.arrow_list = []
        self.canvas.delete("all")
        self.variables_displayed = False

    def delete_arrow(self, event):
        arrow = self.canvas.find_closest(event.x, event.y)[0]
        arrow_tags = canvas.gettags(arrow)
        if self.tags_are_arrow(arrow_tags): 
            self.arrow_list.remove(self.get_arrow_source_and_target(arrow_tags))
            self.canvas.delete(arrow)

    def update_arrow_coordinates(self, event, delta_x, delta_y):

        arrow_source_tags = f"from_{self.canvas.gettags(self.drag_object)[0]}"
        arrow_target_tags = f"to_{self.canvas.gettags(self.drag_object)[0]}"

        for arrow in self.canvas.find_withtag(arrow_source_tags):
            arrow_source_coords = self.canvas.coords(arrow)
            arrow_source_coords[0] += delta_x
            arrow_source_coords[1] += delta_y
            self.canvas.coords(arrow, *arrow_source_coords)

        for arrow in self.canvas.find_withtag(arrow_target_tags):
            arrow_target_coords = self.canvas.coords(arrow)
            arrow_target_coords[2] += delta_x
            arrow_target_coords[3] += delta_y
            self.canvas.coords(arrow, *arrow_target_coords)

    def on_click(self, event):
        if self.drag_object == None:
            clicked_object = self.canvas.find_closest(event.x, event.y)[0]
            tags = self.canvas.gettags(clicked_object)
            if not self.tags_are_arrow(tags):
                self.drag_object = clicked_object
                self.drag_start_x = event.x
                self.drag_start_y = event.y
                print(tags)
                print(self.canvas.find_withtag(tags))
                clicked_rectangle = [elem for elem in self.canvas.find_withtag(tags) if self.canvas.type(elem) == "rectangle"][0]
                canvas.itemconfig(clicked_rectangle, fill='red')
        else:
            self.draw_arrow(event)

    def on_drag(self, event):

        self.in_drag = True

        canvas_buffer = 25
        if event.x >= canvas_buffer and event.y >= canvas_buffer and event.x <= self.canvas.winfo_width()-canvas_buffer and event.y <= self.canvas.winfo_height()-canvas_buffer:

            delta_x = event.x - self.drag_start_x
            delta_y = event.y - self.drag_start_y
            self.canvas.move(self.canvas.gettags(self.drag_object)[0], delta_x, delta_y)

            self.drag_start_x = event.x
            self.drag_start_y = event.y

            self.update_arrow_coordinates(event, delta_x, delta_y)


def add_data_columns_from_file():

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

        dnd.data_source = filename
        data = pd.read_csv(filename)
        columns = data.columns
        dnd.add_model_variables(columns)

def evaluate_model():
    if dnd.variables_displayed:
        from_indices,to_indices = [],[]
        for element_id in canvas.find_all():
            element_tags = canvas.gettags(element_id)
            if dnd.tags_are_arrow(element_tags):
                from_indices.append(element_tags[0].split("boxed_text_")[1])
                to_indices.append(element_tags[1].split("boxed_text_")[1])
        model_id, result = api.evaluate_model(dnd.data_source, [from_indices,to_indices])
        dnd.canvas_print_out.insert(tk.END, result)
        if model_id != None:
            best_model_mse = api.get_best_model_for_dataset(dnd.data_source)[0]
            dnd.canvas_print_out.insert(tk.END, f"\nThe best model in the cache has MSE {str(best_model_mse)[:7]}")
            dnd.save_canvas_to_cache(str(model_id))

def restore_best_model():
    if dnd.data_source == None:
        dnd.canvas_print_out.insert(tk.END, f"\nPlease load a dataset before restoring a model from cache.") 
    else:
        min_mse, model_id = api.get_best_model_for_dataset(dnd.data_source)
        if model_id == None:
            dnd.canvas_print_out.insert(tk.END, f"\nThere is no cached model for this dataset.")
        else:
            dnd.restore_canvas_from_cache(str(model_id))

def clear_model_cache():
    api.clear_model_cache(dnd.data_source)

window = tk.Tk()
window.title("Climate Econometrics Modeling Toolkit")

window.rowconfigure(0, minsize=800, weight=0)
window.rowconfigure(1, minsize=800, weight=0)
window.columnconfigure(1, minsize=800, weight=0)

canvas = tk.Canvas(
    window, 
    width=800, 
    height=600, 
    highlightthickness=5,
    highlightbackground="black",
    highlightcolor="red"
)
dnd = DragAndDropInterface(canvas)

frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
btn_load = tk.Button(frm_buttons, text="Load Dataset", command=add_data_columns_from_file)
btn_clear_canvas = tk.Button(frm_buttons, text="Clear Canvas", command=dnd.clear_canvas)
btn_evaluate = tk.Button(frm_buttons, text="Evaluate Model", command=evaluate_model)
btn_best_model = tk.Button(frm_buttons, text="Restore Best Model", command=restore_best_model)
btn_clear_model_cache = tk.Button(frm_buttons, text="Clear Model Cache", command=clear_model_cache)
result_text = tk.Text(frm_buttons)

dnd.canvas_print_out = result_text

btn_load.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_clear_canvas.grid(row=1, column=0, stick="ew", padx=5)
btn_evaluate.grid(row=2, column=0, sticky="ew", padx=5)
btn_best_model.grid(row=3, column=0, sticky="ew", padx=5)
btn_clear_model_cache.grid(row=4, column=0, sticky="ew", padx=5)
result_text.grid(row=5, column=0)
frm_buttons.grid(row=0, column=0, sticky="ns")
canvas.grid(row=0, column=1, sticky="nsew")

window.mainloop()