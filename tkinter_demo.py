import tkinter as tk
from tkinter import filedialog
import pandas as pd


class DragAndDropInterface():

    def __init__(self, canvas):
        self.canvas = canvas
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_object = None
        self.in_drag = False
        self.arrow_list = []
        self.variables_displayed = False

    def add_model_variables(self, variables):

        if not self.variables_displayed:

            for index, column in enumerate(variables):

                column_box_tag = f"boxed_text_{column}"

                text = self.canvas.create_text(50 * ((index+1)), 10, text=column, fill="white", tags=column_box_tag)
                rect = self.canvas.create_rectangle(self.canvas.bbox(text), fill="orange", tags=column_box_tag)
                self.canvas.lower(rect)

                self.canvas.tag_bind(column_box_tag, "<ButtonPress-1>", self.on_click)
                self.canvas.tag_bind(column_box_tag, "<B1-Motion>", self.on_drag)
                self.canvas.tag_bind(column_box_tag, "<ButtonRelease-1>", self.end_drag)

            self.variables_displayed = True

    def reset_canvas(self):
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_object = None

    def end_drag(self, event):
        if self.in_drag:
            self.reset_canvas()
        self.in_drag = False

    def draw_arrow(self, event):
        source_object = self.drag_object
        target_object = self.canvas.find_closest(event.x, event.y)[0]
        if source_object != target_object and (source_object,target_object) not in self.arrow_list:
            self.canvas.create_line(
                self.drag_start_x, 
                self.drag_start_y, 
                event.x, event.y, 
                arrow=tk.LAST,
                tags=[
                    f"from_{self.canvas.gettags(source_object)[0]}",
                    f"to_{self.canvas.gettags(target_object)[0]}"
                ]
            )
            self.arrow_list.append((source_object,target_object))
            self.reset_canvas()

    def update_arrow_coordinates(self, event, delta_x, delta_y):

        arrow_source_tags = f"from_{self.canvas.gettags(self.canvas.find_closest(event.x, event.y)[0])[0]}"
        arrow_target_tags = f"to_{self.canvas.gettags(self.drag_object)[0]}"

        arrow_source_coords = self.canvas.coords(arrow_source_tags)
        if len(arrow_source_coords) > 0:
            arrow_source_coords[0] += delta_x
            arrow_source_coords[1] += delta_y
            self.canvas.coords(arrow_source_tags, *arrow_source_coords)

        arrow_target_coords = self.canvas.coords(arrow_target_tags)
        if len(arrow_target_coords) > 0:
            arrow_target_coords[2] += delta_x
            arrow_target_coords[3] += delta_y
            self.canvas.coords(arrow_target_tags, *arrow_target_coords)

    def on_click(self, event):
        if self.drag_object == None:
            self.drag_object = self.canvas.find_closest(event.x, event.y)[0]
            self.drag_start_x = event.x
            self.drag_start_y = event.y
        else:
            self.draw_arrow(event)

    def on_drag(self, event):
        self.in_drag = True

        delta_x = event.x - self.drag_start_x
        delta_y = event.y - self.drag_start_y
        self.canvas.move(self.canvas.gettags(self.drag_object)[0], delta_x, delta_y)

        self.drag_start_x = event.x
        self.drag_start_y = event.y

        self.update_arrow_coordinates(event, delta_x, delta_y)


def add_data_columns_from_file():
    # filename = filedialog.askopenfilename(
    #     initialdir = "/",
    #     title = "Select a File",
    #     filetypes = (("CSV files",
    #                 "*.csv*"),
    #                 ("all files",
    #                 "*.*"))
    #     )
    # data = pd.read_csv(filename)
    # columns = data.columns
    
    dnd.add_model_variables(["Temp","Precip","GDP"])


def print_model():
    model_links = {}
    for element_id in canvas.find_all():
        element_tags = canvas.gettags(element_id)
        # TODO: need better rule for identifying link tags
        if len(element_tags) == 2:
            model_links[element_tags[0].split("boxed_text_")[1]] = element_tags[1].split("boxed_text_")[1]
    for target, source in model_links.items():
        print(target, "-->", source)

window = tk.Tk()
window.title("Climate Econometrics Modeling Toolkit")

window.rowconfigure(1, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

canvas = tk.Canvas(window)
dnd = DragAndDropInterface(canvas)

frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
btn_load = tk.Button(frm_buttons, text="Load Model Variables", command=add_data_columns_from_file)
btn_evaluate = tk.Button(frm_buttons, text="Evaluate Model", command=print_model)

btn_load.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_evaluate.grid(row=1, column=0, sticky="ew", padx=5)
frm_buttons.grid(row=0, column=0, sticky="ns")
canvas.grid(row=0, column=1, sticky="nsew")

window.mainloop()