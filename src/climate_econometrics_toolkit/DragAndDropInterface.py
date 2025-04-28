import sys
import os

import tkinter as tk
from tkinter import Menu

import pickle as pkl
import pandas as pd
import webbrowser

import climate_econometrics_toolkit.utils as utils


cet_home = os.getenv("CETHOME")

class DragAndDropInterface():

    def __init__(self, canvas, window):
        self.window = window
        self.canvas = canvas
        self.drag_start_x = None
        self.drag_start_y = None
        self.left_clicked_object = None
        self.left_clicked_object_tk = tk.StringVar(value=(""))
        self.object_to_drag = None
        self.right_clicked_object = None
        self.in_drag = False
        self.arrow_list = []
        self.transformation_list = []
        self.variables_displayed = False
        self.data_source = None
        self.filename = None
        self.canvas_print_out = None
        self.menu = None
        self.time_column = None
        self.panel_column = None
        self.arrow_width = 3
        self.fontsize = 18
        self.right_click_button = "<ButtonPress-3>"
        self.current_model = None
        if sys.platform == "darwin":
            self.right_click_button = "<ButtonPress-2>"

        self.canvas.bind("<ButtonPress-1>", self.handle_canvas_click)
        self.canvas.bind("<ButtonRelease-1>", self.handle_canvas_release)

    def get_menu(self, tag):

        main_menu = Menu(self.window, tearoff=0)

        if not any(tag.startswith(val+"(") for val in utils.supported_effects):

            transformation_menu = Menu(main_menu, tearoff=0)
            time_trends_menu = Menu(main_menu, tearoff=0)

            if f"sq({tag})" not in self.transformation_list:
                transformation_menu.add_command(label="Square",command=lambda : self.add_transformation("sq"))
            if f"cu({tag})" not in self.transformation_list:
                transformation_menu.add_command(label="Cube",command=lambda : self.add_transformation("cu"))
            if f"fd({tag})" not in self.transformation_list:
                transformation_menu.add_command(label="First Difference",command=lambda : self.add_transformation("fd"))
            if f"ln({tag})" not in self.transformation_list:
                transformation_menu.add_command(label="Natural Log",command=lambda : self.add_transformation("ln"))
            if f"lag1({tag})" not in self.transformation_list:
                transformation_menu.add_command(label="Lag 1",command=lambda : self.add_transformation("lag1"))
            if f"lag2({tag})" not in self.transformation_list:
                transformation_menu.add_command(label="Lag 2",command=lambda : self.add_transformation("lag2"))
            if f"lag3({tag})" not in self.transformation_list:
                transformation_menu.add_command(label="Lag 3",command=lambda : self.add_transformation("lag3"))
            if f"scale({tag})" not in self.transformation_list:
                transformation_menu.add_command(label="Scale (Mean=0, SD=1)",command=lambda : self.add_transformation("scale"))
            if not all(f"{func}({tag})" in self.transformation_list for func in utils.supported_functions):
                main_menu.add_cascade(label="Duplicate with Transformation",menu=transformation_menu)

            if not any(tag.startswith(val) for val in utils.supported_functions) and f"fe({tag})" not in self.transformation_list:
                main_menu.add_command(label="Add Group-Level Fixed Intercepts",command=lambda : self.add_transformation("fe"))
            if not any(tag.startswith(val) for val in utils.supported_functions) and not any(val.startswith("re(") for val in self.transformation_list):
                main_menu.add_command(label="Add Group-Level Random Slopes",command=lambda : self.add_transformation("re"))

            if not any(tag.startswith(val) for val in utils.supported_functions):
                main_menu.add_cascade(label="Add Time Trend",menu=time_trends_menu)
                if f"tt1({tag})" not in self.transformation_list:
                    time_trends_menu.add_command(label="X 1",command=lambda : self.add_transformation("tt1"))
                if f"tt2({tag})" not in self.transformation_list:
                    time_trends_menu.add_command(label="X 2",command=lambda : self.add_transformation("tt2"))
                if f"tt3({tag})" not in self.transformation_list:
                    time_trends_menu.add_command(label="X 3",command=lambda : self.add_transformation("tt3"))

        main_menu.add_command(label="Delete Variable", command=self.remove_node)
        main_menu.add_command(label="Swap with Other Variable", command=self.swap_node)

        if not any(tag.startswith(val+"(") for val in utils.supported_functions) and not any(tag.startswith(val) for val in utils.supported_effects) and tag != self.panel_column and tag != self.time_column:
            main_menu.add_command(label="Open Data Explorer", command=self.open_data_explorer)

        return main_menu

    def open_data_explorer(self):
        element_tag = self.canvas.gettags(self.right_clicked_object)[0]
        element_text = element_tag.split("boxed_text_")[1]
        data = pd.read_csv(self.filename)
        data.columns = [col.replace(" ","_") for col in data.columns]
        html_data = data[[self.panel_column,self.time_column,element_text]].set_index([self.panel_column,self.time_column]).unstack().style.applymap(lambda x: "background-color: red" if pd.isnull(x) else "background-color: lime").to_html()
        temp_file_path = f"{cet_home}/html/data_{element_text}.html"
        with open(temp_file_path,"w") as tmpfile:
            tmpfile.write(html_data)
        webbrowser.open(temp_file_path)

    def remove_node(self):
        element_tag = self.canvas.gettags(self.right_clicked_object)[0]
        rectangle = [elem for elem in self.canvas.find_withtag(element_tag) if self.canvas.type(elem) == "rectangle"][0]
        arrow_source_tags = f"from_{element_tag}"
        arrow_target_tags = f"to_{element_tag}"
        for arrow in self.canvas.find_withtag(arrow_source_tags):
            arrow_tags = self.canvas.gettags(arrow)
            self.arrow_list.remove(self.get_arrow_source_and_target(arrow_tags))
            self.canvas.delete(arrow)
        for arrow in self.canvas.find_withtag(arrow_target_tags):
            arrow_tags = self.canvas.gettags(arrow)
            self.arrow_list.remove(self.get_arrow_source_and_target(arrow_tags))
            self.canvas.delete(arrow)
        if element_tag.replace("boxed_text_","") in self.transformation_list:
            self.transformation_list.remove(element_tag.replace("boxed_text_",""))
        self.canvas.delete(self.right_clicked_object)
        self.canvas.delete(rectangle)

    def bind_right_click_to_arrow_tag(self, arrow_tag):
        self.canvas.tag_bind(arrow_tag, self.right_click_button, self.delete_arrow_from_click)
        self.canvas.tag_bind(arrow_tag, "<Control-Button-1>", self.delete_arrow_from_click)
        self.canvas.tag_bind(arrow_tag, "<Command-Button-1>", self.delete_arrow_from_click)

    def update_arrow_list_in_node_swap(self, node1_text):
        arrows_to_add = []
        arrows_to_remove = []
        for arrow_tuple in self.arrow_list:
            if arrow_tuple[0] == node1_text and arrow_tuple[1] == self.left_clicked_object:
                arrows_to_remove.append(arrow_tuple)
                arrows_to_add.append((self.left_clicked_object, node1_text))
            elif arrow_tuple[1] == node1_text and arrow_tuple[0] == self.left_clicked_object:
                arrows_to_remove.append(arrow_tuple)
                arrows_to_add.append((node1_text, self.left_clicked_object))
            elif arrow_tuple[0] == node1_text:
                arrows_to_remove.append(arrow_tuple)
                arrows_to_add.append((self.left_clicked_object, arrow_tuple[1]))
            elif arrow_tuple[0] == self.left_clicked_object:
                arrows_to_remove.append(arrow_tuple)
                arrows_to_add.append((node1_text, arrow_tuple[1]))
            elif arrow_tuple[1] == node1_text:
                arrows_to_remove.append(arrow_tuple)
                arrows_to_add.append((arrow_tuple[0], self.left_clicked_object))
            elif arrow_tuple[1] == self.left_clicked_object:
                arrows_to_remove.append(arrow_tuple)
                arrows_to_add.append((arrow_tuple[0], node1_text))
        for tup in arrows_to_remove:
            self.arrow_list.remove(tup)
        for tup in arrows_to_add:
            self.arrow_list.append(tup)

    def update_arrows_for_node_swap(self, node1_text):
        node1_source_arrows = self.canvas.find_withtag(f"from_{self.canvas.gettags(node1_text)[0]}")
        node1_target_arrows = self.canvas.find_withtag(f"to_{self.canvas.gettags(node1_text)[0]}")
        node2_source_arrows = self.canvas.find_withtag(f"from_{self.canvas.gettags(self.left_clicked_object)[0]}")
        node2_target_arrows = self.canvas.find_withtag(f"to_{self.canvas.gettags(self.left_clicked_object)[0]}")
        for arrow in node1_source_arrows:
            self.canvas.dtag(arrow, f"from_{self.canvas.gettags(node1_text)[0]}")
            self.canvas.addtag(f"from_{self.canvas.gettags(self.left_clicked_object)[0]}", "withtag", arrow)
            self.bind_right_click_to_arrow_tag(f"from_{self.canvas.gettags(self.left_clicked_object)[0]}")
        for arrow in node1_target_arrows:
            self.canvas.dtag(arrow, f"to_{self.canvas.gettags(node1_text)[0]}")
            self.canvas.addtag(f"to_{self.canvas.gettags(self.left_clicked_object)[0]}", "withtag", arrow)
            self.bind_right_click_to_arrow_tag(f"to_{self.canvas.gettags(self.left_clicked_object)[0]}")
        for arrow in node2_source_arrows:
            self.canvas.dtag(arrow, f"from_{self.canvas.gettags(self.left_clicked_object)[0]}")
            self.canvas.addtag(f"from_{self.canvas.gettags(node1_text)[0]}", "withtag", arrow)
            self.bind_right_click_to_arrow_tag(f"from_{self.canvas.gettags(node1_text)[0]}")
        for arrow in node2_target_arrows:
            self.canvas.dtag(arrow, f"to_{self.canvas.gettags(self.left_clicked_object)[0]}")
            self.canvas.addtag(f"to_{self.canvas.gettags(node1_text)[0]}", "withtag", arrow)
            self.bind_right_click_to_arrow_tag(f"to_{self.canvas.gettags(node1_text)[0]}")
        self.update_arrow_list_in_node_swap(node1_text)

    def swap_node(self):
        self.left_clicked_object = None
        self.left_clicked_object_tk.set("")
        element_tag = self.canvas.gettags(self.right_clicked_object)[0]
        rectangle = [elem for elem in self.canvas.find_withtag(element_tag) if self.canvas.type(elem) == "rectangle"][0]
        text = [elem for elem in self.canvas.find_withtag(element_tag) if self.canvas.type(elem) == "text"][0]
        self.window.wait_variable(self.left_clicked_object_tk)
        tags = self.canvas.gettags(self.left_clicked_object)
        clicked_rectangle = [elem for elem in self.canvas.find_withtag(tags[0]) if self.canvas.type(elem) == "rectangle"][0]
        source_coords = self.canvas.coords(text)
        target_coords = self.canvas.coords(self.left_clicked_object)
        self.canvas.coords(text, *target_coords)
        self.canvas.coords(self.left_clicked_object, *source_coords)
        node1_bbox = self.canvas.bbox(text)
        node2_bbox = self.canvas.bbox(self.left_clicked_object)
        self.update_arrows_for_node_swap(text)
        rect1 = self.canvas.create_rectangle(node1_bbox, fill="orange", tags=self.canvas.gettags(rectangle))
        rect2 = self.canvas.create_rectangle(node2_bbox, fill="orange", tags=self.canvas.gettags(clicked_rectangle))
        self.canvas.lower(rect1)
        self.canvas.lower(rect2)
        self.canvas.delete(rectangle)
        self.canvas.delete(clicked_rectangle)
        self.left_clicked_object = None
        self.left_clicked_object_tk.set("")

    def add_transformation(self, transformation):
        element_tag = self.canvas.gettags(self.right_clicked_object)[0]
        element_text = element_tag.split("boxed_text_")[1]
        transformation_text = f"{transformation}({element_text})"
        if transformation_text not in self.transformation_list:
            new_elem_coords = [self.canvas.coords(self.right_clicked_object)[0], self.canvas.coords(self.right_clicked_object)[1] + 30]
            self.add_model_variables([transformation_text], [new_elem_coords])
            new_elem_coords[0] = new_elem_coords[0] - 50
            self.transformation_list.append(transformation_text)
        self.reset_click()

    def save_canvas_to_cache(self, model_id, panel_column, time_column):
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
        with open (f'{cet_home}/model_cache/{self.data_source}/{model_id}/tkinter_canvas.pkl', 'wb') as buff:
            pkl.dump({
                "data_source":self.data_source,
                "canvas_data":canvas_data,
                "transformation_list":self.transformation_list,
                "panel_column":panel_column,
                "time_column":time_column
            },buff)

    def restore_canvas_from_cache(self, model_id):
        cached_canvas = pd.read_pickle(f'{cet_home}/model_cache/{self.data_source}/{model_id}/tkinter_canvas.pkl')
        if cached_canvas["data_source"] != self.data_source:
            self.canvas_print_out.delete(1.0, tk.END)
            self.canvas_print_out.insert(tk.END, "Cached model is for a different data source. Please clear cache to use new dataset.")  
        else:
            self.clear_canvas()
            for item in cached_canvas["canvas_data"]:
                # process all rectanges/text before arrows
                if item["type"] == "rectangle":
                    rect = self.canvas.create_rectangle(*item["coords"], fill="orange", tags=item["tags"])
                elif item["type"] == "text":
                    self.canvas.create_text(*item["coords"], text=item["text"], fill="black", tags=item["tags"], font=("Helvetica", self.fontsize, "bold"))
                    text = item["text"]
                    column_box_tag = f"boxed_text_{text}".replace(" ","_")
                    self.add_tags_to_canvas_elements(column_box_tag)
            for item in cached_canvas["canvas_data"]:
                # now process all arrows
                if item["type"] == "line":
                    self.canvas.create_line(*item["coords"], arrow=tk.LAST, tags=item["tags"], width=self.arrow_width)
                    arrow_source, arrow_target = self.get_arrow_source_and_target(item["tags"])
                    self.arrow_list.append((arrow_source, arrow_target))
                    self.bind_right_click_to_arrow_tag(f"from_{self.canvas.gettags(arrow_source)[0]}")
            self.transformation_list = cached_canvas["transformation_list"]
            self.variables_displayed = True
            self.current_model = pd.read_pickle(f"{cet_home}/model_cache/{self.data_source}/{model_id}/model.pkl")

    def tags_are_arrow(self, element_tags):
        if (
            len(element_tags) >= 2 and 
            (element_tags[0].startswith("from_") and element_tags[1].startswith("to_")) or 
            (element_tags[0].startswith("to_") and element_tags[1].startswith("from_"))
        ):
            return True
        else:
            return False
        
    def color_clicked_rectangle(self, clicked_object, color):
        tags = self.canvas.gettags(clicked_object)
        clicked_rectangle = [elem for elem in self.canvas.find_withtag(tags[0]) if self.canvas.type(elem) == "rectangle"][0]
        self.canvas.itemconfig(clicked_rectangle, fill=color)

    def get_arrow_source_and_target(self, arrow_tags):
        if "from_" in arrow_tags[0]:
            source_tag = arrow_tags[0]
            target_tag = arrow_tags[1]
        else:
            source_tag = arrow_tags[1]
            target_tag = arrow_tags[0]
        arrow_source = [elem for elem in self.canvas.find_withtag(source_tag.split("from_")[1]) if self.canvas.type(elem) == "text"][0]
        arrow_target = [elem for elem in self.canvas.find_withtag(target_tag.split("to_")[1]) if self.canvas.type(elem) == "text"][0]
        return (arrow_source, arrow_target)
    
    def popup_menu(self, event):
        clicked_object = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags(clicked_object)
        if not self.tags_are_arrow(tags):
            self.right_clicked_object = clicked_object
            self.menu = self.get_menu(tags[0].split("boxed_text_")[1])
            try:
                self.menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.menu.grab_release()

    def add_tags_to_canvas_elements(self, column_box_tag):
        self.canvas.tag_bind(column_box_tag, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(column_box_tag, self.right_click_button, self.popup_menu)
        self.canvas.tag_bind(column_box_tag, "<Control-Button-1>", self.popup_menu)
        self.canvas.tag_bind(column_box_tag, "<Command-Button-1>", self.popup_menu)

    def add_model_variables(self, variables, coords=None):
        last_rectangle_right_side = 0
        row_count = 0
        for index, column in enumerate(variables):
            if coords != None:
                var_coords = coords[index]
            else:
                if last_rectangle_right_side > self.canvas.winfo_width() - 125:
                    row_count += 1
                    last_rectangle_right_side = 0
                var_coords = [last_rectangle_right_side + len(column)*5+50, row_count * 50 + 20]
            column_box_tag = f"boxed_text_{column}".replace(" ","_")
            text = self.canvas.create_text(*var_coords, text=column, fill="black", tags=column_box_tag, font=("Helvetica", self.fontsize, "bold"))
            rect = self.canvas.create_rectangle(self.canvas.bbox(text), fill="orange", tags=column_box_tag)
            self.canvas.lower(rect)
            self.add_tags_to_canvas_elements(column_box_tag)
            last_rectangle_right_side = self.canvas.bbox(text)[2]
        self.variables_displayed = True

    def handle_canvas_click(self, event):
        # if ctrl/command key isn't held
        if not (event.state & 0x4) or (event.state & 0x1000) or (event.state & 0x100000):
            clicked_object = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
            if self.menu != None:
                self.menu.unpost()
                self.menu = None
            if len(clicked_object) == 0:
                self.reset_click()
            else:
                self.on_click(event)

    def handle_canvas_release(self, event):
        # if ctrl/command key isn't held
        if not (event.state & 0x4) or (event.state & 0x1000) or (event.state & 0x100000):
            if self.in_drag:
                self.end_drag(event)
            else:
                clicked_objects = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
                if len(clicked_objects) > 0:
                    if self.left_clicked_object != None:
                        for obj in clicked_objects:
                            if self.canvas.type(obj) == "text":
                                self.draw_arrow(self.left_clicked_object, obj)
                                break
                    else:
                        for obj in clicked_objects:
                            if self.canvas.type(obj) == "text":
                                self.left_clicked_object = obj
                                self.left_clicked_object_tk.set(obj)
                                text_tag = self.canvas.gettags(obj)[0]
                                objs = self.canvas.find_withtag(text_tag)
                                for item in objs:
                                    if self.canvas.type(item) == "rectangle":
                                        self.color_clicked_rectangle(item, "red")
                                        break
                                break
                                
    def reset_click(self, reset_left_clicked_object=True):
        self.drag_start_x = None
        self.drag_start_y = None
        if self.left_clicked_object != None and reset_left_clicked_object:
            self.color_clicked_rectangle(self.left_clicked_object, 'orange')
            self.left_clicked_object = None
        self.right_clicked_object = None

    def end_drag(self, event):
        if not (event.state & 0x4) or (event.state & 0x1000) or (event.state & 0x100000):
            if self.in_drag:
                # if self.left_clicked_object == self.object_to_drag:
                #     self.reset_click()
                # else:
                self.reset_click(reset_left_clicked_object=False)
            self.in_drag = False
            self.object_to_drag = None
        
    def draw_arrow(self, source_object, target_object):
        arrow_conditions = [
            source_object != target_object,
            (source_object,target_object) not in self.arrow_list,
            (target_object,source_object) not in self.arrow_list,
            not self.tags_are_arrow(self.canvas.gettags(target_object)),
            self.canvas.type(source_object) == "text",
            self.canvas.type(target_object) == "text"
        ]
        if all(arrow_conditions):
            target_bb = self.canvas.bbox(target_object)
            source_bb = self.canvas.bbox(source_object)
            self.canvas.create_line(
                (source_bb[0] + source_bb[2]) / 2,
                (source_bb[1] + source_bb[3]) / 2,
                (target_bb[0] + target_bb[2]) / 2,
                (target_bb[1] + target_bb[3]) / 2,
                arrow=tk.LAST,
                tags=[
                    f"from_{self.canvas.gettags(source_object)[0]}",
                    f"to_{self.canvas.gettags(target_object)[0]}"
                ],
                width=self.arrow_width
            )
            self.bind_right_click_to_arrow_tag(f"from_{self.canvas.gettags(source_object)[0]}")
            self.arrow_list.append((source_object,target_object))
            self.reset_click()

    def clear_canvas(self):
        self.reset_click()
        self.arrow_list = []
        self.canvas.delete("all")
        self.variables_displayed = False
        self.transformation_list = []

    def delete_arrow_from_click(self, event):
        arrow = self.canvas.find_closest(event.x, event.y)[0]
        arrow_tags = self.canvas.gettags(arrow)
        if self.tags_are_arrow(arrow_tags): 
            self.arrow_list.remove(self.get_arrow_source_and_target(arrow_tags))
            self.canvas.delete(arrow)

    def update_arrow_coordinates(self, event, delta_x, delta_y):
        arrow_source_tags = f"from_{self.canvas.gettags(self.object_to_drag)[0]}"
        arrow_target_tags = f"to_{self.canvas.gettags(self.object_to_drag)[0]}"
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
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag(self, event):
        if self.object_to_drag == None:
            self.object_to_drag = self.canvas.find_closest(event.x, event.y)[0]
        self.in_drag = True
        canvas_buffer = 25
        if event.x >= canvas_buffer and event.y >= canvas_buffer and event.x <= self.canvas.winfo_width()-canvas_buffer and event.y <= self.canvas.winfo_height()-canvas_buffer:
            delta_x = event.x - self.drag_start_x
            delta_y = event.y - self.drag_start_y
            self.canvas.move(self.canvas.gettags(self.object_to_drag)[0], delta_x, delta_y)
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.update_arrow_coordinates(event, delta_x, delta_y)