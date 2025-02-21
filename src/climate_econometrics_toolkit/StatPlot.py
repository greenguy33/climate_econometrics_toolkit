import tkinter as tk
import pandas as pd

class StatPlot():

    def __init__(self, mse_canvas, pred_int_canvas, r2_canvas, rmse_canvas):
        self.mse_canvas = mse_canvas
        self.pred_int_canvas = pred_int_canvas
        self.r2_canvas = r2_canvas
        self.rmse_canvas = rmse_canvas

    def clear_stat_plot(self):
        self.mse_canvas.delete("all")
        self.pred_int_canvas.delete("all")
        self.r2_canvas.delete("all")
        self.rmse_canvas.delete("all")

    def get_adjusted_box_coords(self, box_coords, x_offset=0, y_offset=0):
        return (box_coords[0]+x_offset, box_coords[1]+y_offset, box_coords[2]+x_offset, box_coords[3]+y_offset)

    def update_stat_plot(self, mse, pred_int_cov, r2, rmse):
        self.clear_stat_plot()
        mse_string = '%.2f' % (mse * 100) + "%"
        if not pd.isnull(pred_int_cov):
            pred_int_cov_string = '%.2f' % (pred_int_cov * 100) + "%"
        else:
            pred_int_cov_string = "NaN"
        if not pd.isnull(rmse):
            rmse_string = '%.2f' % rmse
        else:
            rmse_string = "NaN"
        self.mse_canvas.create_text(self.mse_canvas.winfo_width()/2, self.mse_canvas.winfo_height()/2-20, text="Mean Squared Error Reduction %")
        self.pred_int_canvas.create_text(self.mse_canvas.winfo_width()/2, self.mse_canvas.winfo_height()/2-20, text="Prediction Interval Coverage %")
        self.r2_canvas.create_text(self.r2_canvas.winfo_width()/2, self.r2_canvas.winfo_height()/2-20, text="R^2")
        self.rmse_canvas.create_text(self.rmse_canvas.winfo_width()/2, self.rmse_canvas.winfo_height()/2-20, text="Out-of-Sample RMSE")
        mse_text = self.mse_canvas.create_text(self.mse_canvas.winfo_width()/2, self.mse_canvas.winfo_height()/2+20, text=mse_string, font=("Helvetica", 25))
        pred_int_text = self.pred_int_canvas.create_text(self.pred_int_canvas.winfo_width()/2, self.pred_int_canvas.winfo_height()/2+20, text=pred_int_cov_string, font=("Helvetica", 25))
        r2_text = self.r2_canvas.create_text(self.r2_canvas.winfo_width()/2, self.r2_canvas.winfo_height()/2+20, text=r2, font=("Helvetica", 25))
        rmse_text = self.rmse_canvas.create_text(self.rmse_canvas.winfo_width()/2, self.rmse_canvas.winfo_height()/2+20, text=rmse_string, font=("Helvetica", 25))
        mse_box_color = "green" if mse > 0 else "red"
        if pd.isnull(pred_int_cov):
            pred_int_box_color = "gray"
        else:
            pred_int_box_color = "red"
            if pred_int_cov < .96 and pred_int_cov > .94:
                pred_int_box_color = "yellow"
            if pred_int_cov < .951 and pred_int_cov > .949:
                pred_int_box_color = "green"
        
        mse_box_coords = self.get_adjusted_box_coords(self.mse_canvas.bbox(mse_text), y_offset=-4)
        pred_int_box_coords = self.get_adjusted_box_coords(self.pred_int_canvas.bbox(pred_int_text), y_offset=-4)
        r2_box_coords = self.get_adjusted_box_coords(self.r2_canvas.bbox(r2_text), y_offset=-4)
        rmse_box_coords = self.get_adjusted_box_coords(self.rmse_canvas.bbox(rmse_text), y_offset=-4)
        
        mse_rect = self.mse_canvas.create_rectangle(mse_box_coords, fill=mse_box_color)
        pred_int_rect = self.pred_int_canvas.create_rectangle(pred_int_box_coords, fill=pred_int_box_color)
        r2_rect = self.r2_canvas.create_rectangle(r2_box_coords, fill="gray")
        rmse_rect = self.rmse_canvas.create_rectangle(rmse_box_coords, fill="gray")
        self.mse_canvas.lower(mse_rect)
        self.pred_int_canvas.lower(pred_int_rect)
        self.r2_canvas.lower(r2_rect)
        self.rmse_canvas.lower(rmse_rect)
        return [self.mse_canvas, self.pred_int_canvas, self.r2_canvas, self.rmse_canvas]