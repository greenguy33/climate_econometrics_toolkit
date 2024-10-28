import tkinter as tk

class StatPlot():

    def __init__(self, mse_canvas, pred_int_canvas):
        self.mse_canvas = mse_canvas
        self.pred_int_canvas = pred_int_canvas

    def clear_stat_plot(self):
        self.mse_canvas.delete("all")
        self.pred_int_canvas.delete("all")

    def update_stat_plot(self, mse, pred_int_cov):
        self.clear_stat_plot()
        mse_string = '%.2f' % (mse * 100) + "%"
        pred_int_cov_string = '%.2f' % (pred_int_cov * 100) + "%"
        self.mse_canvas.create_text(self.mse_canvas.winfo_width()/2, self.mse_canvas.winfo_height()/2-20, text="Mean Squared Error Reduction %")
        self.pred_int_canvas.create_text(self.mse_canvas.winfo_width()/2, self.mse_canvas.winfo_height()/2-20, text="Prediction Interval Coverage %")
        mse_text = self.mse_canvas.create_text(self.mse_canvas.winfo_width()/2, self.mse_canvas.winfo_height()/2+20, text=mse_string, font=("Helvetica", 25))
        pred_int_text = self.pred_int_canvas.create_text(self.pred_int_canvas.winfo_width()/2, self.pred_int_canvas.winfo_height()/2+20, text=pred_int_cov_string, font=("Helvetica", 25))
        mse_box_color = "green" if mse > 0 else "red"
        pred_int_box_color = "red"
        if pred_int_cov < .96 and pred_int_cov > .94:
            pred_int_box_color = "yellow"
        if pred_int_cov < .951 and pred_int_cov > .949:
            pred_int_box_color = "green"
        mse_rect = self.mse_canvas.create_rectangle(self.mse_canvas.bbox(mse_text), fill=mse_box_color)
        pred_int_rect = self.pred_int_canvas.create_rectangle(self.pred_int_canvas.bbox(pred_int_text), fill=pred_int_box_color)
        self.mse_canvas.lower(mse_rect)
        self.pred_int_canvas.lower(pred_int_rect)