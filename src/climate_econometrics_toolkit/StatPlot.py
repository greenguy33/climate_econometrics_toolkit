import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class StatPlot():

    def __init__(self, plot_frame):
        self.plot_frame = plot_frame
        self.plot_canvas = None

    def update_stat_plot(self, mse, pred_int_cov):
        
        fig, axes = plt.subplots(1,2,figsize=(5,5))

        axes[0].text(.25, .25, "MSE reduction:" + str(mse),
                horizontalalignment='left',
                verticalalignment='top',
                transform=axes[0].transAxes)

        axes[1].text(.25, .25, "Prediction interval coverage:" + str(pred_int_cov),
                horizontalalignment='left',
                verticalalignment='top',
                transform=axes[0].transAxes)

        # ax.set_axis_off()

        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)