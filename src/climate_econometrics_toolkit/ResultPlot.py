class ResultPlot():

    def __init__(self, plot_frame):
        self.plot_frame = plot_frame
        self.plot_canvas = None
        self.plot_data = []
        self.circles = []
        self.models = []

    def clear_figure(self):
        if self.plot_canvas != None:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_data = []
            self.circles = []
            self.models = []
