import tkinter as tk

from climate_econometrics_toolkit import user_prediction_functions as user_predict

class PredictionFunctionPopup(tk.Toplevel):

	function = None

	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("500x300")
		popup.transient(window)

		def on_close():
			self.function = function.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		method_list = ["None"]
		method_list_from_user_predict = [method for method in dir(user_predict) if not method.startswith("__") and not method == "np" and not method == "pd"]
		method_list.extend(method_list_from_user_predict)

		function = tk.StringVar(value=method_list[0])

		radio_button_label = tk.Label(popup, text="Optionally choose a function to apply to the predictions.")
		radio_button_label.grid(row=0, column=0, padx=5, pady=1)
		for index, method in enumerate(method_list):
			button = tk.Radiobutton(popup, text=method, variable=function, value=method)
			button.grid(row=index+1, column=0, padx=5, pady=1)
		
		window.wait_window(popup)
