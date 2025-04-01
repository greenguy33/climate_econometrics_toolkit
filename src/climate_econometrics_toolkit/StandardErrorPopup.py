import tkinter as tk

class StandardErrorPopup(tk.Toplevel):

	function = None

	def __init__(self, window):

		popup = tk.Toplevel()
		popup.geometry("500x300")
		popup.transient(window)

		def on_close():
			self.std_error_type = std_error_type.get()
			popup.destroy()

		popup.protocol("WM_DELETE_WINDOW", on_close)

		std_error_list = ["Nonrobust","White-Huber","Driscoll-Kraay","Newey-West","Time-clustered","Space-clustered"]

		std_error_type = tk.StringVar(value=std_error_list[0])

		radio_button_label = tk.Label(popup, text="Choose a type of standard error to be estimated with the regression:")
		radio_button_label.grid(row=0, column=0, padx=5, pady=1)
		for index, method in enumerate(std_error_list):
			button = tk.Radiobutton(popup, text=method, variable=std_error_type, value=method)
			button.grid(row=index+1, column=0, padx=5, pady=1)
		
		window.wait_window(popup)
