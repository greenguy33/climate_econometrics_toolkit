from climate_econometrics_toolkit import utils

import tkinter as tk

class GcmSelectionPopup(tk.Toplevel):

    def update_checkbox_state(self):
        for gcm in utils.supported_gcms:
            if self.gcm_checkbox_state[gcm].get():
                self.gcms_to_use.add(gcm)
            else:
                if gcm in self.gcms_to_use:
                    self.gcms_to_use.remove(gcm)

    def __init__(self, window):
        self.gcms_to_use = set()
        self.gcm_checkbox_state = {}
        popup = tk.Toplevel()
        for index, gcm in enumerate(utils.supported_gcms):
            var = tk.BooleanVar()
            cb = tk.Checkbutton(popup, text=gcm, variable=var, command=self.update_checkbox_state)
            cb.grid(row=index, column=0, padx=5, pady=1)
            self.gcm_checkbox_state[gcm] = var
        window.wait_window(popup)