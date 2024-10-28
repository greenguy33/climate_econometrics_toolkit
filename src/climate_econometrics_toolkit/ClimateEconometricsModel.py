import numpy as np
import os
import time
import csv

class ClimateEconometricsModel:

	attrib_list = [
		"target_var",
		"covariates",
		"fixed_effects",
		"time_trends",
		"out_sample_mse",
		"in_sample_mse",
		"out_sample_mse_reduction",
		"in_sample_mse_reduction",
		"out_sample_pred_int_cov",
		"in_sample_pred_int_cov",
		"dataset",
		"regression_result"
	]

	def __init__(self):
		self.target_var = np.NaN
		self.covariates = []
		self.model_vars = []
		self.out_sample_mse = np.NaN
		self.in_sample_mse = np.NaN
		self.out_sample_mse_reduction = np.NaN
		self.in_sample_mse_reduction = np.NaN
		self.out_sample_pred_int_cov = np.NaN
		self.in_sample_pred_int_cov = np.NaN
		self.fixed_effects = np.NaN
		self.time_trends = np.NaN
		self.dataset = None
		self.regression_result = None
		self.time_column = None
		self.panel_column = None

	def print(self):
		for val in self.attrib_list:
			print(val, ":", getattr(self, val), flush=True)

	def is_empty(self):
		if self.model_vars == []:
			return True
		else:
			return False

	def save_model_to_cache(self):
		# TODO: make this file path more flexible
		time_based_id = time.time()
		dir_name = f"model_cache/{self.dataset}/{time_based_id}"
		os.makedirs(dir_name)
		with open(f"{dir_name}/model.csv", "w") as write_file:
			writer = csv.writer(write_file)
			writer.writerow(["model_attribute","attribute_value"])
			for val in self.attrib_list:
				writer.writerow([val, getattr(self, val)])
		return time_based_id