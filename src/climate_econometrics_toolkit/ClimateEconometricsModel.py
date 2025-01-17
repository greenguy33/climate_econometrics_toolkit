import numpy as np
import os
import pickle as pkl

cet_home = os.getenv("CETHOME")

class ClimateEconometricsModel:

	attrib_list = [
		"target_var",
		"covariates",
		"fixed_effects",
		"time_trends",
		"time_column",
		"panel_column",
		"out_sample_mse",
		"out_sample_mse_reduction",
		"out_sample_pred_int_cov",
		"r2",
		"rmse",
		"model_id"	
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
		self.r2 = np.NaN
		self.rmse = np.NaN
		self.fixed_effects = []
		self.time_trends = []
		self.data_file = None
		self.regression_result = None
		self.time_column = None
		self.panel_column = None
		self.dataset = None
		self.model_id = None

	def print(self):
		for val in self.attrib_list:
			print(val, ":", getattr(self, val), flush=True)

	def is_empty(self):
		if self.model_vars == []:
			return True
		else:
			return False

	def save_model_to_cache(self):
		dir_name = f"{cet_home}/model_cache/{self.data_file}/{self.model_id}"
		os.makedirs(dir_name)
		with open(f"{dir_name}/model.pkl", "wb") as write_file:
			pkl.dump(self, write_file)
