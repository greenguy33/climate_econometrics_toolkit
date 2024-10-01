import numpy as np

class ClimateEconometricsModel:

	attrib_list = [
		"target_var",
		"covariates",
		"fixed_effects",
		"incremental_effects",
		"out_sample_mse",
		"in_sample_mse",
		"out_sample_mse_reduction",
		"in_sample_mse_reduction",
		"out_sample_pred_int_acc",
		"in_sample_pred_int_acc"
	]

	def __init__(self):
		self.target_var = np.NaN
		self.covariates = []
		self.model_vars = []
		self.out_sample_mse = np.NaN
		self.in_sample_mse = np.NaN
		self.out_sample_mse_reduction = np.NaN
		self.in_sample_mse_reduction = np.NaN
		self.out_sample_pred_int_acc = np.NaN
		self.in_sample_pred_int_acc = np.NaN
		self.fixed_effects = np.NaN
		self.incremental_effects = np.NaN

	def print(self):
		for val in self.attrib_list:
			print(val, ":", getattr(self, val), flush=True)

	def is_empty(self):
		if self.model_vars == []:
			return True
		else:
			return False

	# def save_model_to_file(self):
	# 	with open(f"output/models/{self.target_var}_best_model_from_grid_search.csv", "w") as write_file:
	# 		writer = csv.writer(write_file)
	# 		for val in self.attrib_list:
	# 			writer.writerow([val, getattr(self, val)])