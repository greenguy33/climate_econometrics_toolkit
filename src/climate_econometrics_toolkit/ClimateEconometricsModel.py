import numpy as np
import os
import pickle as pkl
from climate_econometrics_toolkit import utils

cet_home = os.getenv("CETHOME")

class ClimateEconometricsModel:

	attrib_list = [
		"target_var",
		"covariates",
		"fixed_effects",
		"random_effects",
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
		self.random_effects = None
		self.time_trends = []
		self.data_file = None
		self.full_data_path = None
		self.regression_result = None
		self.time_column = None
		self.panel_column = None
		self.dataset = None
		self.model_id = None

	def print(self):
		for val in self.attrib_list:
			print(val, ":", getattr(self, val), flush=True)

	def to_string(self):
		str = ""
		for val in self.attrib_list:
			str += f"{val} : {getattr(self, val)}\n"
		return str

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
		try:
			self.regression_result.summary2().tables[1].to_csv(f"{cet_home}/model_results/{self.model_id}.csv")
		except:
			self.regression_result.params.to_csv(f"{cet_home}/model_results/{self.model_id}.csv")
		

	def save_regression_script(self):
		# TODO: update according to new features: custom SEs, quantile, spatial, etc.
		demean_data = False
		if len(self.fixed_effects) > 0 and len(self.time_trends) == 0:
			demean_data = True

		covariates_as_string = "\",\"".join(self.covariates)
		fe_as_string = "\",\"".join(self.fixed_effects)

		script_text = "from climate_econometrics_toolkit import user_api as api\n"
		script_text += "from climate_econometrics_toolkit import utils as utils\n"
		script_text += "import statsmodels.formula.api as smf\n"
		script_text += "import statsmodels.api as sm\n\n"
		script_text += f"api.load_dataset_from_file(\"{self.full_data_path}\")\n"
		script_text += f"api.set_target_variable(\"{self.target_var}\", existence_check=False)\n"
		script_text += f"api.set_time_column(\"{self.time_column}\")\n"
		script_text += f"api.set_panel_column(\"{self.panel_column}\")\n"

		if len(self.covariates) > 0:
			script_text += f"api.add_covariates([\"{covariates_as_string}\"], existence_check=False)\n"
		if len(self.fixed_effects) > 0:
			script_text += f"api.add_fixed_effects([\"{fe_as_string}\"])\n"
		for tt in self.time_trends:
			tt_split = tt.split(" ")
			script_text += f"api.add_time_trend(\"{tt_split[0]}\", {tt_split[1]})\n"
		if self.random_effects is not None:
			script_text += f"api.add_random_effect(\"{self.random_effects[0]}\", \"{self.random_effects[1]}\")\n"

		script_text += ""

		script_text += f"transformed_data = api.transform_data(api.model.dataset, api.model, "
		if demean_data:
			script_text += "demean=True)\n"
		else:
			script_text += "demean=False)\n"

		if self.random_effects is not None:
			script_text += f"""
model_vars = utils.get_model_vars(transformed_data, api.model)
transformed_data.columns = [col.replace("(","_").replace(")","_") for col in transformed_data.columns]
model_vars = [var.replace("(","_").replace(")","_") for var in model_vars]
mv_as_string = "+".join(model_vars) if len(model_vars) > 0 else "0"
target_var = api.model.target_var.replace("(","_").replace(")","_")
formula = target_var + " ~ " + mv_as_string
reg = smf.mixedlm(formula, data=transformed_data, groups=api.model.random_effects[1], re_formula=f"0+{self.random_effects[0]}").fit()
print(reg.summary())
		"""
		else:
			script_text += f"""
model_vars = utils.get_model_vars(transformed_data, api.model)
regression_data = transformed_data[model_vars]
regression_data = sm.add_constant(regression_data)
reg = sm.OLS(transformed_data["{self.target_var}"],regression_data,missing="drop").fit()
print(reg.summary2().tables[1])
		"""

		file = open(f"{cet_home}/regression_scripts/{self.model_id}.py", "w")
		file.write(script_text)
		file.close()