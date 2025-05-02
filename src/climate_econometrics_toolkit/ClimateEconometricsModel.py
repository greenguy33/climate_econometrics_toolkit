import numpy as np
import pandas as pd
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
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
			with open(f"{dir_name}/model.pkl", "wb") as write_file:
				pkl.dump(self, write_file)
		else:
			utils.print_with_log(f"Cached model with ID {self.model_id} already exists; will not overwrite.", "warning")


	def save_result_to_file(self):
		try:
			# handle non-random-effects model
			self.regression_result.summary2().tables[1].to_csv(f"{cet_home}/OLS_output/{self.model_id}.csv")
		except:
			try:
				# handle random-effects model
				res = self.regression_result.summary().tables[1]
				dir_name = f"{cet_home}/OLS_output/{self.model_id}"
				if not os.path.exists(dir_name):
					os.mkdir(dir_name)
					res.to_csv(f"{dir_name}/covariates.csv")
					np.transpose(pd.DataFrame.from_dict(self.regression_result.random_effects)).to_csv(f"{dir_name}/random_effects.csv")
				else:
					utils.print_with_log(f"OLS results file for model with ID {self.model_id} already exists; will not overwrite.", "warning")
			except:
				res = self.regression_result.summary
				# handle driscoll-kraay model
				filepath = f"{cet_home}/OLS_output/{self.model_id}.csv"
				if not os.path.exists(filepath):
					file = open(filepath, "w")
					file.write(str(res))
					file.close()
				else:
					utils.print_with_log(f"OLS results file for model with ID {self.model_id} already exists; will not overwrite.", "warning")


	def build_model_as_string(self, script_text):

		covariates_as_string = "\",\"".join(self.covariates)
		fe_as_string = "\",\"".join(self.fixed_effects)

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
		return script_text


	def save_quantile_regression_script(self, std_error_type, q):
		demean_data = False
		if len(self.fixed_effects) > 0 and len(self.time_trends) == 0:
			demean_data = True
		script_text = f"""
from climate_econometrics_toolkit import user_api as api
from climate_econometrics_toolkit import utils as utils
import statsmodels.api as sm

{self.build_model_as_string("")}

transformed_data = api.transform_data(api.current_model, demean={str(demean_data)})
model_vars = utils.get_model_vars(transformed_data, api.current_model, exclude_fixed_effects={str(demean_data)})
regression_data = transformed_data[model_vars]
regression_data = sm.add_constant(regression_data)
quant_reg_model = sm.QuantReg(transformed_data["{self.target_var}"], regression_data).fit(q={str(q)}, vcov="{utils.quantile_std_error_map[std_error_type]}")
print(quant_reg_model.summary())		
"""
		file = open(f"{cet_home}/regression_scripts/{self.model_id}_quantile_{q}.py", "w")
		file.write(script_text)
		file.close()


	def save_spatial_regression_script(self, reg_type, k, demean_data):
		script_text = "from climate_econometrics_toolkit import user_api as api\n"
		script_text += "from climate_econometrics_toolkit import utils as utils\n"
		script_text += "from spreg import Panel_FE_Error, Panel_FE_Lag\n"
		script_text += "from libpysal.weights import distance\n"
		script_text += "import copy\n"
		script_text += "import pandas as pd\n"
		script_text += "import numpy as np\n"
		script_text += "from shapely.wkt import loads\n"
		script_text += "import geopandas as gpd\n\n"

		script_text += f"""
{self.build_model_as_string("")}

transformed_data = api.transform_data(api.current_model, demean={str(demean_data)})
model_vars = utils.get_model_vars(transformed_data, api.current_model, exclude_fixed_effects={str(demean_data)})
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['iso_a3', 'geometry']]
countries = countries[countries['iso_a3'].isin(transformed_data["{self.panel_column}"])]
transformed_data = transformed_data[transformed_data["{self.panel_column}"].isin(countries.iso_a3)].reset_index(drop=True)

transformed_data = transformed_data.set_index(["{self.panel_column}","{self.time_column}"])
columns = copy.deepcopy(model_vars)
columns.append("{self.target_var}")
# choose whether to remove nans by georef or time based on which scenario leads to more remaining data
td_rm_ax0 = transformed_data[columns].unstack().dropna(axis=0)
td_rm_ax1 = transformed_data[columns].unstack().dropna(axis=1)
axis0_size = td_rm_ax0.shape[0] * td_rm_ax0.shape[1]
axis1_size = td_rm_ax1.shape[0] * td_rm_ax1.shape[1]
if axis0_size > axis1_size:
	regression_data = td_rm_ax0
else:
	regression_data = td_rm_ax1

country_geo = map(lambda country: countries.loc[countries.iso_a3 == country].geometry.item(), regression_data.index)

W = distance.KNN.from_dataframe(pd.DataFrame([regression_data.index, list(country_geo)]).T.rename(columns="""+"""{0:\""""+self.panel_column+"""\",1:"geometry"}),"""+f"""k={k})
W.transform = "r"
"""

		if reg_type == "error":
			script_text += f"""
spatial_reg_model = Panel_FE_Error(
	y=np.array(regression_data["{self.target_var}"]), 
	x=np.array(regression_data[model_vars]),
	w=W,
	name_x=model_vars,
	name_y="{self.target_var}"
)
"""
		else:
			script_text += f"""
spatial_reg_model = Panel_FE_Lag(
	y=np.array(regression_data["{self.target_var}"]), 
	x=np.array(regression_data[model_vars]),
	w=W,
	name_x=model_vars,
	name_y="{self.target_var}"
)
"""
		script_text += """
print(spatial_reg_model.summary)
"""
		file = open(f"{cet_home}/regression_scripts/{self.model_id}_spatial.py", "w")
		file.write(script_text)
		file.close()


	def save_OLS_regression_script(self, std_error_type):

		script_text = "from climate_econometrics_toolkit import user_api as api\n"
		script_text += "from climate_econometrics_toolkit import utils as utils\n"
		if std_error_type == "driscollkraay" and self.random_effects is None:
			script_text += "from linearmodels import PanelOLS\n"
		elif self.random_effects is not None:
			script_text += "import statsmodels.formula.api as smf\n"
		script_text += "import statsmodels.api as sm\n\n"

		if self.random_effects is not None:
			script_text += f"""

{self.build_model_as_string("")}
transformed_data = api.transform_data(api.current_model, demean=False)
model_vars = utils.get_model_vars(transformed_data, api.current_model, exclude_fixed_effects=True)
transformed_data.columns = [col.replace("(","_").replace(")","_") for col in transformed_data.columns]
model_vars = [var.replace("(","_").replace(")","_") for var in model_vars]
target_var = "{self.target_var}".replace("(","_").replace(")","_")
re_for_model = "{self.random_effects[0]}".replace("(","_").replace(")","_")
mv_as_string = "+".join(model_vars) if len(model_vars) > 0 else "0"
formula = target_var + " ~ " + mv_as_string
random_effects_formatted = api.current_model.random_effects[0].replace("(","_").replace(")","_")
"""
			if len(self.fixed_effects) > 0:
				script_text += """
for fe in api.current_model.fixed_effects:
	formula += " + C("+fe+")"
reg = smf.mixedlm(formula, data=transformed_data, groups=api.current_model.random_effects[1], re_formula=f"0+{random_effects_formatted}").fit()
"""
			else:
				script_text += """
reg = smf.mixedlm(formula, data=transformed_data, groups=api.current_model.random_effects[1], re_formula=f"1+{random_effects_formatted}").fit()
"""
			script_text += """
print(reg.summary())
"""
		else:
			demean_data = False
			if len(self.fixed_effects) > 0 and len(self.time_trends) == 0:
				demean_data = True
			script_text += self.build_model_as_string("")
			script_text += f"transformed_data = api.transform_data(api.current_model, demean={str(demean_data)})\n"
			if std_error_type != "driscollkraay":
				if std_error_type not in utils.std_error_args:
					reg_string = f"""reg = sm.OLS(transformed_data["{self.target_var}"],regression_data,missing="drop").fit(cov_type="{utils.std_error_name_map[std_error_type]}")\n"""
				else:
					if std_error_type == "neweywest":
						cov_kwds_string = """{"maxlags":3}"""
					elif std_error_type == "clusteredtime":
						cov_kwds_string = """{"groups":transformed_data[\""""+self.time_column+"""\"]}"""
					elif std_error_type == "clusteredspace":
						cov_kwds_string = """{"groups":transformed_data[\""""+self.panel_column+"""\"]}"""
					reg_string = f"""reg = sm.OLS(transformed_data["{self.target_var}"],regression_data,missing="drop").fit(cov_type="{utils.std_error_name_map[std_error_type]}", cov_kwds={cov_kwds_string})\n"""
			else:
				reg_string = f"""reg = PanelOLS(transformed_data["{self.target_var}"], regression_data, check_rank=False).fit(cov_type="{utils.std_error_name_map[std_error_type]}")\n"""
			script_text += f"""
model_vars = utils.get_model_vars(transformed_data, api.current_model, exclude_fixed_effects={str(demean_data)})"""
			if std_error_type == "driscollkraay":
				script_text += f"""
transformed_data = transformed_data.set_index([\"{self.panel_column}\", \"{self.time_column}\"])"""
			script_text += f"""
regression_data = transformed_data[model_vars]
regression_data = sm.add_constant(regression_data)
{reg_string}
"""
			if std_error_type != "driscollkraay":
				script_text += """
print(reg.summary2().tables[1])
"""
			else:
				script_text += """
print(reg.params)
print(reg.std_errors)
"""

		file = open(f"{cet_home}/regression_scripts/{self.model_id}_OLS.py", "w")
		file.write(script_text)
		file.close()
