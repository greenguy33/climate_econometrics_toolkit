import os
import pandas as pd
import random
import numpy as np
import threading
from exactextract import exact_extract
import geopandas as gpd

import climate_econometrics_toolkit.utils as utils
import climate_econometrics_toolkit.regression as regression

cet_home = os.getenv("CETHOME")

def predict_from_gcms(model, gcms_to_use, vars_to_use, groups_to_use, use_threading=False):
	if use_threading:
		thread = threading.Thread(target=predict,name="prediction_thread",args=(model, gcms_to_use, vars_to_use, groups_to_use))
		thread.daemon = True
		thread.start()
	else:
		predict(model, gcms_to_use, vars_to_use, groups_to_use)


def aggregate_gcm_data(
		gcm_file, 
		shape_file,
		gcm_obs_per_year, 
		first_year_in_data,
		shape_file_geo_identifier,
		obs_to_include,
		aggregation_func,
		climate_var_name,
		weights_file=None
	):
	# TODO: this is only working for aggregations to the year level now
	assert aggregation_func == "sum" or aggregation_func == "mean", "Argument aggregation_func must be 'sum' or 'mean'"
	out = exact_extract(gcm_file, shape_file, ["mean"], weights=weights_file)
	data = []
	geo_shapes = gpd.read_file(shape_file)
	for index, geo in enumerate(geo_shapes[shape_file_geo_identifier]):
		year = first_year_in_data
		agg_mean = []
		period = 0
		for obs in range(len(out[index]["properties"])):
			period += 1
			if obs_to_include is None or (geo in obs_to_include and period in obs_to_include[geo]):
				agg_mean.append(out[index]["properties"][f"band_{str(obs+1)}_mean"])
			if period == gcm_obs_per_year:
				if aggregation_func == "sum":
					if len(agg_mean) > 0:
						data.append([geo, year, np.nansum(agg_mean)])
					else:
						data.append([geo, year, np.NaN])
				elif aggregation_func == "mean":
					data.append([geo, year, np.nanmean(agg_mean)])
				year += 1
				agg_mean = []
				period = 0
	return pd.DataFrame.from_records(data, columns=[shape_file_geo_identifier,"year",climate_var_name])
	

def predict(model, gcms_to_use, vars_to_use, groups_to_use, gcm_to_model_var_map = {}):

	random.setstate = utils.random_state

	gcm_groups = {}
	gcm_dir = cet_home + "/processed_gcm_data"
	for var in os.listdir(gcm_dir):
		if vars_to_use == "all" or var in vars_to_use:
			for gcm in os.listdir(gcm_dir + "/" + var):
				if gcms_to_use == "all" or gcm in gcms_to_use:
					for group in os.listdir(gcm_dir + "/" + var + "/" + gcm):
						if groups_to_use == "all" or group in groups_to_use:
							if group not in gcm_groups:
								gcm_groups[group] = {}
							if gcm not in gcm_groups[group]:
								gcm_groups[group][gcm] = []
							gcm_groups[group][gcm].append(var)

	gcm_data_store = {}

	for group, gcms in gcm_groups.items():
		gcm_data_store[group] = {}
		for gcm, var_list in gcms.items():
			gcm_data = pd.DataFrame()
			for var in var_list:
				gcm_dir = f"{cet_home}/processed_gcm_data/{var}/{gcm}/{group}"
				var_data = pd.read_csv(gcm_dir + "/" + os.listdir(gcm_dir)[0])
				if model.time_column not in gcm_to_model_var_map.values():
					gcm_data[model.time_column] = var_data[model.time_column]
				else:
					gcm_data[model.time_column] = var_data[[key for key, value in gcm_to_model_var_map.items() if value == model.time_column]]
				if model.panel_column not in gcm_to_model_var_map.values():
					gcm_data[model.panel_column] = var_data[model.panel_column]
				else:
					gcm_data[model.panel_column] = var_data[[key for key, value in gcm_to_model_var_map.items() if value == model.panel_column]]
				if var in gcm_to_model_var_map:
					gcm_data[gcm_to_model_var_map[var]] = var_data[var]
				else:
					gcm_data[var] = var_data[var]
			gcm_data = utils.transform_data(gcm_data, model, include_target_var=False)
			gcm_data_store[group][gcm] = gcm_data

	# remove values of panel and time variables that aren't shared between all GCMs
	common_time_vals = set()
	common_geo_vals = set()
	for group, gcms in gcm_data.items():
		for _, gcm_data in gcms.items():
			if len(common_time_vals) == 0:
				common_time_vals = set(gcm_data[model.time_column])
			else:
				for time_val in common_time_vals:
					if time_val not in set(gcm_data[model.time_column]):
						common_time_vals.remove(time_val)
			if len(common_geo_vals) == 0:
				common_geo_vals = set(gcm_data[model.panel_column])
			else:
				for geo_val in common_geo_vals:
					if geo_val not in set(gcm_data[model.panel_column]):
						common_geo_vals.remove(geo_val)
	for group, gcms in gcm_data.items():
		for _, gcm_data in gcms.items():
			gcm_data = gcm_data[gcm_data[model.time_column in common_time_vals]]
			gcm_data = gcm_data[gcm_data[model.panel_column in common_geo_vals]]

	bayesian_results = os.path.isdir(f"{cet_home}/bayes_samples/{model.model_id}")
	bootstrap_results = os.path.isdir(f"{cet_home}/bootstrap_samples/{model.model_id}")

	# TODO: these will never trigger if calling from the interface because a new model_is assigned
	if bayesian_results:
		coef_samples = pd.read_csv(f"{cet_home}/bayes_samples/{model.model_id}/coefficient_samples_{model.model_id}.csv")
		print("Using Bayesian samples to generate predictions...")
	elif bootstrap_results:
		coef_samples = pd.read_csv(f"{cet_home}/bootstrap_samples/{model.model_id}/coefficient_samples_{model.model_id}.csv")
		print("Using bootstrap samples to generate predictions...")
	else:
		print("No Bayesian or bootstrap samples found...using point estimates to generate predictions...")
		transformed_data = utils.transform_data(model.dataset, model)
		reg_result = regression.run_standard_regression(transformed_data, model).summary2().tables[1]
		coef_map = {covar:[reg_result.loc[reg_result.index == covar]["Coef."].item()] for covar in reg_result.index}
		coef_samples = pd.DataFrame.from_dict(coef_map)

	pred_dict = {model.panel_column:[], model.time_column:[], model.target_var:[]}

	for group, gcm_data in gcm_data_store:
		if len(gcm_data) == 1 and len(coef_samples) == 1:
			gcm_dataset = list(gcm_data.values())[0]
			predictions = np.sum([gcm_dataset[covar] * coef_samples[covar] for covar in model.covariates], axis=1)
	
		else:
			num_samples = 1000
			if len(gcms) > 1:
				gcm_samples = random.choices(list(gcm_data.keys()), k=num_samples)
			else:
				gcm_samples = list(gcm_data.keys())[0] * num_samples
			if len(coef_samples) > 1:
				coef_samples = random.choices(coef_samples, k=num_samples)
			else:
				coef_samples = pd.DataFrame(np.repeat(coef_samples.values, num_samples, axis=0), columns=coef_samples.columns)

			predictions = []
			for i in range(num_samples):
				pred = np.sum(gcm_data[gcm_samples[i]].iloc[i][model.covariates] * coef_samples.iloc[i][model.covariates], axis=1)
				predictions.append(pred)

		pred_dict[model.panel_column] = list(gcm_data.values())[0][model.panel_column]
		pred_dict[model.time_column] = list(gcm_data.values())[0][model.time_column]
		pred_dict[model.target_var] = predictions

		pd.DataFrame.from_dict(pred_dict).to_csv(f"{cet_home}/predictions/predictions_{model.model_id}_{group}")