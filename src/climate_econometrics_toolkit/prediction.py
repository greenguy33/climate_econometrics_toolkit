import os
import pandas as pd
import random
import numpy as np
from exactextract import exact_extract
import geopandas as gpd

import climate_econometrics_toolkit.utils as utils
import climate_econometrics_toolkit.regression as regression

cet_home = os.getenv("CETHOME")

def extract_raster_data(gcm_file, shape_file, weight_file):
	aggregation_func = "weighted_mean"
	if weight_file is None:
		print("No weights file provided for extraction...using uniform weights.")
		aggregation_func = "mean"
	return exact_extract(gcm_file, shape_file, [aggregation_func], weights=weight_file)

def aggregate_raster_data(
		raster_data, shape_file, climate_var_name, aggregation_func, geo_identifier, subperiods_per_time_unit, months_to_use, 
	):
	assert isinstance(subperiods_per_time_unit, int)
	assert aggregation_func == "sum" or aggregation_func == "mean", "Argument aggregation_func must be 'sum' or 'mean'"
	data = []
	geo_shapes = gpd.read_file(shape_file)
	for index, geo in enumerate(geo_shapes[geo_identifier]):
		# this removes the name of the aggregation function from the key
		new_dict = {}
		for key in raster_data[index]["properties"]:
			new_dict[key.split("_")[0] + "_" + key.split("_")[1]] = raster_data[index]["properties"][key]
		period = 0
		agg_mean = []
		subperiod = 0
		for obs in range(len(raster_data[index]["properties"])):
			subperiod += 1
			if months_to_use is None or (geo in months_to_use and subperiod in months_to_use[geo]):
				agg_mean.append(new_dict[f"band_{str(obs+1)}"])
			if subperiod == subperiods_per_time_unit:
				if aggregation_func == "sum":
					if len(agg_mean) > 0:
						data.append([geo, period, np.nansum(agg_mean)])
					else:
						data.append([geo, period, np.NaN])
				elif aggregation_func == "mean":
					data.append([geo, period, np.nanmean(agg_mean)])
				period += 1
				agg_mean = []
				subperiod = 0
	return pd.DataFrame.from_records(data, columns=[geo_identifier,"time",climate_var_name])


def predict_out_of_sample(model, out_sample_data, transform_data, gcm_to_model_var_map):
	
	if transform_data:
		if not all(var in out_sample_data.columns for var in model.covariates):
			out_sample_data = utils.transform_data(out_sample_data, model, include_target_var=False)

	bayesian_results = os.path.exists(f"{cet_home}/bayes_samples/coefficient_samples_{model.model_id}.csv")
	bootstrap_results = os.path.exists(f"{cet_home}/bootstrap_samples/coefficient_samples_{model.model_id}.csv")

	out_sample_data = out_sample_data.dropna().reset_index(drop=True)
	
	pred_df = pd.DataFrame()
	if gcm_to_model_var_map is None or model.time_column not in gcm_to_model_var_map.values():
		pred_df[model.panel_column] = out_sample_data[model.panel_column]
	else:
		pred_df[model.panel_column] = out_sample_data[[key for key, value in gcm_to_model_var_map.items() if value == model.panel_column]]
	if gcm_to_model_var_map is None or model.time_column not in gcm_to_model_var_map.values():
		pred_df[model.time_column] = out_sample_data[model.time_column]
	else:
		pred_df[model.time_column] = out_sample_data[[key for key, value in gcm_to_model_var_map.items() if value == model.time_column]]

	if bayesian_results or bootstrap_results:
		if bayesian_results:
			coef_samples = pd.read_csv(f"{cet_home}/bayes_samples/coefficient_samples_{model.model_id}.csv")
			print("Using Bayesian samples to generate predictions...")
		elif bootstrap_results:
			coef_samples = pd.read_csv(f"{cet_home}/bootstrap_samples/coefficient_samples_{model.model_id}.csv")
			print("Using bootstrap samples to generate predictions...")
		predictions = []
		for i in range(len(coef_samples)):
			pred = np.sum(out_sample_data[model.covariates] * coef_samples.iloc[i][model.covariates], axis=1)
			predictions.append(pred)
		predictions = pd.DataFrame.from_records(np.transpose(predictions))
		pred_df = pd.concat([pred_df, predictions], axis=1)
		
	else:
		print("No Bayesian or bootstrap samples found...using point estimates to generate predictions...")
		reg_result = reg_result = model.regression_result.summary2().tables[1]
		coef_map = {covar:[reg_result.loc[reg_result.index == covar]["Coef."].item()] for covar in reg_result.index}
		coef_samples = pd.DataFrame.from_dict(coef_map)
		coef_samples = pd.DataFrame(np.repeat(coef_samples.values, len(out_sample_data), axis=0), columns=coef_samples.columns)
		predictions = np.sum([out_sample_data[covar] * coef_samples[covar] for covar in model.covariates], axis=0)
		pred_df[model.target_var] = predictions

	return pred_df
