import os
import pandas as pd
import random
import numpy as np
import threading
from exactextract import exact_extract
import geopandas as gpd

import climate_econometrics_toolkit.utils as utils
import climate_econometrics_toolkit.regression as regression

def predict_from_gcms(model, gcms, use_threading=False):
	if use_threading:
		thread = threading.Thread(target=predict,name="prediction_thread",args=(model, gcms))
		thread.daemon = True
		thread.start()
	else:
		predict(model, gcms)


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
	return pd.DataFrame.from_records(data, columns=[geo,"year",climate_var_name])
	

def predict(model, gcms):

	# TODO: way too much hard coded stuff

	random.setstate = utils.random_state

	gcm_data = {}
	for gcm in gcms:
		try:
			data = pd.read_csv(f"gcms/hist-nat_{gcm}_1948-2020_cropland.csv")
		except FileNotFoundError:
			data = pd.read_csv(f"gcms/hist-nat_{gcm}_1950-2020_cropland.csv")
		data["Temp"] = data[["tasmax","tasmin"]].mean(axis=1)-273
		mean_data = pd.DataFrame()
		mean_data["Temp"] = data.groupby(["ISO3","year"])["Temp"].mean()
		mean_data["Precip"] = data.groupby(["ISO3","year"])["pr"].sum() * 2.628e+6
		mean_data = mean_data.reset_index()
		gcm_data[gcm] = mean_data

	bayesian_results = os.path.isdir(f"bayes_samples/{model.model_id}")
	bootstrap_results = os.path.isdir(f"bootstrap_samples/{model.model_id}")

	# TODO: these will never trigger because a new model_is assigned
	if bayesian_results:
		coef_samples = pd.read_csv(f"bayes_samples/{model.model_id}/coefficient_samples_{model.model_id}.csv")
	elif bootstrap_results:
		coef_samples = pd.read_csv(f"bootstrap_samples/{model.model_id}/coefficient_samples_{model.model_id}.csv")
	else:
		transformed_data = utils.transform_data(model.dataset, model)
		reg_result = regression.run_standard_regression(transformed_data, model).summary2().tables[1]
		coef_map = {covar:[reg_result.loc[reg_result.index == covar]["Coef."].item()] for covar in reg_result.index}
		coef_samples = pd.DataFrame.from_dict(coef_map)

	pred_dict = {"country":[],"year":[],"prediction":[]}

	# TODO: make call to utils.transform_data for the gcm data based on the model spec

	if len(gcms) == 1 and len(coef_samples) == 1:
		predictions = gcm_data[gcms[0]]["Temp"] * coef_samples["Temp"].item() + gcm_data[gcms[0]]["Precip"] * coef_samples["Precip"].item()
		if "sq(Temp)" in coef_samples:
			predictions += np.square(gcm_data[gcms[0]]["Temp"]) * coef_samples["sq(Temp)"].item()
		if "sq(Precip)" in coef_samples:
			predictions += np.square(gcm_data[gcms[0]]["Precip"]) * coef_samples["sq(Precip)"].item()
		pred_dict["country"] = gcm_data[gcms[0]].ISO3
		pred_dict["year"] = gcm_data[gcms[0]].year
		pred_dict["prediction"] = predictions
		pd.DataFrame.from_dict(pred_dict).to_csv(f"predictions/predictions_{model.model_id}")
	
	else:
		# TODO: not working because GCMs have different country/year ranges
		num_samples = 1000
		if len(gcms) > 1:
			gcm_samples = random.choices(gcms, k=num_samples)
		else:
			gcm_samples = [gcms[0]] * num_samples
		if len(coef_samples) > 1:
			coef_samples = random.choices(coef_samples, k=num_samples)
		else:
			coef_samples = pd.DataFrame(np.repeat(coef_samples.values, num_samples, axis=0), columns=coef_samples.columns)

		predictions = []
		for i in range(num_samples):
			pred = gcm_data[gcm_samples[i]]["Temp"] * coef_samples["Temp"][i] + gcm_data[gcm_samples[i]]["Precip"] * coef_samples["Precip"][i]
			if "sq(Temp)" in coef_samples:
				pred += np.square(gcm_data[gcm_samples[i]]["Temp"]) * coef_samples["sq(Temp)"][i]
			if "sq(Precip)" in coef_samples:
				pred += np.square(gcm_data[gcm_samples[i]]["Precip"]) * coef_samples["sq(Precip)"][i]
			predictions.append(pred)
		print(predictions)