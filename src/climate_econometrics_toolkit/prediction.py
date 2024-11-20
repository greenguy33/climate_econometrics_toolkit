import os
import pandas as pd
import random
import numpy as np
import threading

import climate_econometrics_toolkit.utils as utils
import climate_econometrics_toolkit.regression as regression

def predict_from_gcms(model, gcms, use_threading=False):
	if use_threading:
		thread = threading.Thread(target=predict,name="prediction_thread",args=(model, gcms))
		thread.daemon = True
		thread.start()
	else:
		predict(model, gcms)


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