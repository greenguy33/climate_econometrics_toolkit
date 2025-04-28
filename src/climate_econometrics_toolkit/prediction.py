import os
import pandas as pd
import numpy as np

import climate_econometrics_toolkit.utils as utils

cet_home = os.getenv("CETHOME")


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
			utils.print_with_log("Using Bayesian samples to generate predictions", "info")
		elif bootstrap_results:
			coef_samples = pd.read_csv(f"{cet_home}/bootstrap_samples/coefficient_samples_{model.model_id}.csv")
			utils.print_with_log("Using bootstrap samples to generate predictions", "info")
		predictions = []
		for i in range(len(coef_samples)):
			pred = np.sum(out_sample_data[model.covariates] * coef_samples.iloc[i][model.covariates], axis=1)
			predictions.append(pred)
		predictions = pd.DataFrame.from_records(np.transpose(predictions))
		pred_df = pd.concat([pred_df, predictions], axis=1)
		
	else:
		utils.print_with_log("No Bayesian or bootstrap samples found. Using point estimates to generate predictions.", "warning")
		reg_result = reg_result = model.regression_result.summary2().tables[1]
		coef_map = {covar:[reg_result.loc[reg_result.index == covar]["Coef."].item()] for covar in reg_result.index}
		coef_samples = pd.DataFrame.from_dict(coef_map)
		coef_samples = pd.DataFrame(np.repeat(coef_samples.values, len(out_sample_data), axis=0), columns=coef_samples.columns)
		predictions = np.sum([out_sample_data[covar] * coef_samples[covar] for covar in model.covariates], axis=0)
		pred_df[model.target_var] = predictions

	return pred_df
