import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels import PanelOLS
import pymc as pm
import os
from pytensor import tensor as pt
import pickle as pkl
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pandas as pd
import threading
import progressbar
import geopandas as gpd
from spreg import GM_Lag, GM_Error
from libpysal.weights import distance
from shapely.wkt import loads

import climate_econometrics_toolkit.utils as utils

cet_home = os.getenv("CETHOME")

def set_up_regression(transformed_data, model, std_error_type, demeaned=False):
	utils.assert_with_log(std_error_type in utils.supported_standard_errors, f"Standard error type most be in: {utils.std_type_string}")
	utils.std_error_args["clusteredtime"]["groups"] = transformed_data[model.time_column]
	utils.std_error_args["clusteredspace"]["groups"] = transformed_data[model.panel_column]
	return utils.get_model_vars(transformed_data, model, demeaned)


def run_statsmodels_regression(transformed_data, model_vars, model, std_error_type, use_panel_indexing=False):
	if use_panel_indexing:
		transformed_data = transformed_data.set_index([model.panel_column, model.time_column])
	regression_data = transformed_data[model_vars]
	regression_data = sm.add_constant(regression_data)
	if std_error_type not in utils.std_error_args:
		return sm.OLS(transformed_data[model.target_var],regression_data,missing="drop").fit(cov_type=utils.std_error_name_map[std_error_type])
	else:
		return sm.OLS(transformed_data[model.target_var],regression_data,missing="drop").fit(cov_type=utils.std_error_name_map[std_error_type], cov_kwds=utils.std_error_args[std_error_type])


def run_linearmodels_regression(transformed_data, model_vars, model, std_error_type):
	transformed_data = transformed_data.set_index([model.panel_column, model.time_column])
	regression_data = transformed_data[model_vars]
	regression_data = sm.add_constant(regression_data)
	# TODO: check_rank = False not the best long-term solution for bootstrap using driscoll-kraay std. error
	return PanelOLS(transformed_data[model.target_var], regression_data, check_rank=False).fit(cov_type=utils.std_error_name_map[std_error_type])


def run_standard_regression(transformed_data, model, std_error_type, demeaned=False, use_panel_indexing=False):
	model_vars = set_up_regression(transformed_data, model, std_error_type, demeaned)
	if std_error_type != "driscollkraay":
		return run_statsmodels_regression(transformed_data, model_vars, model, std_error_type, use_panel_indexing)
	else:
		if not use_panel_indexing:
			utils.print_with_log("Argument 'use_panel_indexing' is ignored. Panel indexing must be used with Driscoll-Kraay standard error.", "warning")
		return run_linearmodels_regression(transformed_data, model_vars, model, std_error_type)


def run_random_effects_regression(transformed_data, model, std_error_type):
	utils.assert_with_log(std_error_type == "nonrobust", "Specialized standard errors are not currently supported with random effects models")
	model_vars = utils.get_model_vars(transformed_data, model, exclude_fixed_effects=False)
	transformed_data.columns = [col.replace("(","_").replace(")","_") for col in transformed_data.columns]
	model_vars = [var.replace("(","_").replace(")","_") for var in model_vars]
	mv_as_string = "+".join(model_vars) if len(model_vars) > 0 else "0"
	target_var = model.target_var.replace("(","_").replace(")","_")
	random_effects_formatted = model.random_effects[0].replace("(","_").replace(")","_")
	formula = f"{target_var} ~ {mv_as_string}"
	reg = smf.mixedlm(formula, data=transformed_data, groups=model.random_effects[1], re_formula=f"0+{random_effects_formatted}").fit()
	return reg


def run_intercept_only_regression(transformed_data, model, std_error_type):
	utils.assert_with_log(std_error_type in utils.supported_standard_errors, f"Standard error type most be in: {utils.std_type_string}")
	transformed_data["const"] = np.ones(len(transformed_data))
	if std_error_type != "driscollkraay":
		return run_statsmodels_regression(transformed_data, ["const"], model, std_error_type)
	else:
		return run_linearmodels_regression(transformed_data, ["const"], model, std_error_type)


def run_spatial_regression(model, std_error_type, reg_type, model_id, geometry_column, k, num_lags):
	utils.assert_with_log(reg_type in ["lag","error"], "Spatial model type must be either 'lag' or 'error'.")
	utils.assert_with_log(std_error_type in ["nonrobust","whitehuber","neweywest"], "Standard error type must be one of 'nonrobust','whitehuber','neweywest'")
	if reg_type == "error":
		utils.print_with_log("Arguments to fields 'std_error_type' and 'num_lags' are ignored with spatial error models.", "warning")
	if model.random_effects != None:
		utils.print_with_log(f"The specified random-effect {model.random_effects} is ignored in spatial regression model.", "warning")
	demean_data = False
	if len(model.fixed_effects) > 0 and len(model.time_trends) == 0:
		demean_data = True
	transformed_data = utils.transform_data(model.dataset, model, demean=demean_data)
	model_vars = utils.get_model_vars(transformed_data, model, exclude_fixed_effects=demean_data)
	if geometry_column is None or geometry_column == "":
		utils.print_with_log("Geometry column unspecified. Defaulting to ISO3 geometry.", "warning")
		# assume ISO3 if no geometry column specified
		geometry_column_to_use = "geometry"
		countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[['iso_a3', 'geometry']]
		countries = countries[countries['iso_a3'].isin(transformed_data[model.panel_column])]
		transformed_data = transformed_data[transformed_data[model.panel_column].isin(countries.iso_a3)].reset_index(drop=True)
		if len(transformed_data) == 0:
			raise Exception("No geometry column was specified and automatic application of ISO3 geometry failed. Please add a geometry column to your data or use ISO3 data.")
		country_geo = map(lambda country: countries.loc[countries.iso_a3 == country].geometry.item(), transformed_data[model.panel_column])
		transformed_data[geometry_column_to_use] = list(country_geo)
	else:
		geometry_column_to_use = geometry_column
	try:
		W = distance.KNN.from_dataframe(transformed_data[[model.panel_column,geometry_column_to_use]], k=k)
	except ValueError:
		transformed_data[geometry_column_to_use] = transformed_data[geometry_column_to_use].apply(loads)
		W = distance.KNN.from_dataframe(transformed_data[[model.panel_column,geometry_column_to_use]], k=k)
	regression_data = transformed_data[model_vars]
	if reg_type == "error":
		spatial_reg_model = GM_Error(
			y=transformed_data[model.target_var], 
			x=regression_data,
			w=W
	)
	else:
		Wk = distance.Kernel.from_dataframe(transformed_data[[model.panel_column,geometry_column_to_use]], k=k)
		spatial_reg_model = GM_Lag(
			y=transformed_data[model.target_var], 
			x=regression_data,
			w=W,
			gwk=Wk,
			robust=utils.spatial_std_error_map[std_error_type],
			w_lags=num_lags
		)
	save_dir = f"{cet_home}/spatial_regression_output/{model_id}"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file = open(f"{save_dir}/summary.txt", "w")
	file.write(spatial_reg_model.summary)
	with open (f'{save_dir}/model.pkl', 'wb') as buff:
		pkl.dump(spatial_reg_model,buff)
	file.close()
	model.save_spatial_regression_script(reg_type, std_error_type, geometry_column, k, num_lags, demean_data)
	utils.print_with_log(f"Spatial regression results saved to {save_dir}", "info")
	return spatial_reg_model


def run_quantile_regression(model, std_error_type, model_id, q):
	utils.assert_with_log(std_error_type in ["nonrobust","greene"], "Standard error type must be one of 'nonrobust','greene'")
	utils.print_with_log(f"Using quantiles {q} for quantile regression.", "info")
	if model.random_effects != None:
		utils.print_with_log(f"The specified random-effect {model.random_effects} is ignored in quantile regression model.", "warning")
	demean_data = False
	if len(model.fixed_effects) > 0 and len(model.time_trends) == 0:
		demean_data = True
	transformed_data = utils.transform_data(model.dataset, model, demean=demean_data)
	model_vars = utils.get_model_vars(transformed_data, model, exclude_fixed_effects=demean_data)
	regression_data = transformed_data[model_vars]
	regression_data = sm.add_constant(regression_data)
	quant_reg_model = sm.QuantReg(transformed_data[model.target_var], regression_data).fit(q=q, vcov=utils.quantile_std_error_map[std_error_type])
	save_dir = f"{cet_home}/quantile_regression_output/{model_id}_q={q}"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file = open(f"{save_dir}/summary.txt", "w")
	file.write(str(quant_reg_model.summary()))
	with open (f'{save_dir}/model.pkl', 'wb') as buff:
		pkl.dump(quant_reg_model,buff)
	file.close()
	model.save_quantile_regression_script(std_error_type, q)
	utils.print_with_log(f"Quantile regression results saved to {save_dir}", "info")
	return quant_reg_model


def run_block_bootstrap(model, std_error_type, num_samples, use_threading=False):
	utils.print_with_log("Bootstrapping may run for awhile. See progress bar on command line for updates.", "info")
	data = model.dataset
	transformed_data = utils.transform_data(data, model)
	if use_threading:
		thread = threading.Thread(target=bootstrap,name="bootstrap_thread",args=(transformed_data,model,num_samples,std_error_type))
		thread.daemon = True
		thread.start()
	else:
		bootstrap(transformed_data,model,num_samples,std_error_type)


def bootstrap(transformed_data, model, num_samples, std_error_type):
	covar_coefs = {}
	panel_ids = list(set(transformed_data[model.panel_column]))
	for i in progressbar.progressbar(range(num_samples)):
		panel_id_resample = resample(panel_ids)
		resampled_data = pd.DataFrame()
		for panel_id in panel_id_resample:
			resampled_data = pd.concat([resampled_data,transformed_data.loc[transformed_data[model.panel_column] == panel_id]])
		if model.random_effects is not None:
			reg_result = run_random_effects_regression(resampled_data, model, std_error_type)
			for covar in model.covariates:
				if covar not in covar_coefs:
					covar_coefs[covar] = []
				covar_coefs[covar].append(reg_result.params[covar.replace("(","_").replace(")","_")])
			for entity in sorted(set(transformed_data[model.random_effects[1]])):
				if model.random_effects[0] + "_" + entity not in covar_coefs:
					covar_coefs[model.random_effects[0] + "_" + entity] = []
				if entity in reg_result.random_effects:
					covar_coefs[model.random_effects[0] + "_" + entity].append(reg_result.random_effects[entity].item())
				else:
					covar_coefs[model.random_effects[0] + "_" + entity].append(np.NaN)
		else:
			reg_result = run_standard_regression(resampled_data, model, std_error_type)
			for covar in model.covariates:
				if covar not in covar_coefs:
					covar_coefs[covar] = []
				covar_coefs[covar].append(reg_result.params[covar])
	pd.DataFrame.from_dict(covar_coefs).to_csv(f"{cet_home}/bootstrap_samples/coefficient_samples_{str(model.model_id)}.csv")


def run_bayesian_regression(model, num_samples, use_threading=False):
	data = model.dataset
	transformed_data = utils.transform_data(data, model)
	if use_threading:
		thread = threading.Thread(target=run_bayesian_inference,name="bayes_sampling_thread",args=(transformed_data,model,num_samples))
		thread.daemon = True
		thread.start()
	else:
		run_bayesian_inference(transformed_data,model,num_samples)


def run_bayesian_inference(transformed_data, model, num_samples):

	utils.print_with_log("Bayesian inference may run for awhile. See progress bar on command line for updates.", "info")

	utils.assert_with_log(model.model_id is not None, "No ID assigned to the model.")
	model_vars = utils.get_model_vars(transformed_data, model)

	scalers, scaled_data = {}, {}
	scalers[model.target_var] = StandardScaler()
	scaled_data[model.target_var] = scalers[model.target_var].fit_transform(np.array(transformed_data[model.target_var]).reshape(-1,1)).flatten()
	for var in model.covariates:
		if transformed_data.dtypes[var] == "float64":
			scalers[var] = StandardScaler()
			scaled_data[var] = scalers[var].fit_transform(np.array(transformed_data[var]).reshape(-1,1)).flatten()

	scaled_df = pd.DataFrame()
	for var in scaled_data:
		scaled_df[var] = scaled_data[var]
	for var in transformed_data:
		if var not in scaled_df:
			scaled_df[var] = transformed_data[var]

	utils.print_with_log(f"Fitting Bayesian model to dataset of length {len(transformed_data)} containing variables: {model_vars}", "info")

	with pm.Model() as pymc_model:

		covar_coefs = pm.Normal("covar_coefs", 0, 10, shape=(len(model_vars)))
		covar_terms = pm.Deterministic("regressors", pt.sum(covar_coefs * scaled_df[model_vars], axis=1))
		intercept = pm.Normal("intercept", 0, 10)

		if model.random_effects is not None:

			# add dummy variable for random effect if not already present
			if model.random_effects[1] not in model.fixed_effects:
				transformed_data = utils.add_dummy_variable_to_data(model.random_effects[1], transformed_data, leave_out_first=False)
			re_dummy_cols = [col for col in transformed_data.columns if col.startswith("fe_") and col.endswith(f"_{model.random_effects[1]}")]
			
			global_rs_mean = pm.Normal("global_rs_mean",0,10)
			global_rs_sd = pm.HalfNormal("global_rs_sd",10)
			rs_means = pm.Normal("rs_means", global_rs_mean, global_rs_sd, shape=(1,len(set(transformed_data[model.random_effects[1]]))))
			rs_sd = pm.HalfNormal("rs_sd", 10)
			rs_coefs = pm.Normal("rs_coefs", rs_means, rs_sd)
			rs_matrix = pm.Deterministic("rs_matrix", pt.sum(rs_coefs * transformed_data[re_dummy_cols],axis=1))
			rs_terms = pm.Deterministic("rs_terms", rs_matrix * transformed_data[model.random_effects[0]])
	
			target_prior = pm.Deterministic("target_prior", covar_terms + rs_terms + intercept)

		else:
		
			target_prior = pm.Deterministic("target_prior", covar_terms + intercept)
		
		target_scale = pm.HalfNormal("target_scale", 10)
		target_std = pm.HalfNormal("target_std", sigma=target_scale)
		target_posterior = pm.Normal('target_posterior', mu=target_prior, sigma=target_std, observed=scaled_df[model.target_var])

		prior = pm.sample_prior_predictive()
		trace = pm.sample(target_accept=.99, cores=4, tune=num_samples, draws=num_samples)
		posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

	with open (f'{cet_home}/bayes_samples/bayes_model_{str(model.model_id)}.pkl', 'wb') as buff:
		pkl.dump({
			"prior":prior,
			"trace":trace,
			"posterior":posterior,
			"var_list":model_vars,
			"target_var":model.target_var
		},buff)

	unscaled_samples = pd.DataFrame()
	for index, var_name in enumerate(model.covariates):
		unscaled_samples = unscale_variable_list(scalers.keys(), var_name, trace.posterior.covar_coefs[:,:,index].data.flatten(), unscaled_samples, transformed_data, model.target_var)
	if model.random_effects is not None:
		for index, var_name in enumerate(re_dummy_cols):
			var_name = var_name.replace("fe_",f"{model.random_effects[0]}_").replace(f"_{model.random_effects[1]}","") 
			unscaled_samples = unscale_variable_list(scalers.keys(), var_name, trace.posterior.rs_coefs[:,:,:,index].data.flatten(), unscaled_samples, transformed_data, model.target_var)
	unscaled_samples.to_csv(f'{cet_home}/bayes_samples/coefficient_samples_{str(model.model_id)}.csv')


def unscale_variable_list(scaled_vars, var_name, var_values, unscaled_samples, data, target_var):
	if var_name in scaled_vars and target_var in scaled_vars:
		unscaled_samples[var_name] = var_values * np.std(data[target_var]) / np.std(data[var_name])
	elif var_name not in scaled_vars and target_var in scaled_vars:
		unscaled_samples[var_name] = var_values * np.std(data[target_var])
	elif var_name in scaled_vars and target_var not in scaled_vars:
		unscaled_samples[var_name] = var_values / np.std(data[var_name])
	else:
		unscaled_samples[var_name] = var_values
	return unscaled_samples