import statsmodels.api as sm
import statsmodels.formula.api as smf
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

import climate_econometrics_toolkit.utils as utils

cet_home = os.getenv("CETHOME")

def run_standard_regression(transformed_data, model, demeaned=False):
	model_vars = utils.get_model_vars(transformed_data, model, demeaned)
	regression_data = transformed_data[model_vars]
	regression_data = sm.add_constant(regression_data)
	reg = sm.OLS(transformed_data[model.target_var],regression_data,missing="drop").fit()
	return reg


def run_random_effects_regression(transformed_data, model):
	model_vars = utils.get_model_vars(transformed_data, model)
	transformed_data.columns = [col.replace("(","_").replace(")","_") for col in transformed_data.columns]
	model_vars = [var.replace("(","_").replace(")","_") for var in model_vars]
	mv_as_string = "+".join(model_vars) if len(model_vars) > 0 else "0"
	target_var = model.target_var.replace("(","_").replace(")","_")
	formula = f"{target_var} ~ {mv_as_string}"
	reg = smf.mixedlm(formula, data=transformed_data, groups=model.random_effects[1], re_formula=f"0+{model.random_effects[0]}").fit()
	return reg


def run_intercept_only_regression(transformed_data, model):
	intercept_col = np.ones(len(transformed_data))
	reg = sm.OLS(transformed_data[model.target_var],intercept_col,missing="drop")
	regression_result = reg.fit()
	return regression_result


def run_block_bootstrap(model, num_samples=5, use_threading=False):
	print("Running bootstrap...this may take awhile")
	data = model.dataset
	transformed_data = utils.transform_data(data, model)
	if use_threading:
		thread = threading.Thread(target=bootstrap,name="bootstrap_thread",args=(transformed_data,model,num_samples))
		thread.daemon = True
		thread.start()
	else:
		bootstrap(transformed_data,model,num_samples)


def bootstrap(transformed_data, model, num_samples):
	covar_coefs = {}
	panel_ids = list(set(transformed_data[model.panel_column]))
	for i in progressbar.progressbar(range(num_samples)):
		panel_id_resample = resample(panel_ids)
		resampled_data = pd.DataFrame()
		for panel_id in panel_id_resample:
			resampled_data = pd.concat([resampled_data,transformed_data.loc[transformed_data[model.panel_column] == panel_id]])
		if model.random_effects is not None:
			reg_result = run_random_effects_regression(resampled_data, model)
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
			reg_result = run_standard_regression(resampled_data, model).summary2().tables[1]
			for covar in model.covariates:
				if covar not in covar_coefs:
					covar_coefs[covar] = []
				covar_coefs[covar].append(reg_result.loc[reg_result.index == covar]["Coef."].item())
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

	print(f"Fitting Bayesian model to dataset of length {len(transformed_data)}")

	assert model.model_id is not None
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

	print("Fitting Bayesian model containing variables: ", model_vars)

	with pm.Model() as pymc_model:

		covar_coefs = pm.Normal("covar_coefs", 0, 10, shape=(len(model_vars)))
		covar_terms = pm.Deterministic("regressors", pt.sum(covar_coefs * scaled_df[model_vars], axis=1))
		intercept = pm.Normal("intercept", 0, 10)

		if model.random_effects is not None:

			# add dummy variable for random effect if not already present
			if model.random_effects[1] not in model.fixed_effects:
				transformed_data = utils.add_dummy_variable_to_data(model.random_effects[1], transformed_data, leave_out_first=False)
			re_dummy_cols = [col for col in transformed_data.columns if col.startswith("fe_") and col.endswith(f"_{model.random_effects[1]}")]

			transformed_data.to_csv("transformed_data.csv")
			
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