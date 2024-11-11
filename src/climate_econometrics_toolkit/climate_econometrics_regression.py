import statsmodels.api as sm
import pymc as pm
from pytensor import tensor as pt
import pickle as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

import climate_econometrics_toolkit.climate_econometrics_utils as utils


def run_standard_regression(transformed_data, model, demeaned=False):
	model_vars = utils.get_model_vars(transformed_data, model, demeaned)
	regression_data = transformed_data[model_vars]
	regression_data = sm.add_constant(regression_data)
	reg = sm.OLS(transformed_data[model.target_var],regression_data,missing="drop")
	regression_result = reg.fit()
	return regression_result


def run_intercept_only_regression(transformed_data, model):
	intercept_col = np.ones(len(transformed_data))
	reg = sm.OLS(transformed_data[model.target_var],intercept_col,missing="drop")
	regression_result = reg.fit()
	return regression_result
	

def run_bayesian_regression(transformed_data, model, model_id):

	model_vars = utils.get_model_vars(transformed_data, model)

	scalers, scaled_data = {}, {}
	scalers[model.target_var] = StandardScaler()
	scaled_data[model.target_var] = scalers[model.target_var].fit_transform(np.array(transformed_data[model.target_var]).reshape(-1,1)).flatten()
	for var in model.covariates:
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
		target_prior = pm.Deterministic("target_prior", covar_terms + intercept)
		
		target_scale = pm.HalfNormal("target_scale", 10)
		target_std = pm.HalfNormal("target_std", sigma=target_scale)
		target_posterior = pm.Normal('target_posterior', mu=target_prior, sigma=target_std, observed=scaled_df[model.target_var])

		prior = pm.sample_prior_predictive()
		trace = pm.sample(target_accept=.99, cores=4, tune=1000, draws=1000)
		posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

	with open (f'bayes_samples/bayes_model_{str(model_id)}.pkl', 'wb') as buff:
		pkl.dump({
			"prior":prior,
			"trace":trace,
			"posterior":posterior,
			"var_list":model_vars,
			"target_var":model.target_var
		},buff)

	unscaled_samples = pd.DataFrame()
	for index, var in enumerate(model.covariates):
		unscaled_samples[var] = trace.posterior.covar_coefs[:,:,index].data.flatten() * np.std(transformed_data[model.target_var]) / np.std(transformed_data[var])
	unscaled_samples.to_csv(f'bayes_samples/coefficient_samples_{str(model_id)}.csv')