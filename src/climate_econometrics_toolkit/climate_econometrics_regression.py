import statsmodels.api as sm
import pymc as pm
from pytensor import tensor as pt
import pickle as pkl
import numpy as np
import climate_econometrics_toolkit.climate_econometrics_utils as utils


def run_standard_regression(transformed_data, model, demeaned=False):
	model_vars = utils.get_model_vars(transformed_data, model, demeaned)
	regression_data = transformed_data[model_vars]
	regression_data = sm.add_constant(regression_data)
	reg = sm.OLS(transformed_data[model.target_var],regression_data,missing="drop")
	regression_result = reg.fit()
	return regression_result


def run_intercept_only_regression(transformed_data, model, demeaned=False):
	intercept_col = np.ones(len(transformed_data))
	reg = sm.OLS(transformed_data[model.target_var],intercept_col,missing="drop")
	regression_result = reg.fit()
	return regression_result


def run_bayesian_regression(transformed_data, model):
	with pm.Model() as pymc_model:
		covar_coefs = pm.Normal("covar_coefs", 0, 10, shape=(len(model.covariates)))
		regressors = pm.Deterministic("regressors", pt.sum(covar_coefs * transformed_data[model.covariates], axis=1))
		target_scale = pm.HalfNormal("target_scale", 10)
		target_std = pm.HalfNormal("target_std", sigma=target_scale)
		target_posterior = pm.Normal('target_posterior', mu=regressors, sigma=target_std, observed=transformed_data[model.target_var])

		prior = pm.sample_prior_predictive()
		trace = pm.sample(target_accept=.99, cores=4, tune=1000, draws=1000)
		posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

	with open ('bayes_model.pkl', 'wb') as buff:
		pkl.dump({
			"prior":prior,
			"trace":trace,
			"posterior":posterior,
			"var_list":model.covariates,
			"target_var":model.target_var
		},buff)