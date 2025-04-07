import pandas as pd
import numpy as np

from scipy.stats import combine_pvalues, norm
from statsmodels.tsa.stattools import adfuller, coint

import climate_econometrics_toolkit.regression as regression


def run_adf_test(data, var):
	# check for series constancy
	if data[var].nunique() == 1:
		return None
	else:
		try:
			return adfuller(data[var])[1]
		# if series is too short ValueError is thrown
		except ValueError:
			return None


def run_panel_unit_root_test(model):
	# implements augmented Dickey-Fuller test
	res = {"var":[],"pval_level":[],"pval_fd":[],"decision":[]}
	vars = set([var.split("(")[-1].split(")")[0] for var in model.model_vars])
	data = model.dataset.dropna(subset=vars)
	for var in vars:
		# TODO: replace this with better fd function when written
		data[f"fd_{var}"] = data.groupby(model.panel_column)[var].diff()
		res["var"].append(var)
		pvals_level, pvals_fd = [], []
		for column in [model.panel_column, model.time_column]:
			for entity in set(data[column]):
				pval_level = run_adf_test(data.loc[data[column] == entity], var)
				if pval_level is not None:
					pvals_level.append(pval_level)
				fd_data = data.dropna(subset=f"fd_{var}")[[model.panel_column,model.time_column,f"fd_{var}"]]
				pval_fd = run_adf_test(fd_data.loc[fd_data[column] == entity], f"fd_{var}")
				if pval_fd is not None:
					pvals_fd.append(pval_fd)
		res["pval_level"].append(combine_pvalues(pvals_level).pvalue)
		res["pval_fd"].append(combine_pvalues(pvals_fd).pvalue)
		if res["pval_level"][-1] is None and res["pval_fd"][-1] is None:
			decision = "Insufficient Data"
		if res["pval_level"][-1] < .05:
			decision = "I(0)"
		elif res["pval_fd"][-1] < .05:
			decision = "I(1)"
		else:
			decision="I(2+)"
		res["decision"].append(decision)
	return res


def panel_unit_root_tests(model):
	res = run_panel_unit_root_test(model)
	# format p-values with scientific notation
	res["pval_level"] = [f"{val:.2e}" if val is not None else None for val in res["pval_level"]]
	res["pval_fd"] = [f"{val:.2e}" if val is not None else None for val in res["pval_fd"]]
	return pd.DataFrame.from_dict(res)


def run_engle_granger_test(data, dep_var, ind_vars):
	# check for series constancy
	if data[dep_var].nunique() == 1:
		return None
	if any([data[var].nunique() == 1 for var in ind_vars]):
		return None
	else:
		try:
			return coint(data[dep_var], data[ind_vars])
		# if series is too short ValueError is thrown
		except ValueError:
			return None


def run_cointegration_tests(model):
	# implements Engle-Granger test
	res = {"dependent_var":[],"pval":[],"significant":[]}
	vars = set([var.split("(")[-1].split(")")[0] for var in model.model_vars])
	data = model.dataset.dropna(subset=vars)
	for dep_var in vars:
		res["dependent_var"].append(dep_var)
		pvals = []
		for column in [model.panel_column, model.time_column]:
			for entity in set(data[column]):
				entity_data = data.loc[data[column] == entity]
				coint_res = run_engle_granger_test(entity_data, dep_var, [ind_var for ind_var in vars if ind_var != dep_var])
				if coint_res is not None and not pd.isnull(coint_res[1]):
					pvals.append(coint_res[1])
		res["pval"].append(combine_pvalues(pvals).pvalue)
		res["significant"].append(True if res["pval"][-1] < .05 else False)
	return res


def cointegration_tests(model):
	res = run_cointegration_tests(model)
	# format p-values with scientific notation
	res["pval"] = [f"{val:.2e}" if val is not None else None for val in res["pval"]]
	return pd.DataFrame.from_dict(res)


def run_cross_sectional_dependence_tests(model):
	# implements pesaran test
	res = {"cd_stat":[],"pval":[],"significant":[]}
	model.model_vars = list(set([var.split("(")[-1].split(")")[0] for var in model.model_vars]))
	model.target_var = model.target_var.split("(")[-1].split(")")[0]
	model.covariates = list(set([var.split("(")[-1].split(")")[0] for var in model.covariates]))
	data = model.dataset.dropna(subset=model.model_vars)
	residuals = regression.run_standard_regression(data, model, "nonrobust", False, use_panel_indexing=True).resid
	resid_corr_matrix = residuals.unstack(level=0).corr()
	num_unique_time_col = len(set(data[model.time_column]))
	res["cd_stat"].append(np.sqrt((2*num_unique_time_col)/(len(resid_corr_matrix)*(len(resid_corr_matrix)-1)))*np.nansum(np.triu(resid_corr_matrix, k=1)))
	res["pval"].append(2 * (1 - norm.cdf(res["cd_stat"][-1])))
	res["significant"].append(True if res["pval"][-1] < .05 else False)
	return res


def cross_sectional_dependence_tests(model):
	res = run_cross_sectional_dependence_tests(model)
	# format p-values with scientific notation
	res["pval"] = [f"{val:.2e}" if val is not None else None for val in res["pval"]]
	return pd.DataFrame.from_dict(res)