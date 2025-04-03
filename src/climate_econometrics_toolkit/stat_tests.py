import pandas as pd
import itertools as it
import copy

from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import climate_econometrics_toolkit.regression as regression
import climate_econometrics_toolkit.utils as utils


def run_panel_unit_root_test(res_dict, model, transformed_data, column):
	for entity in set(transformed_data[column]):
		entity_data = transformed_data.loc[transformed_data[column] == entity]
		for var in model.model_vars:
			res_dict["entity"].append(entity)
			res_dict["var"].append(var)
			# check for series constancy
			if entity_data[var].nunique() == 1:
				res_dict["pval"].append(None)
				res_dict["significant"].append(None)
			else:
				res_dict["pval"].append(adfuller(entity_data[var])[1])
				res_dict["significant"].append(True if res_dict["pval"][-1] < .05 else False)
	return res_dict


def panel_unit_root_tests(model):
	transformed_data = utils.transform_data(model.dataset, model)
	res = {"entity":[],"var":[],"pval":[],"significant":[]}
	res = run_panel_unit_root_test(res, model, transformed_data, model.panel_column)
	res = run_panel_unit_root_test(res, model, transformed_data, model.time_column)
	return pd.DataFrame.from_dict(res)


def run_cointegration_tests(res_dict, model, transformed_data, column):
	for entity in set(transformed_data[column]):
		entity_data = transformed_data.loc[transformed_data[column] == entity]
		for var_pair in it.combinations(model.model_vars, 2):
			res_dict["entity"].append(entity)
			res_dict["var1"].append(var_pair[0])
			res_dict["var2"].append(var_pair[1])
			# check for series constancy
			if entity_data[var_pair[0]].nunique() == 1 or entity_data[var_pair[1]].nunique() == 1:
				res_dict["pval"].append(None)
				res_dict["significant"].append(None)
			else:
				res_dict["pval"].append(coint(entity_data[var_pair[0]], entity_data[var_pair[1]])[1])
				res_dict["significant"].append(True if res_dict["pval"][-1] < .05 else False)
	return res_dict
	

def cointegration_tests(model):
	transformed_data = utils.transform_data(model.dataset, model)
	res = {"entity":[],"var1":[],"var2":[],"pval":[],"significant":[]}
	res = run_cointegration_tests(res, model, transformed_data, model.panel_column)
	res = run_cointegration_tests(res, model, transformed_data, model.time_column)
	return pd.DataFrame.from_dict(res)


def run_cross_sectional_dependence_tests(res_dict, transformed_data, column, cross_section_column):
	entity_data_dict = {}
	for entity in set(transformed_data[column]):
		entity_data_dict[entity] = transformed_data.loc[transformed_data[column] == entity]
	for entity_pair in it.combinations(set(transformed_data[column]), 2):
		entity1_data = entity_data_dict[entity_pair[0]]
		entity2_data = entity_data_dict[entity_pair[1]]
		cross_section = set(entity1_data[cross_section_column]).intersection(set(entity2_data[cross_section_column]))
		entity1_resid = entity1_data[entity1_data[cross_section_column].isin(cross_section)]["resid"]
		entity2_resid = entity2_data[entity2_data[cross_section_column].isin(cross_section)]["resid"]
		res_dict["entity1"].append(entity_pair[0])
		res_dict["entity2"].append(entity_pair[1])
		res_dict["pval"].append(spearmanr(entity1_resid, entity2_resid)[1])
		res_dict["significant"].append(True if res_dict["pval"][-1] < .05 else False)
	return res_dict


def cross_sectional_dependence_tests(model):
	demean_data = False
	if len(model.fixed_effects) > 0 and len(model.time_trends) == 0:
		demean_data = True
	transformed_data = utils.transform_data(model.dataset, model, demean=demean_data)
	transformed_data["resid"] = regression.run_standard_regression(transformed_data, model, "nonrobust", demean_data).resid
	res = {"entity1":[],"entity2":[],"pval":[],"significant":[]}
	res = run_cross_sectional_dependence_tests(res, transformed_data, model.panel_column, model.time_column)
	res = run_cross_sectional_dependence_tests(res, transformed_data, model.time_column, model.panel_column)
	return pd.DataFrame.from_dict(res)