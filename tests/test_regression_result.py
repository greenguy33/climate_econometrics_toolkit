import pandas as pd
import numpy as np
import statsmodels.api as sm
import copy
from statsmodels.tsa.statespace.tools import diff
import pyfixest as pf
import climate_econometrics_toolkit.model_builder as cet
import climate_econometrics_toolkit.climate_econometrics_regression as cer
import climate_econometrics_toolkit.evaluate_model as cee
import climate_econometrics_toolkit.climate_econometrics_utils as utils


def get_data():
	data = pd.read_csv("data/GrowthClimateDataset.csv")
	data["GDP"] = data["TotGDP"]
	data["Temp"] = data["UDel_temp_popweight"]
	data["Precip"] = data["UDel_precip_popweight"]
	return data

def test_simple_covariates():

	data = get_data()
	# model = cet.parse_cxl("example_cmaps/example_cmap_1.cxl")
	from_indices = ['Temp', 'Precip']
	to_indices = ['GDP', 'GDP']
	model = cet.parse_model_input([from_indices, to_indices], "file1.csv")[0]
	
	transformed_data = utils.transform_data(data, model)

	res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
	assert not any(np.isnan(val) for val in res1["Coef."])
	
	model = cee.evaluate_model(data, model)
	assert not np.isnan(model.out_sample_mse)
	assert not np.isnan(model.out_sample_mse_reduction)
	assert not np.isnan(model.in_sample_mse)
	assert not np.isnan(model.out_sample_pred_int_cov)
	assert .99 > model.out_sample_pred_int_cov and .92 < model.out_sample_pred_int_cov
	pd.testing.assert_frame_equal(res1.sort_index(), model.regression_result.summary2().tables[1].sort_index())

	covars = ["Precip", "Temp"]
	regression_data = data[covars]
	regression_data = sm.add_constant(regression_data)
	model = sm.OLS(data["GDP"],regression_data,missing="drop")
	res2 = model.fit().summary2().tables[1]
	
	pd.testing.assert_frame_equal(res1.sort_index() ,res2.sort_index())

def test_transformed_target_simple_covariates():

	data = get_data()
	# model = cet.parse_cxl("example_cmaps/example_cmap_2.cxl")[0]
	from_indices = ['Temp', 'Precip']
	to_indices = ['fd(ln(GDP))', 'fd(ln(GDP))']
	model = cet.parse_model_input([from_indices, to_indices], "file2.csv")[0]
	
	transformed_data = utils.transform_data(data, model)
	assert(all(val in transformed_data for val in ['ln(GDP)','fd(ln(GDP))']))
	
	res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
	assert not any(np.isnan(val) for val in res1["Coef."])
	
	model = cee.evaluate_model(data, model)
	assert not np.isnan(model.out_sample_mse)
	assert not np.isnan(model.out_sample_mse_reduction)
	assert not np.isnan(model.in_sample_mse)
	assert not np.isnan(model.out_sample_pred_int_cov)
	assert .99 > model.out_sample_pred_int_cov and .92 < model.out_sample_pred_int_cov
	pd.testing.assert_frame_equal(res1.sort_index(), model.regression_result.summary2().tables[1].sort_index())

	covars = ["Precip", "Temp"]
	regression_data = data[covars]
	regression_data = sm.add_constant(regression_data)
	data["ln(GDP)"] = np.log(data["GDP"])
	data["fd(ln(GDP))"] = diff(data["ln(GDP)"])
	model = sm.OLS(data["fd(ln(GDP))"],regression_data,missing="drop")
	res2 = model.fit().summary2().tables[1]
	
	pd.testing.assert_frame_equal(res1.sort_index() ,res2.sort_index())


def test_2_transformed_covariates_transformed_target():

	data = get_data()
	# model = cet.parse_cxl("example_cmaps/example_cmap_3.cxl")
	from_indices = ['ln(Precip)', 'sq(Precip)', 'Precip', 'Temp', 'sq(Temp)', 'fd(Temp)', 'ln(Temp)', 'fd(Precip)']
	to_indices = ['ln(fd(GDP))', 'ln(fd(GDP))', 'ln(fd(GDP))', 'ln(fd(GDP))', 'ln(fd(GDP))', 'ln(fd(GDP))', 'ln(fd(GDP))', 'ln(fd(GDP))']
	model = cet.parse_model_input([from_indices, to_indices], "file3.csv")[0]
	
	transformed_data = utils.transform_data(data, model)
	assert(all(val in transformed_data for val in ['fd(GDP)','ln(fd(GDP))','sq(Temp)','fd(Temp)','ln(Temp)','sq(Precip)','fd(Precip)','ln(Precip)']))

	res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
	assert not any(np.isnan(val) for val in res1["Coef."])

	model = cee.evaluate_model(data, model)
	assert not np.isnan(model.out_sample_mse)
	assert not np.isnan(model.out_sample_mse_reduction)
	assert not np.isnan(model.in_sample_mse)
	assert not np.isnan(model.out_sample_pred_int_cov)
	assert .99 > model.out_sample_pred_int_cov and .92 < model.out_sample_pred_int_cov
	pd.testing.assert_frame_equal(res1.sort_index(), model.regression_result.summary2().tables[1].sort_index())

	covars = ["Precip", "Temp", "ln(Temp)", "ln(Precip)", "fd(Temp)", "fd(Precip)", "sq(Temp)", "sq(Precip)"]
	data["ln(Temp)"] = np.log(data["Temp"])
	data["ln(Precip)"] = np.log(data["Precip"])
	data["fd(Temp)"] = diff(data["Temp"])
	data["fd(Precip)"] = diff(data["Precip"])
	data["sq(Temp)"] = np.square(data["Temp"])
	data["sq(Precip)"] = np.square(data["Precip"])
	data["fd(GDP)"] = diff(data["GDP"])
	data["ln(fd(GDP))"] = np.log(data["fd(GDP)"])

	data = utils.remove_nan_rows(data, ["ln(fd(GDP))"])

	regression_data = data[covars]
	regression_data = sm.add_constant(regression_data)
	model = sm.OLS(data["ln(fd(GDP))"],regression_data,missing="drop")
	res2 = model.fit().summary2().tables[1]

	pd.testing.assert_frame_equal(res1.sort_index() ,res2.sort_index())


def test_transformed_covariates_transformed_target():

	data = get_data()
	# model = cet.parse_cxl("example_cmaps/example_cmap_3.cxl")
	from_indices = ['ln(Precip)', 'sq(Precip)', 'Precip', 'Temp', 'sq(Temp)', 'fd(Temp)', 'ln(Temp)', 'fd(Precip)']
	to_indices = ['fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))']
	model = cet.parse_model_input([from_indices, to_indices], "file3.csv")[0]
	
	transformed_data = utils.transform_data(data, model)
	assert(all(val in transformed_data for val in ['ln(GDP)','fd(ln(GDP))','sq(Temp)','fd(Temp)','ln(Temp)','sq(Precip)','fd(Precip)','ln(Precip)']))

	res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
	assert not any(np.isnan(val) for val in res1["Coef."])

	model = cee.evaluate_model(data, model)
	assert not np.isnan(model.out_sample_mse)
	assert not np.isnan(model.out_sample_mse_reduction)
	assert not np.isnan(model.in_sample_mse)
	assert not np.isnan(model.out_sample_pred_int_cov)
	assert .99 > model.out_sample_pred_int_cov and .92 < model.out_sample_pred_int_cov
	pd.testing.assert_frame_equal(res1.sort_index(), model.regression_result.summary2().tables[1].sort_index())

	covars = ["Precip", "Temp", "ln(Temp)", "ln(Precip)", "fd(Temp)", "fd(Precip)", "sq(Temp)", "sq(Precip)"]
	data["ln(Temp)"] = np.log(data["Temp"])
	data["ln(Precip)"] = np.log(data["Precip"])
	data["fd(Temp)"] = diff(data["Temp"])
	data["fd(Precip)"] = diff(data["Precip"])
	data["sq(Temp)"] = np.square(data["Temp"])
	data["sq(Precip)"] = np.square(data["Precip"])
	regression_data = data[covars]
	regression_data = sm.add_constant(regression_data)
	data["ln(GDP)"] = np.log(data["GDP"])
	data["fd(ln(GDP))"] = diff(data["ln(GDP)"])
	model = sm.OLS(data["fd(ln(GDP))"],regression_data,missing="drop")
	res2 = model.fit().summary2().tables[1]

	pd.testing.assert_frame_equal(res1.sort_index() ,res2.sort_index())


def test_fe_transformed_covariates_transformed_target_iso_year_fixed_effects():

	data = get_data()
	# model = cet.parse_cxl("example_cmaps/example_cmap_4.cxl")
	from_indices = ['fe(year)', 'fe(iso)', 'Temp', 'sq(Temp)', 'fd(Temp)', 'Precip', 'fd(Precip)', 'sq(Precip)']
	to_indices = ['fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))']
	model = cet.parse_model_input([from_indices, to_indices], "file4.csv")[0]
	
	transformed_data = utils.transform_data(data, model)
	assert(all(val in transformed_data for val in ['ln(GDP)','fd(ln(GDP))','sq(Temp)','fd(Temp)','sq(Precip)','fd(Precip)','fe_AGO_iso','fe_1963_year']))

	res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
	assert not any(np.isnan(val) for val in res1["Coef."])
	
	model = cee.evaluate_model(data, model)
	assert not np.isnan(model.out_sample_mse)
	assert not np.isnan(model.out_sample_mse_reduction)
	assert not np.isnan(model.in_sample_mse)
	assert not np.isnan(model.out_sample_pred_int_cov)
	assert .99 > model.out_sample_pred_int_cov and .92 < model.out_sample_pred_int_cov

	res2 = model.regression_result.summary2().tables[1].sort_index()
	np.testing.assert_allclose(res1.loc['Temp']["Coef."],res2.loc['Temp']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['Precip']["Coef."],res2.loc['Precip']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['sq(Temp)']["Coef."],res2.loc['sq(Temp)']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['sq(Precip)']["Coef."],res2.loc['sq(Precip)']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['fd(Temp)']["Coef."],res2.loc['fd(Temp)']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['fd(Precip)']["Coef."],res2.loc['fd(Precip)']["Coef."],rtol=1e-04)

	data["fd_temp"] = diff(data["Temp"])
	data["fd_precip"] = diff(data["Precip"])
	data["sq_temp"] = np.square(data["Temp"])
	data["sq_precip"] = np.square(data["Precip"])
	data["ln_gdp"] = np.log(data["GDP"])
	data["fd_ln_gdp"] = diff(data["ln_gdp"])
	
	res3 = pf.feols("fd_ln_gdp ~ Temp + Precip + fd_temp + fd_precip + sq_temp + sq_precip | iso + year", data=data).coef()

	np.testing.assert_allclose(float(res1.loc[["Precip"]]["Coef."]),float(res3.loc[["Precip"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["Temp"]]["Coef."]),float(res3.loc[["Temp"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["sq(Precip)"]]["Coef."]),float(res3.loc[["sq_precip"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["sq(Temp)"]]["Coef."]),float(res3.loc[["sq_temp"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["fd(Precip)"]]["Coef."]),float(res3.loc[["fd_precip"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["fd(Temp)"]]["Coef."]),float(res3.loc[["fd_temp"]]),rtol=1e-04)


def test_fe_transformed_covariates_transformed_target_iso_fixed_effect():

	data = get_data()
	# model = cet.parse_cxl("example_cmaps/example_cmap_4.cxl")
	from_indices = ['fe(iso)', 'Temp', 'sq(Temp)', 'fd(Temp)', 'Precip', 'fd(Precip)', 'sq(Precip)']
	to_indices = ['fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))']
	model = cet.parse_model_input([from_indices, to_indices], "file4.csv")[0]
	
	transformed_data = utils.transform_data(data, model)
	assert(all(val in transformed_data for val in ['ln(GDP)','fd(ln(GDP))','sq(Temp)','fd(Temp)','sq(Precip)','fd(Precip)','fe_AGO_iso']))

	res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
	assert not any(np.isnan(val) for val in res1["Coef."])
	
	model = cee.evaluate_model(data, model)
	assert not np.isnan(model.out_sample_mse)
	assert not np.isnan(model.out_sample_mse_reduction)
	assert not np.isnan(model.in_sample_mse)
	assert not np.isnan(model.out_sample_pred_int_cov)
	assert .99 > model.out_sample_pred_int_cov and .92 < model.out_sample_pred_int_cov

	res2 = model.regression_result.summary2().tables[1].sort_index()
	np.testing.assert_allclose(res1.loc['Temp']["Coef."],res2.loc['Temp']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['Precip']["Coef."],res2.loc['Precip']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['sq(Temp)']["Coef."],res2.loc['sq(Temp)']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['sq(Precip)']["Coef."],res2.loc['sq(Precip)']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['fd(Temp)']["Coef."],res2.loc['fd(Temp)']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['fd(Precip)']["Coef."],res2.loc['fd(Precip)']["Coef."],rtol=1e-04)

	data["fd_temp"] = diff(data["Temp"])
	data["fd_precip"] = diff(data["Precip"])
	data["sq_temp"] = np.square(data["Temp"])
	data["sq_precip"] = np.square(data["Precip"])
	data["ln_gdp"] = np.log(data["GDP"])
	data["fd_ln_gdp"] = diff(data["ln_gdp"])
	
	res3 = pf.feols("fd_ln_gdp ~ Temp + Precip + fd_temp + fd_precip + sq_temp + sq_precip | iso ", data=data).coef()

	np.testing.assert_allclose(float(res1.loc[["Precip"]]["Coef."]),float(res3.loc[["Precip"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["Temp"]]["Coef."]),float(res3.loc[["Temp"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["sq(Precip)"]]["Coef."]),float(res3.loc[["sq_precip"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["sq(Temp)"]]["Coef."]),float(res3.loc[["sq_temp"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["fd(Precip)"]]["Coef."]),float(res3.loc[["fd_precip"]]))
	np.testing.assert_allclose(float(res1.loc[["fd(Temp)"]]["Coef."]),float(res3.loc[["fd_temp"]]),rtol=1e-04)


def test_tt_transformed_covariates_transformed_target_time_trends():

	data = get_data()
	# model = cet.parse_cxl("example_cmaps/example_cmap_5.cxl")
	from_indices = ['Temp', 'Precip', 'fd(Temp)', 'sq(Temp)', 'sq(Precip)', 'fd(Precip)', 'tt3(iso)', 'year']
	to_indices = ['fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'tt3(iso)']
	model = cet.parse_model_input([from_indices, to_indices], "file5.csv")[0]
	
	transformed_data = utils.transform_data(data, model)
	assert(all(val in transformed_data for val in ['ln(GDP)','fd(ln(GDP))','sq(Temp)','fd(Temp)','sq(Precip)','fd(Precip)','tt_AFG_iso_1','tt_AFG_iso_2','tt_AFG_iso_3']))

	res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
	assert not any(np.isnan(val) for val in res1["Coef."])

	model = cee.evaluate_model(data, model)
	assert not np.isnan(model.out_sample_mse)
	assert not np.isnan(model.out_sample_mse_reduction)
	assert not np.isnan(model.in_sample_mse)
	assert not np.isnan(model.out_sample_pred_int_cov)
	assert .99 > model.out_sample_pred_int_cov and .92 < model.out_sample_pred_int_cov
	pd.testing.assert_frame_equal(res1.sort_index(), model.regression_result.summary2().tables[1].sort_index())

	tt_test_data = pd.read_csv("tests/time_trend_test_data.csv")
	climate_covars = ["Precip", "Temp", "fd(Temp)", "fd(Precip)", "sq(Temp)", "sq(Precip)"]
	covars = copy.deepcopy(climate_covars)
	covars.extend([col for col in tt_test_data.columns if col.startswith("tt")])
	regression_data = tt_test_data[covars]
	regression_data = sm.add_constant(regression_data)
	
	model = sm.OLS(tt_test_data["fd(ln(GDP))"],regression_data,missing="drop")
	res2 = model.fit().summary2().tables[1]

	pd.testing.assert_frame_equal(res1.loc[climate_covars], res2.loc[climate_covars])


def test_tt_transformed_covariates_transformed_target_fixed_effects_and_time_trends():

	data = get_data()
	# model = cet.parse_cxl("example_cmaps/example_cmap_5.cxl")
	from_indices = ['Temp', 'Precip', 'fd(Temp)', 'sq(Temp)', 'sq(Precip)', 'fd(Precip)', 'fe(year)', 'fe(iso)', 'tt3(iso)', 'year']
	to_indices = ['fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'tt3(iso)']
	model = cet.parse_model_input([from_indices, to_indices], "file5.csv")[0]
	
	transformed_data = utils.transform_data(data, model)
	assert(all(val in transformed_data for val in ['ln(GDP)','fd(ln(GDP))','sq(Temp)','fd(Temp)','sq(Precip)','fd(Precip)','fe_AGO_iso','fe_1963_year','tt_AFG_iso_1','tt_AFG_iso_2','tt_AFG_iso_3']))

	res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
	assert not any(np.isnan(val) for val in res1["Coef."])

	model = cee.evaluate_model(data, model)
	assert not np.isnan(model.out_sample_mse)
	assert not np.isnan(model.out_sample_mse_reduction)
	assert not np.isnan(model.in_sample_mse)
	assert not np.isnan(model.out_sample_pred_int_cov)
	assert .99 > model.out_sample_pred_int_cov and .92 < model.out_sample_pred_int_cov
	
	res2 = model.regression_result.summary2().tables[1]
	
	np.testing.assert_allclose(res1.loc['Temp']["Coef."],res2.loc['Temp']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['Precip']["Coef."],res2.loc['Precip']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['sq(Temp)']["Coef."],res2.loc['sq(Temp)']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['sq(Precip)']["Coef."],res2.loc['sq(Precip)']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['fd(Temp)']["Coef."],res2.loc['fd(Temp)']["Coef."],rtol=1e-04)
	np.testing.assert_allclose(res1.loc['fd(Precip)']["Coef."],res2.loc['fd(Precip)']["Coef."],rtol=1e-04)

	tt_test_data = pd.read_csv("tests/time_trend_test_data.csv")

	tt_test_data["fd_temp"] = tt_test_data["fd(Temp)"]
	tt_test_data["fd_precip"] = tt_test_data["fd(Precip)"]
	tt_test_data["sq_temp"] = tt_test_data["sq(Temp)"]
	tt_test_data["sq_precip"] = tt_test_data["sq(Precip)"]
	tt_test_data["fd_ln_gdp"] = tt_test_data["fd(ln(GDP))"]

	climate_covars = ["Precip", "Temp", "fd_temp", "fd_precip", "sq_temp", "sq_precip"]
	covars = copy.deepcopy(climate_covars)
	covars.extend([col for col in tt_test_data.columns if col.startswith("tt")])
	covar_string = " + ".join(covars)

	res3 = pf.feols(f"fd_ln_gdp ~ {covar_string} | iso + year", data=tt_test_data).coef()

	np.testing.assert_allclose(float(res1.loc[["Precip"]]["Coef."]),float(res3.loc[["Precip"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["Temp"]]["Coef."]),float(res3.loc[["Temp"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["sq(Precip)"]]["Coef."]),float(res3.loc[["sq_precip"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["sq(Temp)"]]["Coef."]),float(res3.loc[["sq_temp"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["fd(Precip)"]]["Coef."]),float(res3.loc[["fd_precip"]]),rtol=1e-04)
	np.testing.assert_allclose(float(res1.loc[["fd(Temp)"]]["Coef."]),float(res3.loc[["fd_temp"]]),rtol=1e-04)


def test_burke_model():

	# test from original burke dataset

	from_indices = ['Temp', 'Precip', 'sq(Temp)', 'sq(Precip)','fe(year)', 'fe(iso_id)', 'tt2(iso_id)']
	to_indices = ['growthWDI', 'growthWDI', 'growthWDI', 'growthWDI', 'growthWDI', 'growthWDI', 'growthWDI', 'growthWDI', 'growthWDI']
	data = pd.read_csv("data/GrowthClimateDataset.csv")
	model = cet.parse_model_input([from_indices, to_indices], "file5.csv", "iso_id", "year")[0]

	utils.random_state = 123
	model = cee.evaluate_model(data, model)
	assert .1 < model.out_sample_mse_reduction < .15
	assert .949 < model.out_sample_pred_int_cov < .951
	
	utils.random_state = 1
	model = cee.evaluate_model(data, model)
	assert .1 < model.out_sample_mse_reduction < .15
	assert .949 < model.out_sample_pred_int_cov < .951

	# test from dataset with dependent variable created through transformations

	from_indices = ['Temp', 'Precip', 'sq(Temp)', 'sq(Precip)','fe(year)', 'fe(iso_id)', 'tt2(iso_id)']
	to_indices = ['fd(ln(GDP_per_capita))', 'fd(ln(GDP_per_capita))', 'fd(ln(GDP_per_capita))', 'fd(ln(GDP_per_capita))', 'fd(ln(GDP_per_capita))', 'fd(ln(GDP_per_capita))', 'fd(ln(GDP_per_capita))', 'fd(ln(GDP_per_capita))', 'fd(ln(GDP_per_capita))']
	data = pd.read_csv("data/GDP_climate_test_data.csv")
	model = cet.parse_model_input([from_indices, to_indices], "file5.csv", "iso_id", "year")[0]
	
	utils.random_state = 123
	model = cee.evaluate_model(data, model)
	assert .1 < model.out_sample_mse_reduction < .15
	assert .949 < model.out_sample_pred_int_cov < .951

	utils.random_state = 1
	model = cee.evaluate_model(data, model)
	assert .1 < model.out_sample_mse_reduction < .15
	assert .949 < model.out_sample_pred_int_cov < .951
