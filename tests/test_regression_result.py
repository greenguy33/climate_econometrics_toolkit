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
    data = pd.read_csv("GrowthClimateDataset.csv")
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
    
    model = cee.evaluate_model(transformed_data, model)
    assert not np.isnan(model.out_sample_mse)

    covars = ["Precip", "Temp"]
    regression_data = data[covars]
    regression_data = sm.add_constant(regression_data)
    model = sm.OLS(data["GDP"],regression_data,missing="drop")
    res2 = model.fit().summary2().tables[1]
    
    pd.testing.assert_frame_equal(res1.sort_index() ,res2.sort_index())

def test_simple_covariates_transformed_target():

    data = get_data()
    # model = cet.parse_cxl("example_cmaps/example_cmap_2.cxl")[0]
    from_indices = ['Temp', 'Precip']
    to_indices = ['fd(ln(GDP))', 'fd(ln(GDP))']
    model = cet.parse_model_input([from_indices, to_indices], "file2.csv")[0]
    
    transformed_data = utils.transform_data(data, model)
    assert(all(val in transformed_data for val in ['ln(GDP)','fd(ln(GDP))']))
    
    res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
    assert not any(np.isnan(val) for val in res1["Coef."])
    
    model = cee.evaluate_model(transformed_data, model)
    assert not np.isnan(model.out_sample_mse)

    covars = ["Precip", "Temp"]
    regression_data = data[covars]
    regression_data = sm.add_constant(regression_data)
    data["ln(GDP)"] = np.log(data["GDP"])
    data["fd(ln(GDP))"] = diff(data["ln(GDP)"])
    model = sm.OLS(data["fd(ln(GDP))"],regression_data,missing="drop")
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

    model = cee.evaluate_model(transformed_data, model)
    assert not np.isnan(model.out_sample_mse)

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


def test_transformed_covariates_transformed_target_fixed_effects():

    data = get_data()
    # model = cet.parse_cxl("example_cmaps/example_cmap_4.cxl")
    from_indices = ['fe(year)', 'fe(iso)', 'Temp', 'sq(Temp)', 'fd(Temp)', 'Precip', 'fd(Precip)', 'sq(Precip)']
    to_indices = ['fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))']
    model = cet.parse_model_input([from_indices, to_indices], "file4.csv")[0]
    
    transformed_data = utils.transform_data(data, model)
    assert(all(val in transformed_data for val in ['ln(GDP)','fd(ln(GDP))','sq(Temp)','fd(Temp)','sq(Precip)','fd(Precip)','fe_AFG_iso','fe_1963_year']))

    res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
    assert not any(np.isnan(val) for val in res1["Coef."])
    
    model = cee.evaluate_model(transformed_data, model)
    assert not np.isnan(model.out_sample_mse)

    data["fd_temp"] = diff(data["Temp"])
    data["fd_precip"] = diff(data["Precip"])
    data["sq_temp"] = np.square(data["Temp"])
    data["sq_precip"] = np.square(data["Precip"])
    data["ln_gdp"] = np.log(data["GDP"])
    data["fd_ln_gdp"] = diff(data["ln_gdp"])
    
    res2 = pf.feols("fd_ln_gdp ~ Temp + Precip + fd_temp + fd_precip + sq_temp + sq_precip | iso + year", data=data).coef()

    np.testing.assert_allclose(float(res1.loc[["Precip"]]["Coef."]),float(res2.loc[["Precip"]]))
    np.testing.assert_allclose(float(res1.loc[["Temp"]]["Coef."]),float(res2.loc[["Temp"]]))
    np.testing.assert_allclose(float(res1.loc[["sq(Precip)"]]["Coef."]),float(res2.loc[["sq_precip"]]))
    np.testing.assert_allclose(float(res1.loc[["sq(Temp)"]]["Coef."]),float(res2.loc[["sq_temp"]]))
    np.testing.assert_allclose(float(res1.loc[["fd(Precip)"]]["Coef."]),float(res2.loc[["fd_precip"]]))
    np.testing.assert_allclose(float(res1.loc[["fd(Temp)"]]["Coef."]),float(res2.loc[["fd_temp"]]),rtol=1e-05)


def test_transformed_covariates_transformed_target_incremental_effects():

    data = get_data()
    # model = cet.parse_cxl("example_cmaps/example_cmap_5.cxl")
    from_indices = ['Temp', 'Precip', 'fd(Temp)', 'sq(Temp)', 'sq(Precip)', 'fd(Precip)', 'ie3(iso)']
    to_indices = ['fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))', 'fd(ln(GDP))']
    model = cet.parse_model_input([from_indices, to_indices], "file5.csv")[0]
    
    transformed_data = utils.transform_data(data, model)
    assert(all(val in transformed_data for val in ['ln(GDP)','fd(ln(GDP))','sq(Temp)','fd(Temp)','sq(Precip)','fd(Precip)','ie_AFG_iso_1','ie_AFG_iso_2','ie_AFG_iso_3']))

    res1 = cer.run_standard_regression(transformed_data, model).summary2().tables[1]
    assert not any(np.isnan(val) for val in res1["Coef."])

    model = cee.evaluate_model(transformed_data, model)
    assert not np.isnan(model.out_sample_mse)

    climate_covars = ["Precip", "Temp", "fd(Temp)", "fd(Precip)", "sq(Temp)", "sq(Precip)"]
    data["fd(Temp)"] = diff(data["Temp"])
    data["fd(Precip)"] = diff(data["Precip"])
    data["sq(Temp)"] = np.square(data["Temp"])
    data["sq(Precip)"] = np.square(data["Precip"])
    
    data["ln(GDP)"] = np.log(data["GDP"])
    data["fd(ln(GDP))"] = diff(data["ln(GDP)"])

    ie_level = 3
    for element in sorted(list(set(data["iso"]))):
        data[f"ie_{element}_iso_1"] = np.where(data["iso"] == element, 1, 0)
        data[f"ie_{element}_iso_1"] = np.where(data["iso"] == element, data[f"ie_{element}_iso_1"].cumsum(), 0)
        for i in range(1, ie_level+1):
            data[f"ie_{element}_iso_{i}"] = np.power(data[f"ie_{element}_iso_1"], i)


    covars = copy.deepcopy(climate_covars)
    covars.extend([col for col in data.columns if col.startswith("ie")])

    regression_data = data[covars]
    regression_data = sm.add_constant(regression_data)
    
    model = sm.OLS(data["fd(ln(GDP))"],regression_data,missing="drop")
    res2 = model.fit().summary2().tables[1]

    pd.testing.assert_frame_equal(res1.loc[climate_covars], res2.loc[climate_covars])