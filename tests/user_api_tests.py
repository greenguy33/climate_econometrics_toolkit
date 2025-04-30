from climate_econometrics_toolkit import user_api as api

import pandas as pd
import time

# TODO: ensure all methods in user_api are tested here.

def assert_series_not_equal(ser1, ser2):
    try:
        pd.testing.assert_series_equal(ser1, ser2)
    except AssertionError:
        pass
    else:
        raise AssertionError
    
def build_example_model():
    
    panel_data = pd.read_csv("tests/test_data/ortiz_bobea_data.csv")

    api.set_dataset(panel_data, "example")
    api.set_panel_column("ISO3")
    api.set_time_column("year")
    api.set_target_variable("tfp")
    api.add_transformation("tfp", ["ln", "fd"])
    api.add_covariates("tmean")
    api.add_transformation("tmean", ["sq", "fd"], keep_original_var=False)


def build_high_degree_fe_example_model():
    
    panel_data = pd.read_csv("data/kotz_et_al_dataset.csv")

    api.set_dataset(panel_data, "hd_fe_example")
    api.set_panel_column("ID")
    api.set_time_column("year")
    api.set_target_variable("dlgdp_pc_usd")
    api.add_covariates(["T5_varm","P5_totalpr"])


def test_model_construction():

    api.load_dataset_from_file("data/GDP_climate_test_data.csv")

    api.set_panel_column("iso_id")
    api.set_time_column("year")

    api.set_target_variable("GDP_per_capita")
    assert api.model.model_vars == ["GDP_per_capita"]

    assert api.model.panel_column == "iso_id"
    assert api.model.time_column == "year"
    assert api.model.target_var == "GDP_per_capita"
    assert api.model.data_file == "GDP_climate_test_data.csv"
    assert api.model.dataset is not None
	
    api.add_covariates("Temp")

    assert api.model.covariates == ["Temp"]
    assert api.model.model_vars == ["Temp","GDP_per_capita"]

    api.add_covariates(["Precip","Temp"])

    assert api.model.covariates == ["Temp","Precip"]
    assert api.model.model_vars == ["Temp","Precip","GDP_per_capita"]

    api.add_fixed_effects("iso_id")

    assert api.model.fixed_effects == ["iso_id"]

    api.add_fixed_effects(["iso_id","year"])

    assert api.model.fixed_effects == ["iso_id","year"]

    api.add_time_trend("iso_id", 2)

    assert api.model.time_trends == ["iso_id 2"]

    api.add_transformation("GDP_per_capita", ["ln","fd"])

    assert api.model.target_var == "fd(ln(GDP_per_capita))"
    assert api.model.model_vars == ["Temp","Precip","fd(ln(GDP_per_capita))"]

    api.add_transformation("Precip", "sq", keep_original_var=False)
    assert api.model.covariates == ["Temp","sq(Precip)"]
    assert api.model.model_vars == ["Temp","sq(Precip)","fd(ln(GDP_per_capita))"]

    api.add_transformation("Temp", "sq")
    assert api.model.covariates == ["Temp","sq(Precip)","sq(Temp)"]
    assert api.model.model_vars == ["Temp","sq(Precip)","sq(Temp)","fd(ln(GDP_per_capita))"]

    api.add_transformation("Precip", ["sq","fd"])
    # doesn't add anything because Precip node doesn't exist
    assert api.model.covariates == ["Temp","sq(Precip)","sq(Temp)"]
    assert api.model.model_vars == ["Temp","sq(Precip)","sq(Temp)","fd(ln(GDP_per_capita))"]

    api.add_covariates("Precip")
    api.remove_transformation("Precip", "sq")
    api.add_transformation("Precip", ["sq","fd"])
    assert api.model.covariates == ["Temp","sq(Temp)","Precip","fd(sq(Precip))"]
    assert api.model.model_vars == ["Temp","sq(Temp)","Precip","fd(sq(Precip))","fd(ln(GDP_per_capita))"]

    api.add_transformation("Precip", "sq")
    assert api.model.covariates == ["Temp","sq(Temp)","Precip","fd(sq(Precip))","sq(Precip)"]
    assert api.model.model_vars == ["Temp","sq(Temp)","Precip","fd(sq(Precip))","sq(Precip)","fd(ln(GDP_per_capita))"]
    
    model1_id = api.evaluate_model()
    model1 = api.get_model_by_id(model1_id)
    assert model1.regression_result is not None

    api.remove_time_trend("iso_id", 2)

    assert api.model.time_trends == []

    api.remove_covariates("Temp")

    assert api.model.covariates == ["sq(Temp)","Precip","fd(sq(Precip))","sq(Precip)"]
    assert api.model.model_vars == ["sq(Temp)","Precip","fd(sq(Precip))","sq(Precip)","fd(ln(GDP_per_capita))"]

    api.remove_transformation("Temp", "sq")

    assert api.model.covariates == ["Precip","fd(sq(Precip))","sq(Precip)"]
    assert api.model.model_vars == ["Precip","fd(sq(Precip))","sq(Precip)","fd(ln(GDP_per_capita))"]

    api.remove_transformation("Precip", ["sq","fd"])

    assert api.model.covariates == ["Precip","sq(Precip)"]
    assert api.model.model_vars == ["Precip","sq(Precip)","fd(ln(GDP_per_capita))"]
    
    api.remove_transformation("GDP_per_capita", ["ln", "fd"])

    assert api.model.target_var == "GDP_per_capita"
    assert api.model.model_vars == ["Precip","sq(Precip)","GDP_per_capita"]

    model2_id = api.evaluate_model()

    api.remove_fixed_effect("iso_id")

    assert api.model.fixed_effects == ["year"]

    api.add_covariates("Temp")
    assert api.model.covariates == ["Precip","sq(Precip)","Temp"]
    assert api.model.model_vars == ["Precip","sq(Precip)","Temp","GDP_per_capita"]

    api.add_random_effect("Temp", "iso_id")

    assert api.model.random_effects == ["Temp","iso_id"]
    assert api.model.covariates == ["Precip","sq(Precip)"]
    assert api.model.model_vars == ["Precip","sq(Precip)","GDP_per_capita"]

    # does nothing because there is already a random effect in the model
    api.add_random_effect("Precip", "iso_id")

    assert api.model.random_effects == ["Temp","iso_id"]
    assert api.model.covariates == ["Precip","sq(Precip)"]
    assert api.model.model_vars == ["Precip","sq(Precip)","GDP_per_capita"]

    api.remove_random_effect()

    assert api.model.random_effects is None
    assert api.model.covariates == ["Precip","sq(Precip)","Temp"]
    assert api.model.model_vars == ["Precip","sq(Precip)","Temp","GDP_per_capita"]

    api.remove_covariates("Temp")
    assert api.model.covariates == ["Precip","sq(Precip)"]
    assert api.model.model_vars == ["Precip","sq(Precip)","GDP_per_capita"]

    api.add_random_effect("Precip", "iso_id")

    assert api.model.random_effects == ["Precip","iso_id"]
    assert api.model.covariates == ["sq(Precip)"]
    assert api.model.model_vars == ["sq(Precip)","GDP_per_capita"]

    model3_id = api.evaluate_model()

    best_rmse_model = api.get_best_model("rmse")
    best_r2_model = api.get_best_model("r2")
    best_mse_model = api.get_best_model("out_sample_mse")
    best_mse_red_model = api.get_best_model("out_sample_mse_reduction")
    best_pred_int_model = api.get_best_model("out_sample_pred_int_cov")

    model1 = api.get_model_by_id(model1_id)
    assert model1 is not None
    model2 = api.get_model_by_id(model2_id)
    assert model2 is not None
    model3 = api.get_model_by_id(model3_id)
    assert model3 is not None

    best_rmse_model = api.get_model_by_id(best_rmse_model.model_id)
    assert best_rmse_model is not None
    best_r2_model = api.get_model_by_id(best_r2_model.model_id)
    assert best_r2_model is not None
    best_mse_model = api.get_model_by_id(best_mse_model.model_id)
    assert best_mse_model is not None
    best_mse_red_model = api.get_model_by_id(best_mse_red_model.model_id)
    assert best_mse_red_model is not None
    best_pred_int_model = api.get_model_by_id(best_pred_int_model.model_id)
    assert best_pred_int_model is not None

    assert not any(model is None for model in [model1,model2,model3,best_rmse_model,best_r2_model,best_mse_model,best_mse_red_model,best_pred_int_model])

    assert len(api.get_all_model_ids()) > 1

def test_regression_standard_error_univariate():

    build_example_model()

    api.evaluate_model()
    res1 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("whitehuber")
    res2 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("neweywest")
    res3 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("clusteredtime")
    res4 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("clusteredspace")
    res5 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("driscollkraay")
    res6 = api.model.regression_result.std_errors

    assert_series_not_equal(res1, res2)
    assert_series_not_equal(res1, res3)
    assert_series_not_equal(res1, res4)
    assert_series_not_equal(res1, res5)
    assert_series_not_equal(res1, res6)
    assert_series_not_equal(res2, res3)
    assert_series_not_equal(res2, res4)
    assert_series_not_equal(res2, res5)
    assert_series_not_equal(res2, res6)
    assert_series_not_equal(res3, res4)
    assert_series_not_equal(res3, res5)
    assert_series_not_equal(res3, res6)
    assert_series_not_equal(res4, res5)
    assert_series_not_equal(res4, res6)
    assert_series_not_equal(res5, res6)


def test_regression_standard_error_multivariate():

    build_example_model()

    api.evaluate_model()
    res1 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("whitehuber")
    res2 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("neweywest")
    res3 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("clusteredtime")
    res4 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("clusteredspace")
    res5 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("driscollkraay")
    res6 = api.model.regression_result.std_errors

    assert_series_not_equal(res1, res2)
    assert_series_not_equal(res1, res3)
    assert_series_not_equal(res1, res4)
    assert_series_not_equal(res1, res5)
    assert_series_not_equal(res1, res6)
    assert_series_not_equal(res2, res3)
    assert_series_not_equal(res2, res4)
    assert_series_not_equal(res2, res5)
    assert_series_not_equal(res2, res6)
    assert_series_not_equal(res3, res4)
    assert_series_not_equal(res3, res5)
    assert_series_not_equal(res3, res6)
    assert_series_not_equal(res4, res5)
    assert_series_not_equal(res4, res6)
    assert_series_not_equal(res5, res6)

    api.add_fixed_effects(["ISO3","year"])

    api.evaluate_model()
    res1 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("whitehuber")
    res2 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("neweywest")
    res3 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("clusteredtime")
    res4 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("clusteredspace")
    res5 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("driscollkraay")
    res6 = api.model.regression_result.std_errors

    assert_series_not_equal(res1, res2)
    assert_series_not_equal(res1, res3)
    assert_series_not_equal(res1, res4)
    assert_series_not_equal(res1, res5)
    assert_series_not_equal(res1, res6)
    assert_series_not_equal(res2, res3)
    assert_series_not_equal(res2, res4)
    assert_series_not_equal(res2, res5)
    assert_series_not_equal(res2, res6)
    assert_series_not_equal(res3, res4)
    assert_series_not_equal(res3, res5)
    assert_series_not_equal(res3, res6)
    assert_series_not_equal(res4, res5)
    assert_series_not_equal(res4, res6)
    assert_series_not_equal(res5, res6)

    api.add_time_trend("ISO3", 2)

    api.evaluate_model()
    res1 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("whitehuber")
    res2 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("neweywest")
    res3 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("clusteredtime")
    res4 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("clusteredspace")
    res5 = api.model.regression_result.summary2().tables[1]["Std.Err."]
    api.evaluate_model("driscollkraay")
    res6 = api.model.regression_result.std_errors

    assert_series_not_equal(res1, res2)
    assert_series_not_equal(res1, res3)
    assert_series_not_equal(res1, res4)
    assert_series_not_equal(res1, res5)
    assert_series_not_equal(res1, res6)
    assert_series_not_equal(res2, res3)
    assert_series_not_equal(res2, res4)
    assert_series_not_equal(res2, res5)
    assert_series_not_equal(res2, res6)
    assert_series_not_equal(res3, res4)
    assert_series_not_equal(res3, res5)
    assert_series_not_equal(res3, res6)
    assert_series_not_equal(res4, res5)
    assert_series_not_equal(res4, res6)
    assert_series_not_equal(res5, res6)


def test_random_effects_model():

    build_example_model()
    api.add_random_effect("tmean", "ISO3")
    api.add_covariates("tmin")
    res = api.evaluate_model()
    assert res is not None

    build_example_model()
    api.add_random_effect("fd(sq(tmean))", "ISO3")
    api.add_covariates("tmin")
    res = api.evaluate_model()
    assert res is not None

    build_example_model()
    api.add_random_effect("fd(sq(tmean))", "ISO3")
    api.add_covariates("tmin")
    api.add_fixed_effects(["ISO3", "year"])
    res = api.evaluate_model()
    assert res is not None

def test_bootstrap():

    build_example_model()

    model_id = api.evaluate_model()

    api.run_block_bootstrap(model_id, "nonrobust", 2)
    api.run_block_bootstrap(model_id, "whitehuber", 2)
    api.run_block_bootstrap(model_id, "neweywest", 2)
    api.run_block_bootstrap(model_id, "clusteredtime", 2)
    api.run_block_bootstrap(model_id, "clusteredspace", 2)
    api.run_block_bootstrap(model_id, "driscollkraay", 2)

def test_spatial_regression():

    build_example_model()

    api.run_spatial_regression("error")
    api.run_spatial_regression("lag")
    api.add_random_effect("tmean","ISO3")
    api.run_spatial_regression("lag")

    api.run_spatial_regression("lag","nonrobust")
    api.run_spatial_regression("lag","whitehuber")
    api.run_spatial_regression("lag","neweywest")

    api.run_spatial_regression("lag","nonrobust",k=10)
    api.run_spatial_regression("lag","whitehuber",k=10)
    api.run_spatial_regression("lag","neweywest",k=10)

    api.run_spatial_regression("lag","nonrobust",num_lags=2)
    api.run_spatial_regression("lag","whitehuber",num_lags=3)
    api.run_spatial_regression("lag","neweywest",num_lags=4)

    api.run_spatial_regression("lag","nonrobust",k=10,num_lags=2)
    api.run_spatial_regression("lag","whitehuber",k=10,num_lags=2)
    api.run_spatial_regression("lag","neweywest",k=10,num_lags=2)

    api.add_fixed_effects(["ISO3","year"])

    api.run_spatial_regression("error")
    api.run_spatial_regression("lag")
    api.add_random_effect("tmean","ISO3")
    api.run_spatial_regression("lag")

    api.run_spatial_regression("lag","nonrobust")
    api.run_spatial_regression("lag","whitehuber")
    api.run_spatial_regression("lag","neweywest")

    api.run_spatial_regression("lag","nonrobust",k=10)
    api.run_spatial_regression("lag","whitehuber",k=10)
    api.run_spatial_regression("lag","neweywest",k=10)

    api.run_spatial_regression("lag","nonrobust",num_lags=2)
    api.run_spatial_regression("lag","whitehuber",num_lags=3)
    api.run_spatial_regression("lag","neweywest",num_lags=4)

    api.run_spatial_regression("lag","nonrobust",k=10,num_lags=2)
    api.run_spatial_regression("lag","whitehuber",k=10,num_lags=2)
    api.run_spatial_regression("lag","neweywest",k=10,num_lags=2)

def test_quantile_regression():

    build_example_model()

    api.run_quantile_regression(.5)
    api.run_quantile_regression([.1,.99])
    api.run_quantile_regression(.5, std_error_type="greene")
    api.run_quantile_regression([.1,.99], std_error_type="greene")

    api.add_fixed_effects(["ISO3","year"])

    api.run_quantile_regression(.5)
    api.run_quantile_regression([.1,.99])
    api.run_quantile_regression(.5, std_error_type="greene")
    api.run_quantile_regression([.1,.99], std_error_type="greene")

    # Very slow
    # api.add_time_trend("ISO3",2)

    # api.run_quantile_regression(.5)
    # api.run_quantile_regression([.1,.99])
    # api.run_quantile_regression(.5, std_error_type="greene")
    # api.run_quantile_regression([.1,.99], std_error_type="greene")

def test_panel_unit_root():

    build_example_model()
    res = api.run_adf_panel_unit_root_tests()
    assert all(val in res.columns for val in ["var","pval_level","pval_fd","decision"])
    assert not pd.isnull(res).values.any()

    start = time.time()
    build_high_degree_fe_example_model()
    res = api.run_adf_panel_unit_root_tests()
    assert all(val in res.columns for val in ["var","pval_level","pval_fd","decision"])
    assert not pd.isnull(res).values.any()
    end = time.time()
    assert end - start < 120
    
def test_cointegration():

    build_example_model()
    res = api.run_engle_granger_cointegration_check()
    assert all(val in res.columns for val in ["dependent_var","pval","significant"])
    assert not pd.isnull(res).values.any()

    start = time.time()
    build_high_degree_fe_example_model()
    res = api.run_engle_granger_cointegration_check()
    print(res)
    assert all(val in res.columns for val in ["dependent_var","pval","significant"])
    assert not pd.isnull(res).values.any()
    end = time.time()
    assert end - start < 120

def test_csd():

    build_example_model()
    res = api.run_pesaran_cross_sectional_dependence_check()
    assert all(val in res.columns for val in ["cd_stat","pval","significant"])
    assert not pd.isnull(res).values.any()

    start = time.time()
    build_high_degree_fe_example_model()
    res = api.run_pesaran_cross_sectional_dependence_check()
    assert all(val in res.columns for val in ["cd_stat","pval","significant"])
    assert not pd.isnull(res).values.any()
    end = time.time()
    assert end - start < 120


def test_specification_search():

    build_example_model()
    model = api.run_specification_search("out_sample_mse_reduction")
    assert model
    model = api.run_specification_search("out_sample_pred_int_cov")
    assert model


def test_extract_and_aggregate_raster_data():

    shape_file = "tests/test_data/country_shapes/country.shp"
    weight_file = "tests/test_data/CroplandPastureArea2000_Geotiff/ag_raster_resampled.tif"

    # monthly data
    monthly_raster_file = "tests/test_data/climate_raster_data/air.2m.mon.mean.shifted.nc"

    # no weight file
    raster_data = api.extract_raster_data(monthly_raster_file, shape_file)
    assert raster_data is not None
    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 12, 1948, shape_file=shape_file, geo_identifier="GMI_CNTRY")
    assert aggregated_data is not None
    assert all(val in range(1948,2025) for val in aggregated_data.time)

    # with weight file
    raster_data = api.extract_raster_data(monthly_raster_file, shape_file, weight_file=weight_file)
    assert raster_data is not None
    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 12, 1948, shape_file=shape_file, geo_identifier="GMI_CNTRY")
    assert aggregated_data is not None
    assert all(val in range(1948,2025) for val in aggregated_data.time)

    raster_data = api.extract_raster_data(monthly_raster_file, shape_file, weight_file=weight_file)
    assert raster_data is not None
    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 12, 1948, shape_file=shape_file, geo_identifier="GMI_CNTRY")
    assert aggregated_data is not None
    assert all(val in range(1948,2025) for val in aggregated_data.time)

    # with default shape file
    raster_data = api.extract_raster_data(monthly_raster_file)
    assert raster_data is not None
    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 12, 1948)
    assert aggregated_data is not None
    assert all(val in range(1948,2025) for val in aggregated_data.time)

    # test with crop growing season mask
    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 12, 1948, crop="maize")
    assert aggregated_data is not None
    assert all(val in range(1948,2025) for val in aggregated_data.time)

    # daily data
    daily_raster_file = "tests/test_data/climate_raster_data/air.2m.gauss.1948.shifted.nc"

    raster_data = api.extract_raster_data(daily_raster_file, shape_file, weight_file=weight_file)
    assert raster_data is not None
    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 1464, 1948)
    assert aggregated_data is not None
    assert len(set(aggregated_data.time)) == 1

    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 1460, 1948)
    assert aggregated_data is not None
    assert len(set(aggregated_data.time)) == 1

    # test with default shape file
    raster_data = api.extract_raster_data(daily_raster_file)
    assert raster_data is not None
    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 1464, 1948)
    assert aggregated_data is not None
    assert len(set(aggregated_data.time)) == 1

    daily_raster_file = "tests/test_data/climate_raster_data/air.2m.gauss.1949.shifted.nc"

    raster_data = api.extract_raster_data(daily_raster_file, shape_file, weight_file=weight_file)
    assert raster_data is not None
    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 1464, 1949)
    assert aggregated_data is not None
    assert len(set(aggregated_data.time)) == 1

    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 1460, 1949)
    assert aggregated_data is not None
    assert len(set(aggregated_data.time)) == 1

    # test with crop growing season masks    
    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 365, 1949, crop="rice")
    assert aggregated_data is not None
    assert len(set(aggregated_data.time)) == 4

    aggregated_data = api.aggregate_raster_data(raster_data, "tmean", "mean", 366, 1949, crop="wheat.spring")
    assert aggregated_data is not None
    assert len(set(aggregated_data.time)) == 4


def test_extraction_with_built_in_weight_files():

    monthly_raster_file = "tests/test_data/climate_raster_data/air.2m.mon.mean.shifted.nc"
    raster_data = api.extract_raster_data(monthly_raster_file, weights="cropweighted")
    assert raster_data is not None
    raster_data = api.extract_raster_data(monthly_raster_file, weights="maizeweighted")
    assert raster_data is not None
    raster_data = api.extract_raster_data(monthly_raster_file, weights="riceweighted")
    assert raster_data is not None
    raster_data = api.extract_raster_data(monthly_raster_file, weights="soybeanweighted")
    assert raster_data is not None
    # raster_data = api.extract_raster_data(monthly_raster_file, weights="popweighted")
    # assert raster_data is not None
    raster_data = api.extract_raster_data(monthly_raster_file, weights="wheatweighted")
    assert raster_data is not None

    ndvi_raster_file = "tests/test_data/climate_raster_data/PKU_GIMMS_NDVI_V1.2_19820101.tif"
    raster_data = api.extract_raster_data(ndvi_raster_file, weights="cropweighted")
    assert raster_data is not None
    raster_data = api.extract_raster_data(ndvi_raster_file, weights="maizeweighted")
    assert raster_data is not None
    raster_data = api.extract_raster_data(ndvi_raster_file, weights="riceweighted")
    assert raster_data is not None
    raster_data = api.extract_raster_data(ndvi_raster_file, weights="soybeanweighted")
    assert raster_data is not None
    # raster_data = api.extract_raster_data(ndvi_raster_file, weights="popweighted")
    # assert raster_data is not None
    raster_data = api.extract_raster_data(ndvi_raster_file, weights="wheatweighted")
    assert raster_data is not None


def test_compute_degree_days():

    panel_data = pd.read_csv("data/ortiz_bobea_data.csv")
    data = api.compute_degree_days(set(panel_data["year"]), set(panel_data["ISO3"]), 20, "above")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.compute_degree_days(set(panel_data["year"]), set(panel_data["ISO3"]), 20, "below")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.compute_degree_days(set(panel_data["year"]), set(panel_data["ISO3"]), 20, "above")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.compute_degree_days(set(panel_data["year"]), set(panel_data["ISO3"]), 20, "below", crop="maize")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.compute_degree_days(set(panel_data["year"]), set(panel_data["ISO3"]), 20, "above", crop="wheat.spring")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.compute_degree_days(set(panel_data["year"]), set(panel_data["ISO3"]), 20, "below", crop="wheat.winter")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.compute_degree_days(set(panel_data["year"]), set(panel_data["ISO3"]), 20, "above", crop="soybeans")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.compute_degree_days(set(panel_data["year"]), set(panel_data["ISO3"]), 20, "below", crop="rice")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.compute_degree_days(set(panel_data["year"]), set(panel_data["ISO3"]), 20, "between", crop="rice", second_threshold=25)
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))


def test_add_degree_days_to_dataframe():

    panel_data = pd.read_csv("data/ortiz_bobea_data.csv")
    data = api.add_degree_days_to_dataframe(panel_data, 20, mode="above")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.add_degree_days_to_dataframe(panel_data, 20, mode="below")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.add_degree_days_to_dataframe(panel_data, 20, mode="above")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.add_degree_days_to_dataframe(panel_data, 20, mode="below", crop="maize")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.add_degree_days_to_dataframe(panel_data, 20, mode="above", crop="wheat.spring")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.add_degree_days_to_dataframe(panel_data, 20, mode="below", crop="wheat.winter")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.add_degree_days_to_dataframe(panel_data, 20, mode="above", crop="soybeans")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.add_degree_days_to_dataframe(panel_data, 20, mode="below", crop="rice")
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))
    data = api.add_degree_days_to_dataframe(panel_data, 20, mode="between", crop="rice", second_threshold=25)
    assert data is not None
    assert sorted(set(data.year)) == list(range(1962,2016))


def test_load_climate_data():

    data = api.load_climate_data()
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_climate_data("popweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_climate_data("cropweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_climate_data("maizeweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_climate_data("wheatweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_climate_data("riceweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_climate_data("soybeanweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data


def test_load_spei_data():

    data = api.load_spei_data()
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_spei_data("popweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_spei_data("cropweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_spei_data("maizeweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_spei_data("wheatweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_spei_data("riceweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_spei_data("soybeanweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data


def test_load_ndvi_data():

    data = api.load_ndvi_data()
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_ndvi_data("popweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_ndvi_data("cropweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_ndvi_data("maizeweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_ndvi_data("wheatweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_ndvi_data("riceweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_ndvi_data("soybeanweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data


def test_load_thi_data():

    data = api.load_temperature_humidity_index_data()
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_temperature_humidity_index_data("popweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_temperature_humidity_index_data("cropweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_temperature_humidity_index_data("maizeweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_temperature_humidity_index_data("wheatweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_temperature_humidity_index_data("riceweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data
    data = api.load_temperature_humidity_index_data("soybeanweighted")
    assert data is not None
    assert "ISO3" in data
    assert "year" in data


def test_load_workdlbank_data():
    data = api.load_worldbank_gdp_data()
    assert data is not None
    assert "ISO3" in data
    assert "year" in data


def test_load_usda_data():
    data = api.load_usda_fda_data()
    assert data is not None
    assert "ISO3" in data
    assert "year" in data


def test_load_emdat_data():
    data = api.load_emdat_data()
    assert data is not None
    assert "ISO3" in data
    assert "year" in data


def test_load_faostat_data():
    data = api.load_faostat_data()
    assert data is not None
    assert "ISO3" in data
    assert "year" in data


def test_integrate_dataframes():
    integrated_data = api.integrate(
        [
            api.load_ndvi_data(),
            api.load_ndvi_data("riceweighted"),
            api.load_ndvi_data("maizeweighted"),
            api.load_faostat_data(),
            api.load_temperature_humidity_index_data(),
            api.load_temperature_humidity_index_data("popweighted"),
            api.load_temperature_humidity_index_data("wheatweighted"),
            api.load_spei_data(),
            api.load_spei_data("cropweighted"),
            api.load_spei_data("soybeanweighted"),
            api.load_climate_data(),
            api.load_climate_data("popweighted"),
            api.load_climate_data("wheatweighted"),
            api.load_worldbank_gdp_data(),
            api.load_usda_fda_data(),
            api.load_emdat_data()
        ]
    )
    assert integrated_data is not None