import os
import pandas as pd
import time
import shutil
import runpy

from climate_econometrics_toolkit import user_api as api

# TODO: ensure all methods in user_api are tested here.

cet_home = os.getenv("CETHOME")

def assert_series_not_equal(ser1, ser2):
	try:
		pd.testing.assert_series_equal(ser1, ser2)
	except AssertionError:
		pass
	else:
		raise AssertionError
	
def build_example_model():

	dataset_name = "TFP_example"
	cache_dir = f'{cet_home}/model_cache/{dataset_name}'
	if os.path.exists(cache_dir):
		shutil.rmtree(cache_dir)
	
	panel_data = pd.read_csv("tests/test_data/ortiz_bobea_data.csv")

	api.set_dataset(panel_data, dataset_name)
	api.set_panel_column("ISO3")
	api.set_time_column("year")
	api.set_target_variable("tfp")
	api.add_transformation("tfp", ["ln", "fd"])
	api.add_covariates("tmean")
	api.add_transformation("tmean", ["sq", "fd"], keep_original_var=False)


def build_high_degree_fe_example_model():
	
	panel_data = pd.read_csv("tests/test_data/kotz_et_al_dataset.csv")

	dataset_name = "hd_fe_example"
	cache_dir = f'{cet_home}/model_cache/{dataset_name}'
	if os.path.exists(cache_dir):
		shutil.rmtree(cache_dir)

	api.set_dataset(panel_data, dataset_name)
	api.set_panel_column("ID")
	api.set_time_column("year")
	api.set_target_variable("dlgdp_pc_usd")
	api.add_covariates(["T5_varm","P5_totalpr"])


def test_model_construction():

	data = pd.read_csv("tests/test_data/GDP_climate_test_data.csv")

	dataset_name = "GDP_example"
	cache_dir = f'{cet_home}/model_cache/{dataset_name}'
	if os.path.exists(cache_dir):
		shutil.rmtree(cache_dir)

	api.set_dataset(data, dataset_name)

	api.set_panel_column("iso_id")
	api.set_time_column("year")

	api.set_target_variable("GDP_per_capita")
	assert api.current_model.model_vars == ["GDP_per_capita"]

	assert api.current_model.panel_column == "iso_id"
	assert api.current_model.time_column == "year"
	assert api.current_model.target_var == "GDP_per_capita"
	assert api.current_model.data_file == "GDP_example"
	assert api.current_model.dataset is not None
	
	api.add_covariates("Temp")

	assert api.current_model.covariates == ["Temp"]
	assert api.current_model.model_vars == ["Temp","GDP_per_capita"]

	api.add_covariates(["Precip","Temp"])

	assert api.current_model.covariates == ["Temp","Precip"]
	assert api.current_model.model_vars == ["Temp","Precip","GDP_per_capita"]

	api.add_fixed_effects("iso_id")

	assert api.current_model.fixed_effects == ["iso_id"]

	api.add_fixed_effects(["iso_id","year"])

	assert api.current_model.fixed_effects == ["iso_id","year"]

	api.add_time_trend("iso_id", 2)

	assert api.current_model.time_trends == ["iso_id 2"]

	api.add_transformation("GDP_per_capita", ["ln","fd"])

	assert api.current_model.target_var == "fd(ln(GDP_per_capita))"
	assert api.current_model.model_vars == ["Temp","Precip","fd(ln(GDP_per_capita))"]

	api.add_transformation("Precip", "sq")
	assert api.current_model.covariates == ["Temp","sq(Precip)"]
	assert api.current_model.model_vars == ["Temp","sq(Precip)","fd(ln(GDP_per_capita))"]

	api.add_transformation("Temp", "sq", keep_original_var=True)
	# TODO: fails here
	assert api.current_model.covariates == ["Temp","sq(Precip)","sq(Temp)"]
	assert api.current_model.model_vars == ["Temp","sq(Precip)","sq(Temp)","fd(ln(GDP_per_capita))"]

	api.add_transformation("Precip", ["sq","fd"])
	# doesn't add anything because Precip node doesn't exist
	assert api.current_model.covariates == ["Temp","sq(Precip)","sq(Temp)"]
	assert api.current_model.model_vars == ["Temp","sq(Precip)","sq(Temp)","fd(ln(GDP_per_capita))"]

	api.add_covariates("Precip")
	api.remove_transformation("Precip", "sq")
	api.add_transformation("Precip", ["sq","fd"], keep_original_var=True)
	assert api.current_model.covariates == ["Temp","sq(Temp)","Precip","fd(sq(Precip))"]
	assert api.current_model.model_vars == ["Temp","sq(Temp)","Precip","fd(sq(Precip))","fd(ln(GDP_per_capita))"]

	api.add_transformation("Precip", "sq", keep_original_var=True)
	assert api.current_model.covariates == ["Temp","sq(Temp)","Precip","fd(sq(Precip))","sq(Precip)"]
	assert api.current_model.model_vars == ["Temp","sq(Temp)","Precip","fd(sq(Precip))","sq(Precip)","fd(ln(GDP_per_capita))"]
	
	model1_id = api.evaluate_model_with_OLS(api.current_model, cv_folds=2)
	model1 = api.get_model_by_id(model1_id)
	assert model1.regression_result is not None

	api.remove_time_trend("iso_id", 2)

	assert api.current_model.time_trends == []

	api.remove_covariates("Temp")

	assert api.current_model.covariates == ["sq(Temp)","Precip","fd(sq(Precip))","sq(Precip)"]
	assert api.current_model.model_vars == ["sq(Temp)","Precip","fd(sq(Precip))","sq(Precip)","fd(ln(GDP_per_capita))"]

	api.remove_transformation("Temp", "sq")

	assert api.current_model.covariates == ["Precip","fd(sq(Precip))","sq(Precip)"]
	assert api.current_model.model_vars == ["Precip","fd(sq(Precip))","sq(Precip)","fd(ln(GDP_per_capita))"]

	api.remove_transformation("Precip", ["sq","fd"])

	assert api.current_model.covariates == ["Precip","sq(Precip)"]
	assert api.current_model.model_vars == ["Precip","sq(Precip)","fd(ln(GDP_per_capita))"]
	
	api.remove_transformation("GDP_per_capita", ["ln", "fd"])

	assert api.current_model.target_var == "GDP_per_capita"
	assert api.current_model.model_vars == ["Precip","sq(Precip)","GDP_per_capita"]

	model2_id = api.evaluate_model_with_OLS(api.current_model, cv_folds=2)

	api.remove_fixed_effect("iso_id")

	assert api.current_model.fixed_effects == ["year"]

	api.add_covariates("Temp")
	assert api.current_model.covariates == ["Precip","sq(Precip)","Temp"]
	assert api.current_model.model_vars == ["Precip","sq(Precip)","Temp","GDP_per_capita"]

	api.add_random_effect("Temp", "iso_id")

	assert api.current_model.random_effects == ["Temp","iso_id"]
	assert api.current_model.covariates == ["Precip","sq(Precip)"]
	assert api.current_model.model_vars == ["Precip","sq(Precip)","GDP_per_capita"]

	# does nothing because there is already a random effect in the model
	api.add_random_effect("Precip", "iso_id")

	assert api.current_model.random_effects == ["Temp","iso_id"]
	assert api.current_model.covariates == ["Precip","sq(Precip)"]
	assert api.current_model.model_vars == ["Precip","sq(Precip)","GDP_per_capita"]

	api.remove_random_effect()

	assert api.current_model.random_effects is None
	assert api.current_model.covariates == ["Precip","sq(Precip)","Temp"]
	assert api.current_model.model_vars == ["Precip","sq(Precip)","Temp","GDP_per_capita"]

	api.remove_covariates("Temp")
	assert api.current_model.covariates == ["Precip","sq(Precip)"]
	assert api.current_model.model_vars == ["Precip","sq(Precip)","GDP_per_capita"]

	api.add_random_effect("Precip", "iso_id")

	assert api.current_model.random_effects == ["Precip","iso_id"]
	assert api.current_model.covariates == ["sq(Precip)"]
	assert api.current_model.model_vars == ["sq(Precip)","GDP_per_capita"]

	model3_id = api.evaluate_model_with_OLS(api.current_model, cv_folds=2)

	best_rmse_model = api.get_best_model(metric="rmse")
	best_r2_model = api.get_best_model(metric="r2")
	best_mse_model = api.get_best_model(metric="out_sample_mse")
	best_mse_red_model = api.get_best_model(metric="out_sample_mse_reduction")
	best_pred_int_model = api.get_best_model(metric="out_sample_pred_int_cov")

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

	print(api.get_all_model_ids())
	assert len(api.get_all_model_ids()) > 1

def test_fit_models_from_model_id():

	dataset_name = "ds1"
	cache_dir = f'{cet_home}/model_cache/{dataset_name}'
	if os.path.exists(cache_dir):
		shutil.rmtree(cache_dir)

	build_example_model()
	model_id = api.evaluate_model_with_OLS(cv_folds=2)

	api.run_adf_panel_unit_root_tests(model_id)
	api.run_engle_granger_cointegration_check(model_id)
	api.run_pesaran_cross_sectional_dependence_check(model_id)
	api.run_specification_search(model_id, cv_folds=2)
	api.evaluate_model_with_OLS(model_id, cv_folds=2)
	api.run_quantile_regression([.1,.2,.3], model=model_id)
	api.run_spatial_regression("lag", model=model_id)
	api.run_spatial_regression("error", model=model_id)
	api.run_block_bootstrap(model_id, num_samples=2)
	api.run_bayesian_regression(model_id, num_samples=2)

	api.add_fixed_effects(["ISO3", "year"])
	api.evaluate_model_with_OLS(cv_folds=2)

	best_model = api.get_best_model(metric="rmse")
	best_model_id = best_model.model_id
	
	api.run_adf_panel_unit_root_tests(best_model_id)
	api.run_engle_granger_cointegration_check(best_model_id)
	api.run_pesaran_cross_sectional_dependence_check(best_model_id)
	api.run_specification_search(best_model_id, cv_folds=2)
	eval_id = api.evaluate_model_with_OLS(best_model_id, cv_folds=2)
	assert str(eval_id) == str(best_model_id)
	quant_id = api.run_quantile_regression([.1,.2,.3], model=best_model_id)
	assert str(quant_id) == str(best_model_id)
	spatial_id = api.run_spatial_regression("lag", model=best_model_id)
	assert str(spatial_id) == str(best_model_id)
	spatial_id = api.run_spatial_regression("error", model=best_model_id)
	assert str(spatial_id) == str(best_model_id)
	boot_id = api.run_block_bootstrap(best_model_id, num_samples=2)
	assert str(boot_id) == str(best_model_id)
	bayes_id = api.run_bayesian_regression(best_model_id, num_samples=2)
	assert str(bayes_id) == str(best_model_id)

	api.run_adf_panel_unit_root_tests(best_model)
	api.run_engle_granger_cointegration_check(best_model)
	api.run_pesaran_cross_sectional_dependence_check(best_model)
	api.run_specification_search(best_model_id, cv_folds=2)
	eval_id = api.evaluate_model_with_OLS(best_model, cv_folds=2)
	assert str(eval_id) == str(best_model_id)
	quant_id = api.run_quantile_regression([.1,.2,.3], model=best_model)
	assert str(quant_id) == str(best_model_id)
	spatial_id = api.run_spatial_regression("lag", model=best_model)
	assert str(spatial_id) == str(best_model_id)
	spatial_id = api.run_spatial_regression("error", model=best_model)
	assert str(spatial_id) == str(best_model_id)
	boot_id = api.run_block_bootstrap(best_model_id, num_samples=2, overwrite_samples=True)
	assert str(boot_id) == str(best_model_id)
	bayes_id = api.run_bayesian_regression(best_model, num_samples=2, overwrite_samples=True)
	assert str(bayes_id) == str(best_model_id)


def test_regression_standard_error_univariate():

	build_example_model()

	api.evaluate_model_with_OLS(cv_folds=2)
	res1 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="whitehuber", cv_folds=2)
	res2 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="neweywest", cv_folds=2)
	res3 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="clusteredtime", cv_folds=2)
	res4 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="clusteredspace", cv_folds=2)
	res5 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="driscollkraay", cv_folds=2)
	res6 = api.current_model.regression_result.std_errors
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

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

	api.evaluate_model_with_OLS(cv_folds=2)
	res1 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="whitehuber", cv_folds=2)
	res2 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="neweywest", cv_folds=2)
	res3 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="clusteredtime", cv_folds=2)
	res4 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="clusteredspace", cv_folds=2)
	res5 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="driscollkraay", cv_folds=2)
	res6 = api.current_model.regression_result.std_errors
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

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

	api.evaluate_model_with_OLS(cv_folds=2)
	res1 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="whitehuber", cv_folds=2)
	res2 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="neweywest", cv_folds=2)
	res3 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="clusteredtime", cv_folds=2)
	res4 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="clusteredspace", cv_folds=2)
	res5 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="driscollkraay", cv_folds=2)
	res6 = api.current_model.regression_result.std_errors
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

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

	api.evaluate_model_with_OLS(cv_folds=2)
	res1 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="whitehuber", cv_folds=2)
	res2 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="neweywest", cv_folds=2)
	res3 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="clusteredtime", cv_folds=2)
	res4 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="clusteredspace", cv_folds=2)
	res5 = api.current_model.regression_result.summary2().tables[1]["Std.Err."]
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	api.evaluate_model_with_OLS(std_error_type="driscollkraay", cv_folds=2)
	res6 = api.current_model.regression_result.std_errors
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

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
	res = api.evaluate_model_with_OLS(cv_folds=2)
	assert res is not None
	assert api.current_model.regression_result is not None
	assert api.current_model.regression_result.random_effects is not None
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	build_example_model()
	api.add_random_effect("fd(sq(tmean))", "ISO3")
	api.add_covariates("tmin")
	res = api.evaluate_model_with_OLS(cv_folds=2)
	assert res is not None
	assert api.current_model.regression_result is not None
	assert api.current_model.regression_result.random_effects is not None
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

	build_example_model()
	api.add_random_effect("fd(sq(tmean))", "ISO3")
	api.add_covariates("tmin")
	api.add_fixed_effects(["ISO3", "year"])
	api.view_current_model()
	res = api.evaluate_model_with_OLS(cv_folds=2)
	assert res is not None
	assert api.current_model.regression_result is not None
	assert api.current_model.regression_result.random_effects is not None
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_OLS.py")

def test_bootstrap():

	build_example_model()
	api.run_block_bootstrap(num_samples=2)
	api.run_block_bootstrap(std_error_type="whitehuber", num_samples=2, overwrite_samples=True)
	api.run_block_bootstrap(std_error_type="neweywest", num_samples=2, overwrite_samples=True)
	api.run_block_bootstrap(std_error_type="clusteredtime", num_samples=2, overwrite_samples=True)
	api.run_block_bootstrap(std_error_type="clusteredspace", num_samples=2, overwrite_samples=True)
	api.run_block_bootstrap(std_error_type="driscollkraay", num_samples=2, overwrite_samples=True)

	build_example_model()
	api.run_block_bootstrap(num_samples=2)
	api.run_block_bootstrap(std_error_type="whitehuber", num_samples=2, overwrite_samples=True)
	api.run_block_bootstrap(std_error_type="neweywest", num_samples=2, overwrite_samples=True)
	api.run_block_bootstrap(std_error_type="clusteredtime", num_samples=2, overwrite_samples=True)
	api.run_block_bootstrap(std_error_type="clusteredspace", num_samples=2, overwrite_samples=True)
	api.run_block_bootstrap(std_error_type="driscollkraay", num_samples=2, overwrite_samples=True)

	build_example_model()
	api.add_random_effect("tmean", "ISO3")
	api.run_block_bootstrap(num_samples=2)

	build_example_model()
	api.add_random_effect("fd(sq(tmean))", "ISO3")
	api.add_covariates("tmin")
	api.run_block_bootstrap(num_samples=2)

	build_example_model()
	api.add_random_effect("tmean", "ISO3")
	api.add_fixed_effects(["ISO3","year"])
	api.run_block_bootstrap(num_samples=2)

def test_bayesian_inference():

	build_example_model()
	api.run_bayesian_regression(num_samples=2)

	build_example_model()
	api.add_fixed_effects(["ISO3","year"])
	api.run_bayesian_regression(num_samples=2)

	build_example_model()
	api.add_random_effect("tmean","ISO3")
	api.run_bayesian_regression(num_samples=2)

	build_example_model()
	api.add_random_effect("fd(sq(tmean))", "ISO3")
	api.add_covariates("tmin")
	api.run_bayesian_regression(num_samples=2)

	build_example_model()
	api.add_random_effect("fd(sq(tmean))", "ISO3")
	api.add_fixed_effects(["ISO3","year"])
	api.add_covariates("tmin")
	api.run_bayesian_regression(num_samples=2)

def test_spatial_regression():

	build_example_model()

	api.run_spatial_regression("error")
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_spatial.py")

	api.run_spatial_regression("lag")
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_spatial.py")

	api.run_spatial_regression("error",k=10)
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_spatial.py")

	api.run_spatial_regression("lag",k=10)
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_spatial.py")

	api.add_covariates("prcp")

	api.run_spatial_regression("error")
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_spatial.py")

	api.run_spatial_regression("lag")
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_spatial.py")


	# failes with 3 covariates due to more columns than rows error in wide panel format
	api.add_covariates(["prcp","tmax"])
	try:
		api.run_spatial_regression("error")
	except AssertionError as e:
		assert str(e) == "Spatial regression transforms dataset into a wide format: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc. In the wide format, your data has more columns than rows, which breaks an assumption of the estimation. To solve this, try reducing the number of time periods in your data or reduce the number of covariates in your regression model."

	try:
		api.run_spatial_regression("lag")
	except AssertionError as e:
		assert str(e) == "Spatial regression transforms dataset into a wide format: x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc. In the wide format, your data has more columns than rows, which breaks an assumption of the estimation. To solve this, try reducing the number of time periods in your data or reduce the number of covariates in your regression model."
	

def test_quantile_regression():

	build_example_model()

	api.run_quantile_regression(.5)
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_quantile_0.5.py")

	api.run_quantile_regression([.1,.99])
	model_id = api.current_model.model_id
	runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.1.py")
	runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.99.py")

	api.run_quantile_regression(.5, std_error_type="greene")
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_quantile_0.5.py")

	api.run_quantile_regression([.1,.99], std_error_type="greene")
	model_id = api.current_model.model_id
	runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.1.py")
	runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.99.py")

	api.add_fixed_effects(["ISO3","year"])

	api.run_quantile_regression(.5)
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_quantile_0.5.py")

	api.run_quantile_regression([.1,.99])
	model_id = api.current_model.model_id
	runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.1.py")
	runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.99.py")

	api.run_quantile_regression(.5, std_error_type="greene")
	runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_quantile_0.5.py")

	api.run_quantile_regression([.1,.99], std_error_type="greene")
	model_id = api.current_model.model_id
	runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.1.py")
	runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.99.py")

	# Very slow
	# api.add_time_trend("ISO3",2)

	# api.run_quantile_regression(.5)
	# runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_quantile_0.5.py")

	# api.run_quantile_regression([.1,.99])
	# model_id = api.current_model.model_id
	# runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.1.py")
	# runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.99.py")

	# api.run_quantile_regression(.5, std_error_type="greene")
	# runpy.run_path(f"{cet_home}/regression_scripts/{api.current_model.model_id}_quantile_0.5.py")

	# api.run_quantile_regression([.1,.99], std_error_type="greene")
	# model_id = api.current_model.model_id
	# runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.1.py")
	# runpy.run_path(f"{cet_home}/regression_scripts/{model_id}_quantile_0.99.py")

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

	build_example_model()
	api.add_random_effect("tmean","ISO3")
	res = api.run_pesaran_cross_sectional_dependence_check()
	print(res)
	assert all(val in res.columns for val in ["cd_stat","pval","significant"])

def test_specification_search():

	dataset_name = "ds1"
	cache_dir = f'{cet_home}/model_cache/{dataset_name}'
	if os.path.exists(cache_dir):
		shutil.rmtree(cache_dir)

	build_example_model()
	model = api.run_specification_search(metric="out_sample_mse_reduction", cv_folds=2)
	assert model
	model = api.run_specification_search(metric="out_sample_pred_int_cov", cv_folds=2)
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
	# uses a lot of memory
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
	# uses a lot of memory
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

	data = api.compute_degree_days([1961,2006], ["AFG","ESP","USA"], 10, second_threshold=20, mode="between", computation="gridded", crop="maize")
	assert data is not None


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


def test_load_ncep_ncar_data():

	data = api.load_ncep_ncar_data()
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_ncep_ncar_data("popweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_ncep_ncar_data("cropweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_ncep_ncar_data("maizeweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_ncep_ncar_data("wheatweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_ncep_ncar_data("riceweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_ncep_ncar_data("soybeanweighted")
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


def test_load_cdc_max_temp_data():
	data = api.load_cdc_unified_max_temperature_data()
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_max_temperature_data("popweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_max_temperature_data("cropweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_max_temperature_data("maizeweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_max_temperature_data("riceweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_max_temperature_data("soybeanweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_max_temperature_data("wheatweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data


def test_load_cdc_min_temp_data():
	data = api.load_cdc_unified_min_temperature_data()
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_min_temperature_data("popweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_min_temperature_data("cropweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_min_temperature_data("maizeweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_min_temperature_data("riceweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_min_temperature_data("soybeanweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_cdc_unified_min_temperature_data("wheatweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data


def test_load_dewpoint_temp_data():
	data = api.load_dewpoint_temperature_data()
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_dewpoint_temperature_data("popweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_dewpoint_temperature_data("cropweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_dewpoint_temperature_data("maizeweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_dewpoint_temperature_data("riceweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_dewpoint_temperature_data("soybeanweighted")
	assert data is not None
	assert "ISO3" in data
	assert "year" in data
	data = api.load_dewpoint_temperature_data("wheatweighted")
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


def test_transform_data():

	build_example_model()
	api.remove_transformation("tfp", ["ln", "fd"])
	api.remove_transformation("tmean", ["sq", "fd"])
	td = api.transform_data(api.current_model)
	expected_cols = ["tfp","tmean","year","ISO3"]
	assert all(val in td for val in expected_cols)
	assert td[expected_cols].isnull().any().sum() == 0
	assert all(year in set(td.year) for year in range(1962,2016))

	build_example_model()
	api.remove_transformation("tfp", ["ln", "fd"])
	api.remove_transformation("tmean", ["sq", "fd"])
	api.add_transformation("tfp", "lag2")
	td = api.transform_data(api.current_model)
	expected_cols = ["tfp","lag2(tfp)","tmean","year","ISO3"]
	assert all(val in td for val in expected_cols)
	assert td[expected_cols].isnull().any().sum() == 0
	assert all(year in set(td.year) for year in range(1964,2016))
	assert all(year not in set(td.year) for year in [1962,1963])

	build_example_model()
	td = api.transform_data(api.current_model)
	expected_cols = ["fd(ln(tfp))","ln(tfp)","tfp","fd(sq(tmean))","sq(tmean)","tmean","year","ISO3"]
	assert all(val in td for val in expected_cols)
	assert td[expected_cols].isnull().any().sum() == 0
	assert all(year in set(td.year) for year in range(1963,2016))
	assert 1962 not in set(td.year)

	build_example_model()
	td = api.transform_data(api.current_model, include_target_var=False)
	expected_cols = ["tfp","fd(sq(tmean))","sq(tmean)","tmean","year","ISO3"]
	assert all(val in td for val in expected_cols)
	assert all(val not in td for val in ["fd(sq(tfp))","sq(tfp)"])
	assert td[expected_cols].isnull().any().sum() == 0
	assert all(year in set(td.year) for year in range(1963,2016))
	assert 1962 not in set(td.year)

	build_example_model()
	api.add_fixed_effects(["ISO3","year"])
	td = api.transform_data(api.current_model)
	expected_cols = ["fd(ln(tfp))","ln(tfp)","tfp","fd(sq(tmean))","sq(tmean)","tmean","year","ISO3"]
	for val in set(td["ISO3"]):
		expected_cols.append(f"fe_{val}_ISO3")
	for val in set(td["year"]):
		expected_cols.append(f"fe_{val}_year")
	expected_cols.remove("fe_AFG_ISO3")
	expected_cols.remove("fe_1963_year")
	for col in expected_cols:
		assert col in td
	assert td[expected_cols].isnull().any().sum() == 0
	assert all(year in set(td.year) for year in range(1963,2016))
	assert 1962 not in set(td.year)

