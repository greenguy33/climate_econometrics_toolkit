from climate_econometrics_toolkit import user_api as api

import pandas as pd

def test_user_api():

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