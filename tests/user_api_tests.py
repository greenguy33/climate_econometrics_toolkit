from climate_econometrics_toolkit import user_api as api

import pandas as pd

def test_user_api():

    api.load_dataset_from_file("data/GDP_climate_test_data.csv")

    api.set_panel_column("iso_id")
    api.set_time_column("year")

    api.set_target_variable("GDP_per_capita")
	
    api.add_covariate("Temp")
    api.add_covariates(["Precip","Temp"])

    # api.add_fixed_effects(["iso_id","year"])
    # api.add_time_trend("iso_id", 2)

    api.add_transformations("GDP_per_capita", ["ln","fd"])
    api.add_transformation("Temp", "sq")
    api.add_transformation("Precip", "sq")

    api.view_current_model()

    model1_id = api.evaluate_model()

    # api.remove_time_trend("iso_id", 2)
    # api.remove_covariate("sq(Precip)")

    # api.view_current_model()

    # model2_id = api.evaluate_model()

    # best_rmse_model_id = api.get_best_model("rmse")
    # best_r2_model_id = api.get_best_model("r2")
    # best_mse_model_id = api.get_best_model("out_sample_mse")
    # best_mse_red_model_id = api.get_best_model("out_sample_mse_reduction")
    # best_pred_int_model_id = api.get_best_model("out_sample_pred_int_cov")

    model1 = api.get_model_by_id(model1_id)
    # model2 = api.get_model_by_id(model2_id)
    # best_rmse_model = api.get_model_by_id(best_rmse_model_id)
    # best_r2_model = api.get_model_by_id(best_r2_model_id)
    # best_mse_model = api.get_model_by_id(best_mse_model_id)
    # best_mse_red_model = api.get_model_by_id(best_mse_red_model_id)
    # best_pred_int_model = api.get_model_by_id(best_pred_int_model_id)

    # best_r2_model.print()

    # api.run_bayesian_regression(model1)
    # api.run_block_bootstrap(model2)

    api.predict_from_gcms(model1, ["BCC-CSM2-MR"])
    api.predict_from_gcms(model1, ["BCC-CSM2-MR","CanESM5","CNRM-CM6-1","HadGEM3-GC31-LL"])