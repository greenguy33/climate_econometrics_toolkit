import random
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
import pandas as pd

import climate_econometrics_toolkit.utils as utils
import climate_econometrics_toolkit.regression as regression

# TODO: Let the user pick which withholding method they would like to use
def split_data_by_column(data, column, splits=10):
	random.seed(utils.random_state)
	random_years = random.sample(list(set(data[column])), k=len(set(data[column])))
	col_splits = np.array_split(random_years, splits)
	split_list = []
	for col_split in col_splits:
		split_data = []
		split_data.append(list(data.loc[~data[column].isin(col_split)].index))
		split_data.append(list(data.loc[data[column].isin(col_split)].index))
		split_list.append(split_data)
	return split_list


def split_data_randomly(data, model, splits=10):
	target_var = model.target_var
	if any(target_var.startswith(func) for func in utils.supported_functions):
		target_var = target_var.split("(")[-1].split(")")[0]
	# split data based on the target variable to reproduce same train/test split between different model variations
	data = data[target_var]
	kf = KFold(n_splits=splits, shuffle=True, random_state=utils.random_state)
	return kf.split(data)


def generate_withheld_data(data, model):
	# TODO: does this introduce problems for comparing fe/non-fe models?
	# TODO: changing this mades HUGE difference in result of Burke model
	# return split_data_by_column(data, model.time_column)
	return split_data_randomly(data, model)


def calculate_prediction_interval_accuracy(y, predictions, in_sample_mse):
	pred_data = pd.DataFrame(np.transpose([y, predictions.predicted_mean, predictions.var_pred_mean]), columns=["real_y", "pred_mean", "pred_var"])
	pred_data["pred_int_acc"] = np.where(
		(pred_data.pred_mean + np.sqrt(pred_data.pred_var + in_sample_mse) * 1.9603795 > pred_data.real_y) &
		(pred_data.pred_mean - np.sqrt(pred_data.pred_var + in_sample_mse) * 1.9603795 < pred_data.real_y),
		1,
		0
	)
	return np.mean(pred_data.pred_int_acc)


def evaluate_model(data, std_error_type, model):
	if model.random_effects is None:
		return evaluate_non_random_effects_model(data, std_error_type, model)
	else:
		return evaluate_random_effects_model(data, std_error_type, model)


def evaluate_random_effects_model(data, std_error_type, model):

	transformed_data = utils.transform_data(data, model)

	in_sample_mse_list, out_sample_mse_list, intercept_only_mse_list = [], [], []

	for train_indices, test_indices in generate_withheld_data(transformed_data, model):

		train_data_transformed = transformed_data.iloc[train_indices]
		test_data_transformed = transformed_data.iloc[test_indices]
		test_data_transformed.columns = [col.replace("(","_").replace(")","_") for col in test_data_transformed.columns]
		modified_target_var = model.target_var.replace("(","_").replace(")","_")
	
		reg_result = regression.run_random_effects_regression(train_data_transformed, model, std_error_type)

		in_sample_predictions = reg_result.predict(train_data_transformed)
		out_sample_predictions = reg_result.predict(test_data_transformed)

		in_sample_mse = np.mean(np.square(in_sample_predictions-train_data_transformed[modified_target_var]))
		out_sample_mse = np.mean(np.square(out_sample_predictions-test_data_transformed[modified_target_var]))

		intercept_only_model = regression.run_intercept_only_regression(transformed_data, model, std_error_type)
		intercept_only_predictions = intercept_only_model.predict(np.ones(len(test_data_transformed)))
		intercept_only_mse = np.mean(np.square(intercept_only_predictions-test_data_transformed[modified_target_var]))

		intercept_only_mse_list.append(intercept_only_mse)
		in_sample_mse_list.append(in_sample_mse)
		out_sample_mse_list.append(out_sample_mse)

	model.out_sample_mse = np.mean(out_sample_mse_list)
	model.out_sample_mse_reduction = (np.mean(intercept_only_mse_list) - np.mean(out_sample_mse_list)) / np.mean(intercept_only_mse_list)
	model.in_sample_mse = np.mean(in_sample_mse_list)
	model.regression_result = regression.run_random_effects_regression(transformed_data, model, std_error_type)
	model.rmse = np.sqrt(model.out_sample_mse)

	return model


def evaluate_non_random_effects_model(data, std_error_type, model):

	demean_data = False
	if len(model.fixed_effects) > 0 and len(model.time_trends) == 0:
		demean_data = True
	transformed_data = utils.transform_data(data, model, demean=demean_data)

	in_sample_mse_list, out_sample_mse_list, out_sample_pred_int_cov_list, intercept_only_mse_list = [], [], [], []

	for train_indices, test_indices in generate_withheld_data(transformed_data, model):

		train_data_transformed = transformed_data.iloc[train_indices]
		test_data_transformed = transformed_data.iloc[test_indices] 
	
		reg_result = regression.run_standard_regression(train_data_transformed, model, std_error_type)
		
		train_regression_data = train_data_transformed[utils.get_model_vars(test_data_transformed, model, exclude_fixed_effects=demean_data)]
		train_regression_data = sm.add_constant(train_regression_data)
		test_regression_data = test_data_transformed[utils.get_model_vars(test_data_transformed, model, exclude_fixed_effects=demean_data)]
		test_regression_data = sm.add_constant(test_regression_data)
		
		in_sample_predictions = reg_result.get_prediction(train_regression_data)
		out_sample_predictions = reg_result.get_prediction(test_regression_data)

		in_sample_mse = np.mean(np.square(in_sample_predictions.predicted_mean-train_data_transformed[model.target_var]))
		out_sample_mse = np.mean(np.square(out_sample_predictions.predicted_mean-test_data_transformed[model.target_var]))
		
		intercept_only_model = regression.run_intercept_only_regression(train_data_transformed, model, std_error_type)
		intercept_only_predictions = intercept_only_model.predict(np.ones(len(test_data_transformed)))
		intercept_only_mse = np.mean(np.square(intercept_only_predictions-test_data_transformed[model.target_var]))

		intercept_only_mse_list.append(intercept_only_mse)
		in_sample_mse_list.append(in_sample_mse)
		out_sample_mse_list.append(out_sample_mse)
		out_sample_pred_int_cov_list.append(calculate_prediction_interval_accuracy(test_data_transformed[model.target_var], out_sample_predictions, in_sample_mse))

	model.out_sample_mse = np.mean(out_sample_mse_list)
	model.out_sample_mse_reduction = (np.mean(intercept_only_mse_list) - np.mean(out_sample_mse_list)) / np.mean(intercept_only_mse_list)
	model.out_sample_pred_int_cov = np.mean(out_sample_pred_int_cov_list)
	model.in_sample_mse = np.mean(in_sample_mse_list)
	model.regression_result = regression.run_standard_regression(transformed_data, model, std_error_type, demeaned=demean_data)
	model.r2 = float(model.regression_result.summary2().tables[0].loc[model.regression_result.summary2().tables[0][0]=="R-squared:"][1].item())
	model.rmse = np.sqrt(model.out_sample_mse)

	return model