import random
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
import pandas as pd

import climate_econometrics_toolkit.climate_econometrics_utils as utils
import climate_econometrics_toolkit.climate_econometrics_regression as regression


def get_year_column(data):
	for col in data.columns:
		if all(len(str(val)) == 4 for val in data[col]) and min(data[col]) > 1600 and "year" in col:
			return col
		else:
			return None


def split_data_by_column(data, column):
	random.seed(1)
	unique_vals = len(set(data[column]))
	withheld_years = random.sample(set(data[column]), int(unique_vals/5))
	train_data = data.loc[~data[column].isin(withheld_years)]
	test_data = data.loc[data[column].isin(withheld_years)]
	return train_data, test_data


def split_data_randomly(data, splits=10):
	kf = KFold(n_splits=splits, shuffle=True, random_state=1)
	return kf.split(data)


def generate_withheld_data(data, model):
	# TOOD: hardcode year col for now - bad
	# TODO: does this introduce problems for comparing fe/non-fe models?
	# split_column = "year"
	# 	if "year" not in data:
	# 		split_column = model.fixed_effects[0]
	# if split_column in data:
	# 	return split_data_by_column(data, split_column)
	# else:
	return split_data_randomly(data)


def calculate_prediction_interval_accuracy(y, predictions, in_sample_mse):
	pred_data = pd.DataFrame(np.transpose([y, predictions.predicted_mean, predictions.var_pred_mean]), columns=["real_y", "pred_mean", "pred_var"])
	assert not any(val < 0 for val in pred_data.pred_var)
	pred_data["pred_int_acc"] = np.where(
		(pred_data.pred_mean + np.sqrt(pred_data.pred_var + in_sample_mse) * 1.9603795 > pred_data.real_y) &
		(pred_data.pred_mean - np.sqrt(pred_data.pred_var + in_sample_mse) * 1.9603795 < pred_data.real_y),
		1,
		0
	)
	return np.mean(pred_data.pred_int_acc)


def evaluate_model(data, model):

	in_sample_mse_list, out_sample_mse_list, out_sample_mse_reduction_list, out_sample_pred_int_cov_list = [], [], [], []

	transformed_data = utils.transform_data(data, model, demean=True)
	for train_indices, test_indices in generate_withheld_data(transformed_data, model):
		train_data_transformed = transformed_data.iloc[train_indices]
		test_data_transformed = transformed_data.iloc[test_indices] 
		reg_result = regression.run_standard_regression(train_data_transformed, model)
		
		train_regression_data = train_data_transformed[utils.get_model_vars(test_data_transformed, model, demeaned=True)]
		train_regression_data = sm.add_constant(train_regression_data)
		test_regression_data = test_data_transformed[utils.get_model_vars(test_data_transformed, model, demeaned=True)]
		test_regression_data = sm.add_constant(test_regression_data)
		
		in_sample_predictions = reg_result.get_prediction(train_regression_data)
		out_sample_predictions = reg_result.get_prediction(test_regression_data)
		in_sample_mse = np.mean(np.square(in_sample_predictions.predicted_mean-train_data_transformed[model.target_var]))
		out_sample_mse = np.mean(np.square(out_sample_predictions.predicted_mean-test_data_transformed[model.target_var]))

		intercept_only_model = regression.run_intercept_only_regression(transformed_data, model)
		intercept_only_predictions = intercept_only_model.predict(np.ones(len(test_data_transformed)))
		intercept_only_mse = np.mean(np.square(intercept_only_predictions-test_data_transformed[model.target_var]))

		in_sample_mse_list.append(in_sample_mse)
		out_sample_mse_list.append(out_sample_mse)
		out_sample_mse_reduction_list.append((out_sample_mse - intercept_only_mse) / intercept_only_mse)
		out_sample_pred_int_cov_list.append(calculate_prediction_interval_accuracy(test_data_transformed[model.target_var], out_sample_predictions, in_sample_mse))

	model.out_sample_mse = np.mean(out_sample_mse_list)
	model.out_sample_mse_reduction = np.mean(out_sample_mse_reduction_list)
	model.out_sample_pred_int_cov = np.mean(out_sample_pred_int_cov_list)
	model.in_sample_mse = np.mean(in_sample_mse_list)
	model.regression_result = regression.run_standard_regression(transformed_data, model, demeaned=True)
	return model