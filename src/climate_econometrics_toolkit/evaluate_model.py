import random
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
import pandas as pd

import climate_econometrics_toolkit.climate_econometrics_utils as utils
import climate_econometrics_toolkit.climate_econometrics_regression as regression

# TODO: Let the user pick which withholding method they woudl like to use
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
	# TOOD: hardcode year col for now - bad
	# TODO: does this introduce problems for comparing fe/non-fe models?
	return split_data_by_column(data, model.time_column)
	# return split_data_randomly(data, model)


def calculate_prediction_interval_accuracy(y, predictions, in_sample_mse):
	pred_data = pd.DataFrame(np.transpose([y, predictions.predicted_mean, predictions.var_pred_mean]), columns=["real_y", "pred_mean", "pred_var"])
	pred_data["pred_int_acc"] = np.where(
		(pred_data.pred_mean + np.sqrt(pred_data.pred_var + in_sample_mse) * 1.9603795 > pred_data.real_y) &
		(pred_data.pred_mean - np.sqrt(pred_data.pred_var + in_sample_mse) * 1.9603795 < pred_data.real_y),
		1,
		0
	)
	return np.mean(pred_data.pred_int_acc)


def evaluate_model(data, model):

	in_sample_mse_list, out_sample_mse_list, out_sample_pred_int_cov_list, intercept_only_mse_list = [], [], [], []

	demean_data = True
	transformed_data = utils.transform_data(data, model, demean=demean_data)

	for train_indices, test_indices in generate_withheld_data(transformed_data, model):

		train_data_transformed = transformed_data.iloc[train_indices]
		test_data_transformed = transformed_data.iloc[test_indices] 
	
		reg_result = regression.run_standard_regression(train_data_transformed, model)
		
		train_regression_data = train_data_transformed[utils.get_model_vars(test_data_transformed, model, demeaned=demean_data)]
		train_regression_data = sm.add_constant(train_regression_data)
		test_regression_data = test_data_transformed[utils.get_model_vars(test_data_transformed, model, demeaned=demean_data)]
		test_regression_data = sm.add_constant(test_regression_data)
		
		in_sample_predictions = reg_result.get_prediction(train_regression_data)
		out_sample_predictions = reg_result.get_prediction(test_regression_data)

		in_sample_mse = np.mean(np.square(in_sample_predictions.predicted_mean-train_data_transformed[model.target_var]))
		out_sample_mse = np.mean(np.square(out_sample_predictions.predicted_mean-test_data_transformed[model.target_var]))
		
		intercept_only_model = regression.run_intercept_only_regression(transformed_data, model)
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
	model.regression_result = regression.run_standard_regression(transformed_data, model, demeaned=demean_data)

	# transformed_data.to_csv("test_regression_data.csv")

	return model