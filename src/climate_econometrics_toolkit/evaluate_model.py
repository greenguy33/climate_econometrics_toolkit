import random
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
import pandas as pd
import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import climate_econometrics_toolkit.utils as utils
import climate_econometrics_toolkit.regression as regression

cet_home = os.getenv("CETHOME")

# TODO: Let the user pick which withholding method they would like to use
def split_data_by_column(data, column, splits):
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


def split_data_randomly(data, model, cv_folds):
	target_var = model.target_var
	if any(target_var.startswith(func) for func in utils.supported_functions):
		target_var = target_var.split("(")[-1].split(")")[0]
	# split data based on the target variable to reproduce same train/test split between different model variations
	data = data[target_var]
	kf = KFold(n_splits=cv_folds, shuffle=True, random_state=utils.random_state)
	return kf.split(data)


def generate_withheld_data(data, model, cv_folds):
	# TODO: does this introduce problems for comparing fe/non-fe models?
	# return split_data_by_column(data, model.time_column)
	utils.print_with_log(f"Splitting data using technique 'random' with {str(cv_folds)} cross-validation folds.")
	return split_data_randomly(data, model, cv_folds)


def generate_prediction_interval_figure(mean_pred_int_cov, predictions, in_sample_mse, target_var, model_id, iteration):
	predictions["uppers"] = predictions.pred_mean + np.sqrt(predictions.pred_var + in_sample_mse) * 1.9603795
	predictions["lowers"] = predictions.pred_mean - np.sqrt(predictions.pred_var + in_sample_mse) * 1.9603795
	predictions = predictions.sort_values("real_y").reset_index(drop=True)
	fig, axis = plt.subplots()
	last_line = None
	for index, row in enumerate(predictions.itertuples()):
		if last_line != None:
			axis.add_patch(
				patches.Polygon(
					xy=[[last_line[0],index-1],[last_line[1],index-1],[row.uppers,index],[row.lowers,index]]
				)
			)
		last_line = [row.lowers,row.uppers]
	axis.scatter(predictions["lowers"], list(range(len(predictions))), color="orange", s=10)
	axis.scatter(predictions["uppers"], list(range(len(predictions))), color="orange", s=10)
	axis.scatter(predictions["real_y"], list(range(len(predictions))), color="red", s=10)
	axis.set_xlabel(target_var, weight="bold")
	axis.set_ylabel("Withheld Row #", weight="bold")
	axis.xaxis.label.set_size(10)
	axis.yaxis.label.set_size(10)
	axis.xaxis.set_tick_params(labelsize=15)
	axis.yaxis.set_tick_params(labelsize=15)
	axis.set_title(f"Prediction Interval Coverage : {f'{mean_pred_int_cov*100:.2f}'}% \n Target coverage: 95%", weight="bold")
	axis.title.set_size(15)
	fig.tight_layout()
	if not os.path.isdir(f"{cet_home}/prediction_intervals/{model_id}"):
		os.makedirs(f"{cet_home}/prediction_intervals/{model_id}")
	plt.savefig(f"{cet_home}/prediction_intervals/{model_id}/{model_id}_cv_iter_{iteration}.png", bbox_inches='tight')
	plt.close()


def calculate_prediction_interval_accuracy(y, predictions, in_sample_mse, target_var, model_id, iteration, gen_figure=True):
	pred_data = pd.DataFrame(np.transpose([y, predictions["predicted_mean"], predictions["var_pred_mean"]]), columns=["real_y", "pred_mean", "pred_var"])
	pred_data["pred_int_acc"] = np.where(
		(pred_data.pred_mean + np.sqrt(pred_data.pred_var + in_sample_mse) * 1.9603795 > pred_data.real_y) &
		(pred_data.pred_mean - np.sqrt(pred_data.pred_var + in_sample_mse) * 1.9603795 < pred_data.real_y),
		1,
		0
	)
	mean_pred_int_cov = np.mean(pred_data.pred_int_acc)
	if gen_figure:
		generate_prediction_interval_figure(mean_pred_int_cov, pred_data, in_sample_mse, target_var, model_id, iteration)
	return mean_pred_int_cov


def get_predictions_from_reg_results(model, model_vars, reg_result, predict_data, std_error_type):
	if std_error_type != "driscollkraay":
		if model_vars is not None:
			predict_data = predict_data[model_vars]
			predict_data = sm.add_constant(predict_data)
		else:
			predict_data = np.ones(len(predict_data))
		predictions = reg_result.get_prediction(predict_data)
		predicted_mean = predictions.predicted_mean
		var_pred_mean = predictions.var_pred_mean
	else:
		predict_data = predict_data.set_index([model.panel_column, model.time_column])
		if model_vars is not None:
			predict_data = predict_data[model_vars]
		predict_data = sm.add_constant(predict_data)
		predicted_mean = list(reg_result.predict(predict_data)["predictions"])
		var_pred_mean = np.diag(predict_data@reg_result.cov@np.transpose(predict_data))
	return {
		"predicted_mean":predicted_mean,
		"var_pred_mean":var_pred_mean
	}


def evaluate_model(data, std_error_type, model, cv_folds):
	if model.model_id is None:
		model.model_id = time.time()
	if model.random_effects is None:
		return evaluate_non_random_effects_model(data, std_error_type, model, cv_folds)
	else:
		return evaluate_random_effects_model(data, std_error_type, model, cv_folds)


def evaluate_non_random_effects_model(data, std_error_type, model, cv_folds):

	utils.print_with_log(f"Evalating non-random-effects model using standard error type '{std_error_type}'", "info")

	demean_data = False
	if len(model.fixed_effects) > 0 and len(model.time_trends) == 0:
		demean_data = True
	utils.print_with_log(f"Demeaning applied: {demean_data}", "info")
	transformed_data = utils.transform_data(data, model, demean=demean_data)

	in_sample_mse_list, out_sample_mse_list, out_sample_pred_int_cov_list, intercept_only_mse_list = [], [], [], []

	for iteration, (train_indices, test_indices) in enumerate(generate_withheld_data(transformed_data, model, cv_folds)):

		train_data_transformed = transformed_data.iloc[train_indices]
		test_data_transformed = transformed_data.iloc[test_indices]
	
		reg_result = regression.run_standard_regression(train_data_transformed, model, std_error_type, demean_data)
		
		model_vars = utils.get_model_vars(train_data_transformed, model, exclude_fixed_effects=demean_data)
		in_sample_predictions = get_predictions_from_reg_results(model, model_vars, reg_result, train_data_transformed, std_error_type)
		out_sample_predictions = get_predictions_from_reg_results(model, model_vars, reg_result, test_data_transformed, std_error_type)

		in_sample_mse = np.mean(np.square(in_sample_predictions["predicted_mean"]-train_data_transformed[model.target_var]))
		out_sample_mse = np.mean(np.square(out_sample_predictions["predicted_mean"]-test_data_transformed[model.target_var]))
		
		intercept_reg_result = regression.run_intercept_only_regression(train_data_transformed, model, std_error_type)
		
		intercept_only_predictions = get_predictions_from_reg_results(model, None, intercept_reg_result, test_data_transformed[[model.time_column,model.panel_column]], std_error_type)
		intercept_only_mse = np.mean(np.square(intercept_only_predictions["predicted_mean"]-test_data_transformed[model.target_var]))

		intercept_only_mse_list.append(intercept_only_mse)
		in_sample_mse_list.append(in_sample_mse)
		out_sample_mse_list.append(out_sample_mse)
		out_sample_pred_int_cov_list.append(calculate_prediction_interval_accuracy(test_data_transformed[model.target_var], out_sample_predictions, in_sample_mse, model.target_var, model.model_id, iteration))

	model.out_sample_mse = np.mean(out_sample_mse_list)
	model.out_sample_mse_reduction = (np.mean(intercept_only_mse_list) - np.mean(out_sample_mse_list)) / np.mean(intercept_only_mse_list)
	model.out_sample_pred_int_cov = np.mean(out_sample_pred_int_cov_list)
	model.in_sample_mse = np.mean(in_sample_mse_list)
	model.regression_result = regression.run_standard_regression(transformed_data, model, std_error_type, demeaned=demean_data)
	if std_error_type != "driscollkraay":
		model.r2 = round(float(model.regression_result.rsquared),2)
	else:
		model.r2 = round(float(model.regression_result._r2),2)
	model.rmse = np.sqrt(model.out_sample_mse)

	return model


def evaluate_random_effects_model(data, std_error_type, model, cv_folds):

	utils.print_with_log(f"Evalating random-effects model using standard error type '{std_error_type}'", "info")

	transformed_data = utils.transform_data(data, model)

	in_sample_mse_list, out_sample_mse_list, intercept_only_mse_list = [], [], []

	for train_indices, test_indices in generate_withheld_data(transformed_data, model, cv_folds):

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