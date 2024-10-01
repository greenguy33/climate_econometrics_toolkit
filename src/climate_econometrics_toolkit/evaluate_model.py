import pyfixest as pf
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import random
import numpy as np
import statsmodels.api as sm
import climate_econometrics_toolkit.climate_econometrics_utils as utils
import climate_econometrics_toolkit.climate_econometrics_regression as regression


def get_year_column(data):
	for col in data.columns:
		if all(len(str(val)) == 4 for val in data[col]) and min(data[col]) > 1600 and "year" in col:
			return col
		else:
			return None


def demean_fixed_effects(data, model):
	# TODO: cache demean results for future models
	fixed_effects = []
	for fe in model.fixed_effects:
		if not np.issubdtype(data[fe].dtype, np.number):
			enc = OrdinalEncoder()
			ordered_list = list(dict.fromkeys(data[fe]))
			enc.fit(np.array(ordered_list).reshape(-1,1))
			data[f"encoded_{fe}"] = [int(val) for val in enc.transform(np.array(data[fe]).reshape(-1,1))]
			fixed_effects.append(f"encoded_{fe}")
		else:
			fixed_effects.append(fe)
	centered_data = pf.estimation.demean(
		np.array(data[model.model_vars]), 
		np.array(data[fixed_effects]), 
		np.ones(len(data))
	)[0]
	centered_data = pd.DataFrame(centered_data, columns=model.model_vars)
	for fe in model.fixed_effects:
		centered_data = pd.concat([data[fe], centered_data], axis=1).reset_index()
	return centered_data


def split_data_by_year(year_col, data):
	random.seed(1)
	withheld_years = random.sample(set(data[year_col]), 10)
	train_data = data.loc[~data[year_col].isin(withheld_years)]
	test_data = data.loc[data[year_col].isin(withheld_years)]
	return train_data, test_data


def generate_withheld_data(data, model):
	# TODO: support three different split methods (year-based, fixed effect stratified, target distributed)
	# hardcode for now - bad
	split_method = "year"
	year_col = "year"
	if len(model.fixed_effects) > 0:
		data = demean_fixed_effects(data, model)
	if split_method == "year":
		return split_data_by_year(year_col, data)


def evaluate_model(data, model):
	transformed_data = utils.transform_data(data, model)
	missing_indices = []
	no_nan_cols = model.covariates + model.fixed_effects + [model.target_var]
	for index, row in enumerate(transformed_data.iterrows()):
		if any(pd.isna(row[1][col]) for col in no_nan_cols):
			missing_indices.append(index)
	transformed_data = transformed_data.drop(missing_indices).reset_index(drop=True)
	train_data_transformed, test_data_transformed = generate_withheld_data(transformed_data, model)
	reg_result = regression.run_standard_regression(train_data_transformed, model)
	test_regression_data = test_data_transformed[utils.get_model_vars(test_data_transformed, model, demeaned=True)]
	if len(model.fixed_effects) == 0:
		test_regression_data = sm.add_constant(test_regression_data)
	predictions = reg_result.predict(test_regression_data)
	out_sample_mse = np.mean(np.square(predictions-test_data_transformed[model.target_var]))
	model.out_sample_mse = out_sample_mse
	return model