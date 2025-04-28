import numpy as np
import pandas as pd
import math

def cumulative_sum_of_predictions_by_geolocation(model, predictions, geo_weights=None, prediction_columns=None):
    total_sum = []
    if prediction_columns is None:
        prediction_columns = [model.target_var]
    for geo_loc, geo_data in predictions.sort_values(model.time_column).groupby(model.panel_column):
        prediction_data = geo_data[prediction_columns]
        if geo_weights is not None:
            if geo_loc in geo_weights:
                prediction_data = prediction_data * geo_weights[geo_loc]
            else:
                continue
        total_sum.append(np.cumsum(prediction_data))
    return pd.DataFrame(np.sum(total_sum, axis=0))

def multiply_geo_coefficients_by_data_column(group_column, data, coefficients, multiplier_column):
    geo_results = {}
    for geo_loc, geo_data in data.groupby(group_column):
        if multiplier_column + "_" + geo_loc in coefficients: 
            geo_result = 0
            for dataval in geo_data[multiplier_column]:
                geo_result += dataval * coefficients[multiplier_column + "_" + geo_loc]
            geo_results[geo_loc] = geo_result
    return geo_results


def convert_geo_log_loss_to_percent(effect_by_geo_loc):
    return {
        geo_loc:np.array([math.expm1(val)*100 for val in effect_by_geo_loc[geo_loc]])
        for geo_loc in set(effect_by_geo_loc)
    }