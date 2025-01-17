import numpy as np
import pandas as pd

# All functions should return a pd.DataFrame

def geotemporal_cumulative_sum(model, predictions, geo_weights=None, prediction_columns=None):
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