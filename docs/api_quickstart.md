# API Quickstart

This document contains some examples of using the API. For full documentation of the API, see the [API documentation](api_documentation.pdf).

### Import the API module

```
import climate_econometrics_toolkit.user_api as api
```

## Data Preprocessing

### Extract some raster data using the built-in country shapes file and soybean weights:
```
extracted = api.extract_raster_data(path_to_raster_file, weights="soybeanweighted")
```
### Aggregate monthly extracted raster data to the country/year level, starting with the year 1948, using the soybean growing season
```
aggregated = api.aggregate_raster_data(extracted, "temp", "mean", 12, 1948, crop="soybeans")
```

### Alternatively, load some pre-loaded datasets for agricultural productiivty (TFP) from USDA FDA and climate data from NCAP-NCER
```
clim_data = api.load_ncep_ncar_data(weight="soybeanweighted")
tfp_data = api.load_usda_fda_data()
```
### Add annual growing degree days between 10C and 15C to data 
```
tfp_data = api.add_degree_days_to_dataframe(tfp_data, threshold=10, mode="between", crop="soybeans", second_threshold=15)
```
### Integrate pre-loaded data and aggregated raster data into a single dataset
```
reg_data = api.integrate([clim_data,tfp_data])
```
![Screenshot from 2025-05-02 14-56-08](https://github.com/user-attachments/assets/ae8887b5-6c40-475c-9dca-95d96c33ea4b)


## Model Building

### Build a model using the first difference of the log of TFP as the dependent variable and the first difference of quadratic soybean-weighted temperature as the covariates, and with country and year specific-intercepts (fixed-effects)
```
api.set_dataset(reg_data, "clim_tfp_data")
api.set_panel_column("ISO3")
api.set_time_column("year")
api.set_target_variable("TFP")
api.add_transformation("TFP", ["ln", "fd"])
api.add_covariates("temp_mean_soybeanweighted")
api.add_transformation("temp_mean_soybeanweighted", ["sq","fd"], keep_original_var=True)
api.add_transformation("temp_mean_soybeanweighted", "fd")
api.add_fixed_effects(["ISO3","year"])
```

### View current model
```
api.view_current_model()
```
```
target_var : fd(ln(TFP))
covariates : ['fd(sq(temp_mean_soybeanweighted))', 'fd(temp_mean_soybeanweighted)']
fixed_effects : ['ISO3', 'year']
random_effects : None
time_trends : []
time_column : year
panel_column : ISO3
out_sample_mse : nan
out_sample_mse_reduction : nan
out_sample_pred_int_cov : nan
r2 : nan
rmse : nan
model_id : None
```

### Evaluate the model using FEOLS ten-fold cross-validation with Newey-West standard error
```
api.evaluate_model_with_OLS(std_error_type="neweywest", cv_folds=10)
```
```
                                          Coef.  Std.Err.             z  \
const                             -2.778268e-19  0.000703 -3.952315e-16   
fd(sq(temp_mean_soybeanweighted)) -3.150014e-04  0.000126 -2.496856e+00   
fd(temp_mean_soybeanweighted)      5.169734e-03  0.003108  1.663373e+00   
```

### View current model again (this time with evaluation stats and assigned model ID)
```
api.view_current_model()
```
```
target_var : fd(ln(TFP))
covariates : ['fd(sq(temp_mean_soybeanweighted))', 'fd(temp_mean_soybeanweighted)']
fixed_effects : ['ISO3', 'year']
random_effects : None
time_trends : []
time_column : year
panel_column : ISO3
out_sample_mse : 0.0065666139847628344
out_sample_mse_reduction : 0.0006865962973973078
out_sample_pred_int_cov : 0.9495509769841926
r2 : 0.0
rmse : 0.08103464681704262
model_id : 1746217056.9088187
```

### Get best model based on supplied metric of all models fit to this dataset
```
best_model = api.get_best_model(metric="rmse")
```

### Get best permutation of best model using specification search
```
best_model = api.run_specification_search(best_model, metric="rmse", cv_folds=2)
```

### Run model as quantile regression
```
api.run_quantile_regression([.1,.5,.99], model=best_model, std_error_type="greene",)
```

### Run model as panel spatial lag regression
```
api.run_spatial_regression("lag", model=best_model, k=5)
```

### Run model as spatial error regression
```
api.run_spatial_regression("error", model=best_model, k=5)
```

## Quantification of Uncertainty and Computation of Impacts

### Run Bayesian Inference to generate coefficient samples
```
api.run_bayesian_regression(best_model, num_samples=1000)
```
### Run block bootstrap to generate coefficient samples
```
api.run_block_bootstrap(best_model, std_error_type="driscollkraay", num_samples=1000)
```
### Use Bayesian samples to generate predictions (used when available; otherwise bootstrap samples are used; otherwise point estimates are used)
```
api.predict_out_of_sample(best_model, out_sample_data)
```
