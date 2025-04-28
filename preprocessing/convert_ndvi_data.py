import pandas as pd
import numpy as np
import os

ndvi_data_dir = "../../hierarchical_bayesian_drought_study_code/data/PKU_GIMMS_NDVI_AVHRR_MODIS/extracted_with_weights"
ndvi_files = os.listdir(ndvi_data_dir)

for weight in ["popweighted","cropweighted","maizeweighted","wheatweighted","riceweighted","soybeanweighted"]:
    yearly_mean_dataframes = []
    print(weight)
    for year in range(1982, 2023):
        print(year)
        yearly_data = [pd.read_csv(ndvi_data_dir + "/" + file) for file in ndvi_files if str(year) in file]
        yearly_data_files = [file for file in ndvi_files if str(year) in file]
        country_means = {}
        for country in yearly_data[0]["ISO3"]:
            country_means[country] = []
            datasets_to_use = [index for index, file in enumerate(yearly_data_files)]
            for dataset in datasets_to_use:
                country_data = yearly_data[dataset].loc[yearly_data[dataset]["ISO3"]==country]["raw_ndvi"]
                if len(country_data) == 1:
                    country_means[country].append(country_data.item())
                else:
                    country_data = country_data.dropna()
                    if len(country_data) == 1:
                        country_means[country].append(country_data.item())
                    else:
                        country_means[country].append(np.NaN)
            country_means[country] = np.mean(country_means[country]) / 1000
        
        yearly_mean_data = pd.DataFrame()
        yearly_mean_data["ISO3"] = list(country_means.keys())
        yearly_mean_data["year"] = [year] * len(country_means)
        yearly_mean_data[f"ndvi_{weight}"] = list(country_means.values())
        yearly_mean_dataframes.append(yearly_mean_data)
    
    pd.concat(yearly_mean_dataframes).replace(0,np.NaN).dropna().reset_index(drop=True).sort_values(["ISO3","year"]).to_csv(f"../src/climate_econometrics_toolkit/preprocessed_data/NDVI/pku_ndvi_data_aggregated_{weight}.csv")