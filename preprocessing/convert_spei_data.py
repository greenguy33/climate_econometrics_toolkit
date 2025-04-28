import pandas as pd
import numpy as np

for weight in ["soybeanweighted","riceweighted","maizeweighted","wheatweighted"]:
    data_dict = {"ISO3":[],"year":[],f"spei_{weight}":[]}
    data = pd.read_csv(f"../../econometric_model_comparison/data/SPEI/SPEI.daily.bycountry.{weight}.csv")
    for year in range(1901,2024):
        year_cols = data[[col for col in data.columns if str(year) in col]]
        data[f"yearly_mean_{str(year)}"] = np.mean(year_cols, axis=1)
    data["ISO3"] = data["country"]
    for row in data.iterrows():
        row = row[1]
        for year in range(1901,2024):
            data_dict["ISO3"].append(row.ISO3)
            data_dict["year"].append(year)
            data_dict[f"spei_{weight}"].append(row[f"yearly_mean_{str(year)}"])
    pd.DataFrame.from_dict(data_dict).sort_values(["ISO3","year"]).to_csv(f"../src/climate_econometrics_toolkit/preprocessed_data/SPEI/spei_{weight}.csv")