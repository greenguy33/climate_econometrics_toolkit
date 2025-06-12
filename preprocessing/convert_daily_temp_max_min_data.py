import pandas as pd
import numpy as np
from countrycode import countrycode as cc

for weight in ["unweighted","agweighted","popweighted","maizeweighted","wheatweighted","riceweighted","soybeanweighted"]:
    for timeframe in ["daily_max","daily_min"]:
        res_dict = {"ISO3":[],"year":[],f"temp_{timeframe}_mean":[]}
        print(weight)
        for year in range(1979,2025):
            data = pd.read_csv(f"../../econometric_model_comparison/data/temp/{timeframe}/processed_by_country/{weight}/temp.{timeframe}.bycountry.{weight}.{year}.csv")
            if len(data.iloc[0]["country"]) == 2:
                data["ISO3"] = cc(data["country"], origin="fips", destination="iso3c")
            else:
                data["ISO3"] = data["country"]
            climate_columns = data[[col for col in data.columns if col.startswith(weight)]]
            data[f"{timeframe}_mean"] = np.mean(climate_columns, axis=1)
            for row in data.iterrows():
                row = row[1]
                res_dict["year"].append(year)
                res_dict["ISO3"].append(row.country)
                res_dict[f"temp_{timeframe}_mean"].append(row[f"{timeframe}_mean"])
        if timeframe == "daily_max":
            directory = "daily_temp_max"
        else:
            directory = "daily_temp_min"
        pd.DataFrame.from_dict(res_dict).sort_values(["ISO3","year"]).reset_index(drop=True).to_csv(f"../src/climate_econometrics_toolkit/preprocessed_data/{directory}/{weight}/temp.{timeframe}.daily.bycountry.{weight}.csv")