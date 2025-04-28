import pandas as pd
import numpy as np
from countrycode import countrycode as cc

for weight in ["unweighted","agweighted","popweighted"]:
    print(weight)
    for year in range(1948,2025):
        data = pd.read_csv(f"../../econometric_model_comparison/data/temp/daily/processed_by_country/{weight}/temp.daily.bycountry.{weight}.{year}.csv")
        if len(data.iloc[0]["country"]) == 2:
            data["ISO3"] = cc(data["country"], origin="fips", destination="iso3c")
        else:
            data["ISO3"] = data["country"]
        climate_columns = data[[col for col in data.columns if col.startswith(weight)]]
        for measurement in range(0,1464,4):
            data[f"daily_mean_{int(measurement/4)}"] = np.mean(climate_columns.iloc[:,measurement:measurement+4], axis=1) - 273.15
        columns_to_keep = ["ISO3"]
        for column in data.columns:
            if column.startswith("daily_mean"):
                columns_to_keep.append(column)
        data = data[columns_to_keep]
        data.set_index("ISO3").transpose().to_csv(f"../src/climate_econometrics_toolkit/preprocessed_data/daily_temp/{weight}/temp.daily.bycountry.{weight}.{year}.csv")