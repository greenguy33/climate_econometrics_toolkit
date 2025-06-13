import pandas as pd
import numpy as np

for weight in ["unweighted","popweighted","cropweighted","maizeweighted","riceweighted","wheatweighted","soybeanweighted"]:
    data_dict = {"ISO3":[],"year":[],f"dewpoint_temperature_{weight}":[]}
    for year in range(1948,2025):
        data = pd.read_csv(f"../../econometric_model_comparison/data/dewpoint/dewpoint.{year}.{weight}.csv")
        climate_cols = data[[col for col in data.columns if col.startswith("dewpoint")]]
        data["year_means"] = np.mean(climate_cols, axis=1)
        for row in data.iterrows():
            row = row[1]
            data_dict["ISO3"].append(row.ISO3)
            data_dict["year"].append(year)
            data_dict[f"dewpoint_temperature_{weight}"].append(row.year_means)
    pd.DataFrame.from_dict(data_dict).sort_values(["ISO3","year"]).reset_index(drop=True).to_csv(f"../src/climate_econometrics_toolkit/preprocessed_data/dewpoint_temperature/dewpoint_temperature_{weight}_1948_2024.csv")