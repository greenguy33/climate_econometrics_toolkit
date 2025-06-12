import pandas as pd
import numpy as np
from countrycode import countrycode as cc

for weight in ["unweighted","pop_weighted","ag_weighted","maize_weighted","rice_weighted","wheat_weighted","soybean_weighted"]:
    print(weight)
    final_dict = {}
    weight_no_dash = weight.replace("_","")
    for climate_var in ["temp","precip","spec_humidity","rel_humidity"]:
        print(climate_var)
        func = "mean"
        try:
            data = pd.read_csv(f"../../econometric_model_comparison/data/{climate_var}/monthly/processed_by_country/{weight}/{climate_var}.monthly.bycountry.{weight_no_dash}.{func}.csv")
        except FileNotFoundError:
            data = pd.read_csv(f"../../econometric_model_comparison/data/{climate_var}/monthly/processed_by_country/{weight_no_dash}/{climate_var}.monthly.bycountry.{weight_no_dash}.{func}.csv")
        for row in data.iterrows():
            row = row[1]
            country = row.country
            if len(country) == 2:
                country = cc(country, origin="fips", destination="iso3c")
            if country not in final_dict:
                final_dict[country] = {}
            for year in range(1948,2024):
                if year not in final_dict[country]:
                    final_dict[country][year] = {}
                monthly_climate_vals = []
                for month in range(1,13):
                    if month < 10:
                        month = "0" + str(month)
                    if weight == "unweighted":
                        monthly_climate_vals.append(row[f"{weight_no_dash}_by_country.mean.X{year}.{month}.01"])
                    else:
                        monthly_climate_vals.append(row[f"{weight_no_dash}_by_country.weighted_mean.X{year}.{month}.01"])
                annual_climate_mean = np.mean(monthly_climate_vals)
                if climate_var == "temp":
                    # celsius to kelvin
                    annual_climate_mean = annual_climate_mean - 273.15
                elif climate_var == "precip":
                    # precipitation rate per second to total monthly precipitation (X by approx. # of seconds in a month)
                    annual_climate_mean = annual_climate_mean * 2.628e+6
                final_dict[country][year][climate_var] = annual_climate_mean
    df = {"ISO3":[],"year":[],"temp":[],"precip":[],"spec_humidity":[],"rel_humidity":[]}
    for country, year_data in final_dict.items():
        for year, climate_data in year_data.items():
            df["ISO3"].append(country)
            df["year"].append(year)
            try:
                df["temp"].append(climate_data["temp"])
            except KeyError:
                df["temp"].append(np.NaN)
            try:
                df["precip"].append(climate_data["precip"])
            except KeyError:
                df["precip"].append(np.NaN)
            try:
                df["spec_humidity"].append(climate_data["spec_humidity"])
            except KeyError:
                df["spec_humidity"].append(np.NaN)
            try:
                df["rel_humidity"].append(climate_data["rel_humidity"])
            except KeyError:
                df["rel_humidity"].append(np.NaN)
    pd.DataFrame.from_dict(df).rename(columns={"temp":"temp_mean","precip":"precip_total","spec_humidity":"spec_humidity_mean","rel_humidity":"rel_humidity_mean"}).sort_values(["ISO3","year"]).reset_index(drop=True).to_csv(f"../src/climate_econometrics_toolkit/preprocessed_data/weather_data/NCEP_reanalaysis_climate_data_1948_2024_{weight_no_dash}.csv")