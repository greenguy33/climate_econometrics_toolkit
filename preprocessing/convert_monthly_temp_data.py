for weight in ["maize_weighted","rice_weighted","wheat_weighted","soybean_weighted"]:
    final_dict = {}
    weight_no_dash = weight.replace("_","")
    for climate_var in ["temp","precip","spec_humidity"]:
        func = "mean"
        if climate_var == "precip":
            func = "shifted"
        data = pd.read_csv(f"../../econometric_model_comparison/data/{climate_var}/monthly/processed_by_country/{weight}/{climate_var}.monthly.bycountry.{weight_no_dash}.{func}.csv")
        for row in data.iterrows():
            row = row[1]
            if row.country not in final_dict:
                final_dict[row.country] = {}
            for year in range(1948,2024):
                if year not in final_dict[row.country]:
                    final_dict[row.country][year] = {}
                monthly_climate_vals = []
                for month in range(1,13):
                    if month < 10:
                        month = "0" + str(month)
                    monthly_climate_vals.append(row[f"{weight_no_dash}_by_country.weighted_mean.X{year}.{month}.01"])
                annual_climate_mean = np.mean(monthly_climate_vals)
                if climate_var == "temp":
                    # celsius to kelvin
                    annual_climate_mean = annual_climate_mean - 273.15
                elif climate_var == "precip":
                    # precipitation rate per second to total monthly precipitation (X by approx. # of seconds in a month)
                    annual_climate_mean = annual_climate_mean * 2.628e+6
                final_dict[row.country][year][climate_var] = annual_climate_mean
    df = {"ISO3":[],"year":[],"temp":[],"precip":[],"spec_humidity":[]}
    for country, year_data in final_dict.items():
        for year, climate_data in year_data.items():
            df["ISO3"].append(country)
            df["year"].append(year)
            for climate_var, climate_item in climate_data.items():
                df[climate_var].append(climate_item)
    pd.DataFrame.from_dict(df).rename(columns={"temp":"temp_mean","precip":"precip_total","spec_humidity":"humidity_mean"}).sort_values(["ISO3","year"]).to_csv(f"NCEP_reanalaysis_climate_data_1948_2024_{weight_no_dash}.csv")