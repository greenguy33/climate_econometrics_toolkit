import csv
import pandas as pd
import numpy as np
from countrycode import countrycode as cc
from calendar import monthrange

def write_regression_data_to_file(file, data):
	writer = csv.writer(file)
	headers =["country","admin1","year","ssp1_temp","ssp1_precip","ssp5_temp","ssp5_precip"]
	writer.writerow(headers)
	for country, admin1_data in dict(sorted(data.items(), key=lambda x: x[0])).items():
		for admin1, year_data in dict(sorted(admin1_data.items(), key=lambda x: x[0])).items():
			for year, data in year_data.items():
				new_row = [country,admin1,year,data["ssp1"]["temp"],data["ssp1"]["precip"],data["ssp5"]["temp"],data["ssp5"]["precip"]]
				writer.writerow(new_row)

regression_dataset = {"ssp1":{},"ssp5":{}}
climate_labels = {"temp":"air_temp","precip":"precip"}
for ssp in ["ssp1","ssp5"]:
	climate_data = pd.read_csv(f"data/climate_data/cnrm_cmip6_{ssp}_gcm_projections.csv")
	for climate_var in ["temp","precip"]:
		aggregate_var = "mean"
		for row in climate_data.iterrows():
			row = row[1]
			if not pd.isnull(row.admin1):
				country = row.country
				if country == "Tanzania":
					country = "Tanzania, United Republic of"
				for year in range(2015,2101):
					monthly_climate_vals = []
					for month in range(1,13):
						if month < 10:
							month = "0" + str(month)
						row_string = f"{climate_labels[climate_var]}.weighted_mean.X{year}.{month}.15"
						try:
							monthly_climate_vals.append(row[row_string])
						except:
							row_string = f"{climate_labels[climate_var]}.weighted_mean.X{year}.{month}.16"
							monthly_climate_vals.append(row[row_string])
					annual_climate_mean = np.mean(monthly_climate_vals)
					if climate_var == "temp":
						# celsius to kelvin
						annual_climate_mean = annual_climate_mean - 273.15
					elif climate_var == "precip":
						# precipitation rate per second to total monthly precipitation (X by approx. # of seconds in a month)
						annual_climate_mean = annual_climate_mean * 2.628e+6
					if country not in regression_dataset:
						regression_dataset[country] = {}
					if row.admin1 not in regression_dataset[country]:
						regression_dataset[country][row.admin1] = {}
					if year not in regression_dataset[country][row.admin1]:
						regression_dataset[country][row.admin1][year] = {}
					if ssp not in regression_dataset[country][row.admin1][year]:
						regression_dataset[country][row.admin1][year][ssp] = {}
					regression_dataset[country][row.admin1][year][ssp][climate_var] = annual_climate_mean

filename = "data/cnrm_cmip6_regional_data.csv"

with open(filename, "w") as regression_file:
	write_regression_data_to_file(regression_file, regression_dataset)

# handle duplicate admin1 names
regression_dataset = pd.read_csv(filename)
for admin1 in set(regression_dataset["admin1"]):
	countries = set(regression_dataset.loc[regression_dataset["admin1"] == admin1]["country"])
	if len(countries) > 1:
		for country in countries:
			regression_dataset["admin1"] = np.where((regression_dataset["country"]==country) & (regression_dataset["admin1"] == admin1), admin1+"_"+country, regression_dataset["admin1"])
regression_dataset.to_csv(filename)

