import csv
import pandas as pd
import numpy as np
from countrycode import countrycode as cc
from calendar import monthrange

def write_regression_data_to_file(file, data):
	writer = csv.writer(file)
	headers =["country","admin1","year"]
	for column in data["Angola"]["Bengo"][2008]:
		headers.append(column.replace(" ","_").lower())
	writer.writerow(headers)
	for country, admin1_data in dict(sorted(data.items(), key=lambda x: x[0])).items():
		for admin1, year_data in dict(sorted(admin1_data.items(), key=lambda x: x[0])).items():
			for year, data in year_data.items():
				new_row = [country,admin1,year]
				for column in data:
					new_row.append(data[column])
				writer.writerow(new_row)

climate_data = pd.read_csv("../data/monthly_climate_data_by_geo_region.csv")
ag_data = pd.read_csv("../data/hvstat_africa_data_v1.0.csv")
regression_dataset = {}

for row in ag_data.iterrows():
	row = row[1]
	if row.country not in regression_dataset:
		regression_dataset[row.country] = {}
	if row.admin_1 not in regression_dataset[row.country]:
		regression_dataset[row.country][row.admin_1] = {}
	if row.harvest_year not in regression_dataset[row.country][row.admin_1]:
		regression_dataset[row.country][row.admin_1][row.harvest_year] = {}
	regression_dataset[row.country][row.admin_1][row.harvest_year]["crop_production"] = row.production
	regression_dataset[row.country][row.admin_1][row.harvest_year]["crop_yield"] = row["yield"]

climate_labels = {"temp":"air_temp","precip":"precip","humidity":"humidity"}
omitted_countries = set()
for climate_var in ["temp","precip","humidity"]:
	aggregate_var = "mean"
	for row in climate_data.iterrows():
		row = row[1]
		country = row.country
		if country == "Tanzania":
			country = "Tanzania, United Republic of"
		if country not in regression_dataset:
			omitted_countries.add(country)
		elif row.admin1 in regression_dataset[country]:
			for year in range(1960,2024):
				if year in regression_dataset[country][row.admin1]:
					monthly_climate_vals = []
					for month in range(1,13):
						if month < 10:
							month = "0" + str(month)
						monthly_climate_vals.append(row[f"{climate_labels[climate_var]}.weighted_mean.X{year}.{month}.01"])
					annual_climate_mean = np.mean(monthly_climate_vals)
					if climate_var == "temp":
						# celsius to kelvin
						annual_climate_mean = annual_climate_mean - 273.15
					elif climate_var == "precip":
						# precipitation rate per second to total monthly precipitation (X by approx. # of seconds in a month)
						annual_climate_mean = annual_climate_mean * 2.628e+6
					regression_dataset[country][row.admin1][year][f"{climate_var}"] = annual_climate_mean

with open("../data/regression_dataset.csv", "w") as regression:
	write_regression_data_to_file(regression, regression_dataset)

print("Omitted countries:", list(omitted_countries))