import pandas as pd
import numpy as np
import os

from importlib.resources import files

import geopandas as gpd
import xarray as xr
from exactextract import exact_extract
import progressbar
import calendar

import climate_econometrics_toolkit.utils as utils

cet_home = os.getenv("CETHOME")


def kelvin_to_celsius(x):
	return x - 273.15


def degree_days(x, threshold, mode, second_threshold=None):
	if mode == "above":
		return np.where(x >= threshold, x-threshold, 0)
	elif mode == "below":
		return np.where(x <= threshold, threshold-x, 0)
	else:
		return np.where((x >= threshold) & (x <= second_threshold), x-threshold, 0)


def compute_gridded_degree_days(years, countries, threshold, mode, panel_column_name, time_column_name, crop, second_threshold, col_name):
	utils.print_with_log(f"Please be aware that computing degree days using gridded daily temperature data is a time-consuming process.", "warning")
	if crop is not None:
		col_name += f"_{crop}_growing_season"
		# if crop specified, get growing season dates for specified crop
		country_start_days, country_end_days = utils.get_growing_season_data_by_crop(crop)
	shape_file = files("climate_econometrics_toolkit.preprocessed_data.shape_files.country_shapes").joinpath("country.shp")
	shape_file = gpd.read_file(shape_file)
	res = {panel_column_name:[], time_column_name:[], col_name:[]}
	for year in progressbar.progressbar(years):
		raster_data = xr.open_dataset(f"{cet_home}/daily_temp_mean/gridded/air.2m.gauss.{year}.shifted.nc")
		raster_c = kelvin_to_celsius(raster_data["air"])
		raster_dd = degree_days(raster_c, threshold, mode, second_threshold)
		xr.DataArray(
			data=raster_dd,
			dims=raster_data['air'].dims,
			coords=raster_data['air'].coords,
			name="degree_days"
		).to_netcdf(f"{cet_home}/raster_output/dd_{year}.nc")
		extracted = exact_extract(f"{cet_home}/raster_output/dd_{year}.nc", shape_file, "mean")
		country_degree_days = {}
		year_periods = 1461
		if calendar.isleap(year):
			year_periods = 1465
		for index, country in enumerate(shape_file["GMI_CNTRY"]):
			if country in countries:
				country_degree_days[country] = []
				for day_obs in range(1,year_periods,4):
					country_degree_days[country].append(np.mean([
						extracted[index]["properties"][f"band_{day_obs}_mean"],
						extracted[index]["properties"][f"band_{day_obs+1}_mean"],
						extracted[index]["properties"][f"band_{day_obs+2}_mean"],
						extracted[index]["properties"][f"band_{day_obs+3}_mean"]
					]))
				if crop is not None:
					if country_end_days[country] < country_start_days[country]:
						country_degree_days[country] = np.concatenate([country_degree_days[country][:int(country_end_days[country])+1],country_degree_days[country][int(country_start_days[country]):]])
					else:
						country_degree_days[country] = country_degree_days[country][int(country_start_days[country]):int(country_end_days[country])+1]
				res[panel_column_name].append(country)
				res[time_column_name].append(year)
				res[col_name].append(int(np.sum(country_degree_days[country])))
	# cleanup temp raster files
	for file in os.listdir(f"{cet_home}/raster_output/"):
		if file.startswith("dd_"):
			os.remove(f"{cet_home}/raster_output/{file}")
	return pd.DataFrame.from_dict(res)


def compute_country_degree_days(years, countries, threshold, mode, panel_column_name, time_column_name, crop, second_threshold, col_name):
	if crop is not None:
		col_name += f"_{crop}_growing_season"
		# if crop specified, get growing season dates for specified crop
		country_start_days, country_end_days = utils.get_growing_season_data_by_crop(crop)
	res = {panel_column_name:[], time_column_name:[], col_name:[]}
	years_missing_data, countries_missing_temp_data, countries_missing_crop_data = set(), set(), set()
	for year in years:
		try:
			file = files(f"climate_econometrics_toolkit.preprocessed_data.daily_temp.unweighted").joinpath(f'temp.daily.bycountry.unweighted.{year}.csv')
			daily_temp_data = pd.read_csv(file)
			for country in countries:
				if country in daily_temp_data:
					if crop is None:
						# if no crop specified, compute degree days for entire year
						daily_temps = daily_temp_data[country]
					else:
						# if crop specified, extract only crop growing days
						try:
							if country_end_days[country] < country_start_days[country]:
								daily_temps = pd.concat([daily_temp_data[country].iloc[:int(country_end_days[country])+1],daily_temp_data[country].iloc[int(country_start_days[country]):]])
							else:
								daily_temps = daily_temp_data[country].iloc[int(country_start_days[country]):int(country_end_days[country])+1]
						except (KeyError,ValueError):
							# except case where no crop growing season data exists
							daily_temps = None
					if daily_temps is not None:
						if mode == "above":
							degree_days = int(np.sum([val-threshold for val in daily_temps if val >= threshold]))
						elif mode == "below":
							degree_days = int(np.sum([threshold-val for val in daily_temps if val <= threshold]))
						elif mode == "between":
							degree_days = int(np.sum([val-threshold for val in daily_temps if val >= threshold and val <= second_threshold]))
					else:
						degree_days = pd.NA
						countries_missing_crop_data.add(country)
					res[panel_column_name].append(country)
					res[time_column_name].append(year)
					res[col_name].append(degree_days)
				else:
					countries_missing_temp_data.add(country)
		except FileNotFoundError:
			years_missing_data.add(year)
	if len(countries_missing_temp_data) > 0:
		utils.print_with_log(f"No daily temperature data available for countries: {sorted(countries_missing_temp_data)}", "warning")
	if len(countries_missing_crop_data) > 0:
		utils.print_with_log(f"No {crop} growing season data available for countries: {sorted(countries_missing_crop_data)}", "warning")
	if len(years_missing_data) > 0:
		utils.print_with_log(f"No daily temperature data available for years: {sorted(years_missing_data)}", "warning")        
	return pd.DataFrame.from_dict(res)
