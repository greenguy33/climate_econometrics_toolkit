import os
import pandas as pd
import numpy as np
import calendar
import time

import geopandas as gpd
from exactextract import exact_extract
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

import climate_econometrics_toolkit.utils as utils

cet_home = os.getenv("CETHOME")


def resample_rasters(raster1, raster2):
	with rasterio.open(raster1) as weights, rasterio.open(raster2) as target_grid:
		raster_data = weights.read(1)
		raster_data[np.isnan(raster_data)] = 0
		weights_resampled = np.empty((target_grid.height, target_grid.width))
		reproject(
				source=raster_data,
				destination=weights_resampled,
				src_transform=weights.transform,
				src_crs=weights.crs,
				dst_transform=target_grid.transform,
				dst_crs=weights.crs,
				resampling=Resampling.bilinear
			)
		raster_id = raster1.split("/")[-1].rsplit(".",1)[0]
		save_file_path = f"{cet_home}/resampled_raster_files/{raster_id}_resampled.tif"
		with rasterio.open(save_file_path, "w",
		driver="GTiff",
		height=target_grid.height,
		width=target_grid.width,
		count=1,
		dtype=weights_resampled.dtype,
		crs=weights.crs,
		transform=target_grid.transform) as dst:
			dst.write(weights_resampled, 1)
		utils.print_with_log(f"Resampled weights file has been saved to {save_file_path}", "info")
		return save_file_path


def extract_raster_data(raster_file, shape_file, weight_file):
	if weight_file is None:
		utils.print_with_log("No weights file provided for extraction. Using uniform weights.", "info")
		return exact_extract(raster_file, shape_file, "mean")
	else:
		resampled_weight_file = resample_rasters(weight_file, raster_file)
		return exact_extract(raster_file, shape_file, "weighted_mean", weights=resampled_weight_file)


def make_subperiods_from_crop_growing_season(crop, geo_shapes, geo_identifier, subperiods_per_year):
	utils.assert_with_log(subperiods_per_year in [12,365,366], "Crop growing seasons can currently only be implemented at the monthly (12 subperiods/year) or daily (365/6 subperiods/year) levels.")
	country_start_days, country_end_days = utils.get_growing_season_data_by_crop(crop)
	utils.assert_with_log(len(set(geo_shapes[geo_identifier]).intersection(set(country_start_days.keys()))) != 0, "No overlap between supplied shape file/geo-identifier and ISO3 codes in crop data. Supplied shape file/geo-identifier must use ISO3 country codes in order to apply crop growing season data.")
	subperiods_to_use = {}
	for geoid in geo_shapes[geo_identifier]:
		if geoid in country_start_days and not pd.isnull(country_start_days[geoid]) and not pd.isnull(country_end_days[geoid]):
			if country_end_days[geoid] > country_start_days[geoid]:
				if subperiods_per_year in [365,366]:
					subperiods_to_use[geoid] = list(range(int(country_start_days[geoid]),int(country_end_days[geoid])+1))
				else:
					subperiods_to_use[geoid] = list(range(int(country_start_days[geoid]/30.417),int(country_end_days[geoid]/30.417)+1))
			else:
				if subperiods_per_year in [365,366]:
					subperiods_to_use[geoid] = list(range(int(country_end_days[geoid])+1)) + list(range(int(country_start_days[geoid]),367))
				else:
					subperiods_to_use[geoid] = list(range(int(country_end_days[geoid]/30.417)+1)) + list(range(int(country_start_days[geoid]/30.417),13))
	return subperiods_to_use


def make_leapyear_subperiods(subperiods_per_year):
	utils.assert_with_log((subperiods_per_year % 365 == 0) or (subperiods_per_year % 366 == 0), "Argument 'subperiods_per_year' at the sub-daily level must be a multiple of 365 or 366. Leap years will be automatically applied.")
	if subperiods_per_year % 365 == 0:
		reg_year_sbp = subperiods_per_year
		leap_year_sbp = subperiods_per_year + (subperiods_per_year / 365)
	elif subperiods_per_year % 366 == 0:
		reg_year_sbp = subperiods_per_year - (subperiods_per_year / 366)
		leap_year_sbp = subperiods_per_year
	return reg_year_sbp, leap_year_sbp


def aggregate_raster_data(
		raster_data, shape_file, climate_var_name, aggregation_func, geo_identifier, subperiods_per_year, starting_year, subperiods_to_use=None, crop=None
	):
	utils.assert_with_log(isinstance(subperiods_per_year, int), f"Value {subperiods_per_year} supplied as 'subperiods per time unit' argument is not an integer.")
	utils.assert_with_log(isinstance(starting_year, int), f"Value {starting_year} supplied as 'starting_year' argument is not an integer.")
	utils.assert_with_log(aggregation_func in ["sum","mean"], "Argument aggregation_func must be 'sum' or 'mean'")
	utils.assert_with_log(subperiods_to_use is None or crop is None, "Arguments 'subperiods_to_use' and 'crop' cannot both be supplied.")
	geo_shapes = gpd.read_file(shape_file)
	if crop is not None:
		subperiods_to_use = make_subperiods_from_crop_growing_season(crop, geo_shapes, geo_identifier, subperiods_per_year)
	reg_year_sbp, leap_year_sbp = None, None
	if subperiods_per_year > 365:
		reg_year_sbp, leap_year_sbp = make_leapyear_subperiods(subperiods_per_year)
	data = []
	omitted_geoids = set()
	for index, geo in enumerate(geo_shapes[geo_identifier]):
		# this removes the name of the aggregation function from the key
		new_dict = {}
		for key in raster_data[index]["properties"]:
			new_dict[key.split("_")[0] + "_" + key.split("_")[1]] = raster_data[index]["properties"][key]
		agg_mean = []
		period = starting_year
		subperiod = 0
		for index2, obs in enumerate(range(len(raster_data[index]["properties"]))):
			subperiod += 1
			if subperiods_to_use is not None and geo not in subperiods_to_use:
				omitted_geoids.add(geo)
			if subperiods_to_use is None or (geo in subperiods_to_use and subperiod in subperiods_to_use[geo]):
				agg_mean.append(new_dict[f"band_{str(obs+1)}"])
			if leap_year_sbp is None:
				condition = (subperiod == subperiods_per_year)
			elif calendar.isleap(period):
				condition = (subperiod == leap_year_sbp)
			else:
				condition = (subperiod == reg_year_sbp)
			if condition or index2 == len(raster_data[index]["properties"]) - 1:
				if aggregation_func == "sum":
					if len(agg_mean) > 0:
						data.append([geo, period, np.nansum(agg_mean)])
					else:
						data.append([geo, period, np.NaN])
				elif aggregation_func == "mean":
					data.append([geo, period, np.nanmean(agg_mean)])
				period += 1
				agg_mean = []
				subperiod = 0
	if len(omitted_geoids) > 0:
		utils.print_with_log(f"These GeoIDs were omitted from the aggregated dataset: {sorted(omitted_geoids)}", "warning")        
	return pd.DataFrame.from_records(data, columns=[geo_identifier,"time",climate_var_name]).sort_values([geo_identifier,"time"])