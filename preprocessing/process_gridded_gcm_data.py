import os
import pandas as pd
import numpy as np
import climate_econometrics_toolkit.user_api as api
from osgeo import gdal
import geopandas as gpd
import subprocess

cet_home = os.getenv("CETHOME")

reproduction_dir = cet_home + "/ortiz_bobea_reproduction/"
gcm_data_dir = f"{reproduction_dir}/gcms/"

# remove some non-populated areas from shape file
areas_to_remove = ["BV","JN","SV","DQ","FQ","HQ","JQ","MQ","WQ","JU","GO","WE","GZ"]
raster = gpd.read_file(f"{reproduction_dir}/country_shapes/country.shp")
raster2 = raster.loc[~raster.FIPS_CNTRY.isin(areas_to_remove)]
raster2 = gpd.GeoDataFrame(raster2)
raster2.to_file(f"{reproduction_dir}/country_shapes/country2.shp")
shape_file = reproduction_dir + "/country_shapes/country2.shp"

# shift climate raster data to correct scale
def create_shifted_file(filename, new_filename):
	cdo_command = ["cdo","sellonlatbox,-180,180,-90,90",filename,gcm_data_dir + "/" + gcm + "/" + new_filename]
	subprocess.run(cdo_command)

for gcm in os.listdir(gcm_data_dir):
	for file in os.listdir(gcm_data_dir + "/" + gcm):
		filename_segs = file.split(".")[0:-1]
		filename_segs.append("shifted")
		filename_segs.append("nc")
		new_filename = ".".join(filename_segs)
		if "shifted" not in file and not os.path.exists(gcm_data_dir + "/" + gcm + "/" + new_filename):
			if file.endswith(".nc"):
				create_shifted_file(gcm_data_dir + "/" + gcm + "/" + file, new_filename)


# process GCM data
weight_dir = f"{cet_home}/src/climate_econometrics_toolkit/preprocessed_data/raster_weights/"
weight_files = {
	"areaweighted":None,
	"cropweighted":f"{weight_dir}/cropland_weights/cropland_weights_5m.tif",
	"maizeweighted":f"{weight_dir}/cropland_weights/maize_weights_5m.tif",
	"soybeanweighted":f"{weight_dir}/cropland_weights/soybean_weights_5m.tif",
	"riceweighted":f"{weight_dir}/cropland_weights/rice_weights_5m.tif",
	"wheatweighted":f"{weight_dir}/cropland_weights/wheat_weights_5m.tif",
	"popweighted":f"{weight_dir}/population_weights/population_weights_5m.tif"
}

# for weight in ["areaweighted","cropweighted","popweighted","maizeweighted","soybeanweighted","riceweighted","wheatweighted"]:
for weight in ["popweighted"]:
	for gcm in os.listdir(gcm_data_dir):
		for grouping in ["historical", "hist-nat", "ssp245"]:
			base_dir = f"{cet_home}/src/climate_econometrics_toolkit/preprocessed_data/GCM_data/{gcm}/{weight}/"
			if not os.path.isfile(f"{base_dir}/{grouping}.csv"):
				group_files = []
				for file in os.listdir(gcm_data_dir + "/" + gcm):
					if file.endswith(".shifted.nc") and grouping in file:
						group_files.append(file)
				if gcm != "MRI" or grouping != "hist-nat":
					assert len(group_files) == 3, group_files
				else:
					assert len(group_files) == 6, group_files
				df = pd.DataFrame()
				for var in ["pr","tasmax","tasmin"]:
					files = [file for file in group_files if var in file]
					if gcm != "MRI" or grouping != "hist-nat":
						assert len(files) == 1, files
					else:
						assert len(files) == 2, files
					file_data = []
					for file in files:
						filepath = gcm_data_dir + "/" + gcm + "/" + file
						func = "sum" if var == "pr" else "mean"
						first_year_in_data = int(file.split("_")[-1].split("-")[0][0:4])
						raster = gdal.Open(filepath)
						print(file, filepath, shape_file, weight_files[weight], first_year_in_data, func, var)
						out = api.extract_raster_data(filepath, shape_file=shape_file, weight_file=weight_files[weight])
						file_data.append(api.aggregate_raster_data_to_year_level(out, var, "mean", 12, first_year_in_data, shape_file=shape_file, geo_identifier="GMI_CNTRY"))
							
					if gcm != "MRI" or grouping != "hist-nat":
						assert len(file_data) == 1
						data = file_data[0]
					else:
						assert len(file_data) == 2
						data = pd.concat([file_data[0], file_data[1]]).sort_values(["GMI_CNTRY","year"]).reset_index(drop=True)

					if "year" not in df:
						df["year"] = data["year"]
					if "ISO3" not in df:
						df["ISO3"] = data["GMI_CNTRY"]
					data_type = grouping.replace("-","_")
					if var == "pr":
						df[f"{data_type}_total_precipitation"] = data["pr"] * 2.628e+6
					elif var == "tasmax":
						df[f"{data_type}_temperature_max"] = data["tasmax"] - 273.15
					elif var == "tasmin":
						df[f"{data_type}_temperature_min"] = data["tasmin"] -273.15
				
				save_path = f"{base_dir}/{grouping}.csv"
				if not os.path.exists(base_dir):
					os.makedirs(base_dir)
				df.to_csv(save_path)