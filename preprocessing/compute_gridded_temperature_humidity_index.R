setwd("econometric_model_comparison/data")

library("exactextractr")
library("sf")
library("stringr")
library("raster")
library("weathermetrics")
library("terra")

pop_raster = raster("gpw-v4-population-density-rev11_2000_30_sec_tif/gpw_v4_population_density_rev11_2000_30_sec.tif")
crop_raster = raster("CroplandPastureArea2000_Geotiff/Cropland2000_5m.tif")
rice_raster = raster("crop_maps/harvested_area/rice_HarvestedAreaFraction.tif")
maize_raster = raster("crop_maps/harvested_area/maize_HarvestedAreaFraction.tif")
soybean_raster = raster("crop_maps/harvested_area/soybean_HarvestedAreaFraction.tif")
wheat_raster = raster("crop_maps/harvested_area/wheat_HarvestedAreaFraction.tif")

single_climate_raster = stack("temp/daily/shifted/air.2m.gauss.1961.shifted.nc")

pop_raster = resample(pop_raster, single_climate_raster)
crop_raster = resample(crop_raster, single_climate_raster)
rice_raster = resample(rice_raster, single_climate_raster)
maize_raster = resample(maize_raster, single_climate_raster)
soybean_raster = resample(soybean_raster, single_climate_raster)
wheat_raster = resample(wheat_raster, single_climate_raster)

weight_vec = c(pop_raster, crop_raster, rice_raster, maize_raster, soybean_raster, wheat_raster)
weight_vec_names = c("popweighted","cropweighted","riceweighted","maizeweighted","soybeanweighted","wheatweighted")

country_shapes = read_sf("country_shapes",layer="country")

kelvin_to_celsius <- function(x) {
	x - 273.15
}

temp_humidity_index <- function(temp, rel_hum) {
	heat.index(t = temp, dp = c(), rh = rel_hum, temperature.metric = "celsius")
}

for (year in 1948:2024) {
	print(year)
	temp_raster = stack(str_interp("temp/daily/shifted/air.2m.gauss.${year}.shifted.nc"))
	rel_hum_raster = stack(str_interp("rel_humidity/daily/shifted/rhum.sig995.${year}.shifted.nc"))
	temp_raster_c = calc(temp_raster, kelvin_to_celsius)
	rel_hum_raster = resample(rel_hum_raster, temp_raster_c)
	thi_raster = overlay(temp_raster_c, rel_hum_raster, fun = temp_humidity_index)

	thi_extracted = exact_extract(thi_raster, country_shapes, fun = "mean")
	data = c()
	data$ISO3 <- country_shapes$GMI_CNTRY
	data$thi <- thi_extracted
	write.csv(data, str_interp("THI/THI.${year}.unweighted.csv"))

	index = 0
	for (vec in weight_vec) {
		thi_extracted = exact_extract(thi_raster, country_shapes, fun = "weighted_mean", weights=weight_vec)
		data = c()
		data$ISO3 <- country_shapes$GMI_CNTRY
		data$thi <- thi_extracted
		weight_vec_name = weight_vec_name[index]
		write.csv(data, str_interp("THI/THI.${year}.${weight_vec_name}.csv"))
		index = index + 1
	}
}
