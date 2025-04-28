setwd("econometric_model_comparison/data")

library("exactextractr")
library("sf")
library("stringr")
library("raster")
library("weathermetrics")
library("terra")

pop_raster = terra::rast("gpw-v4-population-density-rev11_2000_30_sec_tif/gpw_v4_population_density_rev11_2000_30_sec.tif")
crop_raster = terra::rast("CroplandPastureArea2000_Geotiff/Cropland2000_5m.tif")
rice_raster = terra::rast("crop_maps/harvested_area/rice_HarvestedAreaFraction.tif")
maize_raster = terra::rast("crop_maps/harvested_area/maize_HarvestedAreaFraction.tif")
soybean_raster = terra::rast("crop_maps/harvested_area/soybean_HarvestedAreaFraction.tif")
wheat_raster = terra::rast("crop_maps/harvested_area/wheat_HarvestedAreaFraction.tif")

single_climate_raster = terra::rast("temp/daily/shifted/air.2m.gauss.1961.shifted.nc")

pop_raster = terra::resample(pop_raster, single_climate_raster)
crop_raster = terra::resample(crop_raster, single_climate_raster)
rice_raster = terra::resample(rice_raster, single_climate_raster)
maize_raster = terra::resample(maize_raster, single_climate_raster)
soybean_raster = terra::resample(soybean_raster, single_climate_raster)
wheat_raster = terra::resample(wheat_raster, single_climate_raster)

pop_raster[is.na(pop_raster)] <- 0
crop_raster[is.na(crop_raster)] <- 0
rice_raster[is.na(rice_raster)] <- 0
maize_raster[is.na(maize_raster)] <- 0
soybean_raster[is.na(soybean_raster)] <- 0
wheat_raster[is.na(wheat_raster)] <- 0

weight_vec = list(pop_raster, crop_raster, rice_raster, maize_raster, soybean_raster, wheat_raster)
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
	temp_raster = terra::rast(str_interp("temp/daily/shifted/air.2m.gauss.${year}.shifted.nc"))
	rel_hum_raster = terra::rast(str_interp("rel_humidity/daily/shifted/rhum.sig995.${year}.shifted.nc"))
	temp_raster_c = terra::app(temp_raster, kelvin_to_celsius)
	crs(temp_raster_c) <- "EPSG:4326"
	crs(rel_hum_raster) <- "EPSG:4326"
	rel_hum_raster = terra::resample(rel_hum_raster, temp_raster_c)
	thi_raster = terra::lapp(x=terra::sds(temp_raster_c, rel_hum_raster), fun = temp_humidity_index)

	thi_extracted = exact_extract(thi_raster, country_shapes, fun = "mean")
	data = c()
	data$ISO3 <- country_shapes$GMI_CNTRY
	data$thi <- thi_extracted
	write.csv(data, str_interp("THI/THI.${year}.unweighted.csv"))

	index = 1
	for (vec in weight_vec) {
		thi_extracted = exact_extract(thi_raster, country_shapes, fun = "weighted_mean", weights=vec)
		data = c()
		data$ISO3 <- country_shapes$GMI_CNTRY
		data$thi <- thi_extracted
		weight_vec_name = weight_vec_names[index]
		write.csv(data, str_interp("THI/THI.${year}.${weight_vec_name}.csv"))
		index = index + 1
	}
}
