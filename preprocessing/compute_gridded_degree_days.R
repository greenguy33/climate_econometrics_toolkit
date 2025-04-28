# setwd("econometric_model_comparison/data")

library("exactextractr")
library("sf")
library("stringr")
library("raster")

kelvin_to_celsius <- function(x) {
	x - 273.15
}

degree_days <- function(x) {
	ifelse(x > threshold, x - threshold, 0)
}

threshold = 10

country_shapes = read_sf("country_shapes",layer="country")
climate_raster = climate_raster = stack("temp/daily/shifted/air.2m.gauss.1961.shifted.nc")

climate_raster_c = calc(climate_raster, kelvin_to_celsius)
climate_raster_dd = calc(climate_raster_c, degree_days)
dd_extracted = exact_extract(climate_raster_dd, country_shapes, fun = "mean")