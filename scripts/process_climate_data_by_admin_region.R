setwd("climate_econometrics_toolkit/scripts")

library("exactextractr")
library("sf")
library("stringr")
library("raster")

geo_shapes = read_sf("../data/hvstat_boundary.gpkg")
ag_raster = raster("../data/CroplandPastureArea2000_Geotiff/Cropland2000_5m.tif")

temp_raster = stack("../data/climate_data/air.2m.mon.mean.shifted.nc")
precip_raster = stack("../data/climate_data/prate.mon.mean.shifted.nc")
humidity_raster = stack("../data/climate_data/shum.2m.mon.mean.shifted.nc")

data <- c()
data$country <- geo_shapes$ADMIN0
data$admin1 <- geo_shapes$ADMIN1
data$admin2 <- geo_shapes$ADMIN2

ag_raster <- resample(ag_raster, temp_raster)
ag_raster[is.na(ag_raster)] <- 0
data$air_temp = exact_extract(temp_raster, geo_shapes, fun = "weighted_mean", weights=ag_raster)

ag_raster <- resample(ag_raster, precip_raster)
ag_raster[is.na(ag_raster)] <- 0
data$precip = exact_extract(precip_raster, geo_shapes, fun = "weighted_mean", weights=ag_raster)

ag_raster <- resample(ag_raster, humidity_raster)
ag_raster[is.na(ag_raster)] <- 0
data$humidity = exact_extract(humidity_raster, geo_shapes, fun = "weighted_mean", weights=ag_raster)

write.csv(data, "../data/monthly_climate_data_by_geo_region.csv")