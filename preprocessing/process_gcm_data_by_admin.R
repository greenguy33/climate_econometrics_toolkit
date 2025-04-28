# setwd("climate_econometrics_toolkit/scripts")

library("exactextractr")
library("sf")
library("stringr")
library("raster")

geo_shapes = read_sf("../data/hvstat_boundary.gpkg")

temp_raster = stack("../data/climate_data/gcm_data/tas_Amon_CNRM-ESM2-1_ssp119_r1i1p1f2_gr_201501-210012.nc")
precip_raster = stack("../data/climate_data/gcm_data/pr_Amon_CNRM-ESM2-1_ssp119_r1i1p1f2_gr_201501-210012.nc")

data <- c()
data$country <- geo_shapes$ADMIN0
data$admin1 <- geo_shapes$ADMIN1
data$admin2 <- geo_shapes$ADMIN2

ag_raster = raster("../data/CroplandPastureArea2000_Geotiff/Cropland2000_5m.tif")
ag_raster <- resample(ag_raster, temp_raster)
ag_raster[is.na(ag_raster)] <- 0
data$air_temp = exact_extract(temp_raster, geo_shapes, fun = "weighted_mean", weights=ag_raster)
data$precip = exact_extract(precip_raster, geo_shapes, fun = "weighted_mean", weights=ag_raster)

write.csv(data, "../data/climate_data/cnrm_cmip6_ssp1_gcm_projections.csv")

temp_raster = stack("../data/climate_data/gcm_data/tas_Amon_CNRM-CM6-1_ssp585_r1i1p1f2_gr_201501-210012.nc")
precip_raster = stack("../data/climate_data/gcm_data/pr_Amon_CNRM-CM6-1_ssp585_r1i1p1f2_gr_201501-210012.nc")

data <- c()
data$country <- geo_shapes$ADMIN0
data$admin1 <- geo_shapes$ADMIN1
data$admin2 <- geo_shapes$ADMIN2

ag_raster = raster("../data/CroplandPastureArea2000_Geotiff/Cropland2000_5m.tif")
ag_raster <- resample(ag_raster, temp_raster)
ag_raster[is.na(ag_raster)] <- 0
data$air_temp = exact_extract(temp_raster, geo_shapes, fun = "weighted_mean", weights=ag_raster)
data$precip = exact_extract(precip_raster, geo_shapes, fun = "weighted_mean", weights=ag_raster)

write.csv(data, "../data/climate_data/cnrm_cmip6_ssp5_gcm_projections.csv")