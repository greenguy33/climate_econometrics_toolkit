setwd("econometric_model_comparison/data")

library("exactextractr")
library("sf")
library("stringr")
library("raster")
library("terra")

terra::gdalCache(3000)

spei_raster = stack("SPEI/spei01.nc")

country_shapes = read_sf("country_shapes",layer="country")

rice_raster = raster("crop_maps/harvested_area/rice_HarvestedAreaFraction.tif")
rice_raster = resample(rice_raster, spei_raster)
rice_raster[is.na(rice_raster)] <- 0

spei_raster = terra::rast(spei_raster)
riceweighted_by_country = exact_extract(spei_raster, country_shapes, fun = "weighted_mean", weights=rice_raster)

data <- c()
data$country <- country_shapes$GMI_CNTRY
data$riceweighted_by_country <- riceweighted_by_country
write.csv(data, str_interp("../data/SPEI/SPEI.daily.bycountry.riceweighted.csv"))