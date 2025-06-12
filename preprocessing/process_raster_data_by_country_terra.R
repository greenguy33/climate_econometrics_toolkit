setwd("econometric_model_comparison/data")

library("exactextractr")
library("sf")
library("stringr")
library("raster")
library("terra")

climate_vars = c("temp")
timeframes = c("daily_max","daily_min")

terra::gdalCache(1000)

country_shapes = read_sf("country_shapes",layer="country")
pop_raster = terra::rast("../data/gpw-v4-population-density-rev11_2000_30_sec_tif/gpw_v4_population_density_rev11_2000_30_sec.tif")
ag_raster = terra::rast("../data/CroplandPastureArea2000_Geotiff/Cropland2000_5m.tif")
maize_raster = terra::rast("crop_maps/harvested_area/maize_HarvestedAreaFraction.tif")
rice_raster = terra::rast("crop_maps/harvested_area/rice_HarvestedAreaFraction.tif")
wheat_raster = terra::rast("crop_maps/harvested_area/wheat_HarvestedAreaFraction.tif")
soybean_raster = terra::rast("crop_maps/harvested_area/soybean_HarvestedAreaFraction.tif")

single_climate_raster = terra::rast(str_interp("../data/temp/daily_max/shifted/tmax.1979.shifted.nc"))

pop_raster <- terra::resample(pop_raster, single_climate_raster)
pop_raster[is.na(pop_raster)] <- 0
ag_raster <- terra::resample(ag_raster, single_climate_raster)
ag_raster[is.na(ag_raster)] <- 0
maize_raster <- terra::resample(maize_raster, single_climate_raster)
maize_raster[is.na(maize_raster)] <- 0
rice_raster <- terra::resample(rice_raster, single_climate_raster)
rice_raster[is.na(rice_raster)] <- 0
wheat_raster <- terra::resample(wheat_raster, single_climate_raster)
wheat_raster[is.na(wheat_raster)] <- 0
soybean_raster <- terra::resample(soybean_raster, single_climate_raster)
soybean_raster[is.na(soybean_raster)] <- 0


lapply(climate_vars, function(climate_var) {
  lapply(timeframes, function(timeframe) {
    files <- list.files(path=str_interp("../data/${climate_var}/${timeframe}/shifted/"))
    lapply(files, function(file) {
      
      year = strsplit(file, split = "\\.")[[1]][2]
      climate_raster = terra::rast(str_interp("../data/${climate_var}/${timeframe}/shifted/${file}"))
      
      unweighted_by_country = exact_extract(climate_raster, country_shapes, fun = "mean")
      pop_weighted_by_country = exact_extract(climate_raster, country_shapes, fun = "weighted_mean", weights=pop_raster)
      ag_weighted_by_country = exact_extract(climate_raster, country_shapes, fun = "weighted_mean", weights=ag_raster)
      maizeweighted_by_country = exact_extract(climate_raster, country_shapes, fun = "weighted_mean", weights=maize_raster)
      riceweighted_by_country = exact_extract(climate_raster, country_shapes, fun = "weighted_mean", weights=rice_raster)
      wheatweighted_by_country = exact_extract(climate_raster, country_shapes, fun = "weighted_mean", weights=wheat_raster)
      soybeanweighted_by_country = exact_extract(climate_raster, country_shapes, fun = "weighted_mean", weights=soybean_raster)

      data <- c()
      data$country <- country_shapes$GMI_CNTRY
      data$unweighted_by_country <- unweighted_by_country
      write.csv(data, str_interp("../data/${climate_var}/${timeframe}/processed_by_country/unweighted/${climate_var}.${timeframe}.bycountry.unweighted.${year}.csv"))

      data <- c()
      data$country <- country_shapes$GMI_CNTRY
      data$popweighted_by_country <- pop_weighted_by_country
      write.csv(data, str_interp("../data/${climate_var}/${timeframe}/processed_by_country/popweighted/${climate_var}.${timeframe}.bycountry.popweighted.${year}.csv"))

      data <- c()
      data$country <- country_shapes$GMI_CNTRY
      data$agweighted_by_country <- ag_weighted_by_country
      write.csv(data, str_interp("../data/${climate_var}/${timeframe}/processed_by_country/agweighted/${climate_var}.${timeframe}.bycountry.agweighted.${year}.csv"))

      data <- c()
      data$country <- country_shapes$GMI_CNTRY
      data$maizeweighted_by_country <- maizeweighted_by_country
      write.csv(data, str_interp("../data/${climate_var}/${timeframe}/processed_by_country/maizeweighted/${climate_var}.${timeframe}.bycountry.maizeweighted.${year}.csv"))

      data <- c()
      data$country <- country_shapes$GMI_CNTRY
      data$riceweighted_by_country <- riceweighted_by_country
      write.csv(data, str_interp("../data/${climate_var}/${timeframe}/processed_by_country/riceweighted/${climate_var}.${timeframe}.bycountry.riceweighted.${year}.csv"))

      data <- c()
      data$country <- country_shapes$GMI_CNTRY
      data$wheatweighted_by_country <- wheatweighted_by_country
      write.csv(data, str_interp("../data/${climate_var}/${timeframe}/processed_by_country/wheatweighted/${climate_var}.${timeframe}.bycountry.wheatweighted.${year}.csv"))

      data <- c()
      data$country <- country_shapes$GMI_CNTRY
      data$soybeanweighted_by_country <- soybeanweighted_by_country
      write.csv(data, str_interp("../data/${climate_var}/${timeframe}/processed_by_country/soybeanweighted/${climate_var}.${timeframe}.bycountry.soybeanweighted.${year}.csv"))
    })
  })
})