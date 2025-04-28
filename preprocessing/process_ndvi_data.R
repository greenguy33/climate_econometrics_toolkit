setwd("hierarchical_bayesian_drought_study_code/data")

library("exactextractr")
library("sf")
library("stringr")
library("raster")
library("terra")

country_shapes = read_sf("country_shapes",layer="country")

pop_raster = terra::rast("gpw-v4-population-density-rev11_2000_30_sec_tif/gpw_v4_population_density_rev11_2000_30_sec.tif")
crop_raster = terra::rast("CroplandPastureArea2000_Geotiff/Cropland2000_5m.tif")
rice_raster = terra::rast("crop_maps/harvested_area/rice_HarvestedAreaFraction.tif")
maize_raster = terra::rast("crop_maps/harvested_area/maize_HarvestedAreaFraction.tif")
soybean_raster = terra::rast("crop_maps/harvested_area/soybean_HarvestedAreaFraction.tif")
wheat_raster = terra::rast("crop_maps/harvested_area/wheat_HarvestedAreaFraction.tif")

single_climate_raster = terra::rast("PKU_GIMMS_NDVI_AVHRR_MODIS/PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated/PKU_GIMMS_NDVI_V1.2_19820101.tif")

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

files <- list.files(path=str_interp("PKU_GIMMS_NDVI_AVHRR_MODIS/PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated/"))
lapply(files, function(file) {
	ndvi_raster = terra::rast(str_interp("PKU_GIMMS_NDVI_AVHRR_MODIS/PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated/${file}"))
	ndvi_raster[ndvi_raster == 65535] <- 0

	ndvi_extracted = exact_extract(ndvi_raster, country_shapes, fun = "mean")
	colnames(ndvi_extracted) <- c("raw_ndvi","qc_layer")
	data = c()
	data$ISO3 <- country_shapes$GMI_CNTRY
	data$ndvi <- ndvi_extracted[1]
	write.csv(data, str_interp("PKU_GIMMS_NDVI_AVHRR_MODIS/extracted_with_weights/${file}.unweighted.csv"))

	index = 1
	for (vec in weight_vec) {
		ndvi_extracted = exact_extract(ndvi_raster, country_shapes, fun = "weighted_mean", weights=vec)
		colnames(ndvi_extracted) <- c("raw_ndvi","qc_layer")
		data = c()
		data$ISO3 <- country_shapes$GMI_CNTRY
		data$ndvi <- ndvi_extracted[1]
		weight_vec_name = weight_vec_names[index]
		write.csv(data, str_interp("PKU_GIMMS_NDVI_AVHRR_MODIS/extracted_with_weights/${file}.${weight_vec_name}.csv"))
		index = index + 1
	}
}
)