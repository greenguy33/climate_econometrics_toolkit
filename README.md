![graphical abstract](figures/graphical_abstract.png)

This project contains both a user interface and a Python API designed for helping researchers with climate impact modeling using econometric-style regression models. This project has been developed as part of my PhD research. It is a work in progress. Please reach out with any questions or make an issue.

Developer contact: Hayden Freedman (hfreedma@uci.edu)

This README provides a brief overview of the toolkit and instructions for installation. For further documentation, see the [API Quickstart Guide](docs/api_quickstart.md), [Interface Quickstart Guide](docs/interface_quickstart.md) and the [API Documentation](docs/api_documentation.pdf). In addition, for those users just getting started with climate impacts research, please see the [Guide for New Users](docs/guide_for_new_users.md), which provides some guidance as to the basics of climate econometrics modeling, which types of models are appropriate to construct using the tool, and some pointers to helpful external resources.

# Is this the right tool for you?

The Climate Econometrics Toolkit is designed to help researchers interested in using econometric regression models to calculate the downstream societal impacts of climate change.
While the tool is not limited to this use case, and theoretically could be used to analyze data from other domains, certain features have been added with the assumption of the climate data use case.

If this tool will be helpful to you, you most likely...

- Are a climate impacts researcher or statistician interested in econometric-style analyses
- Have some basic knowledge of econometric regression and working with panel data
- Are interested in putting together a workflow for modeling climate impacts without starting completely from scratch
- Are familiar with Python (API only)

If you are aren't sure what type of analysis this project is designed to help with, I recommend checking out the paper [Anthropogenic climate change has slowed global agricultural productivity growth](https://www.nature.com/articles/s41558-021-01000-1)
by Ortiz-Bobea et al. as an example. I based some of the implementations in this tool off of the codebase attached to this paper and implemented a reproduction of this paper using the tool (see code [here](notebooks/ortiz-bobea-reproduction.ipynb)).

# Overview

After analyzing the workflows of several climate econometric research papers, I identified a three-step workflow which many of these papers have in common.

![three step workflow](figures/cet_fig1.png "Three Step Climate Econometrics Workflow")

There are functions available in this tool to help with each step of this workflow. Ideally, this can help reduce code duplication between papers, improve reproducibility of paper results, and reduce barriers to entry for researchers entering the field.

**For Step 1**, the toolkit leverages the [exactextract](https://github.com/isciences/exactextract) package along with some extra functionality to help you aggregate your data from multiple sources, including gridded raster data, to a single regression-ready panel dataset. You provide your raster data (usually .nc or .tif), and you get a CSV file with data aggregated to the country level using built-in country shapes. You can optionally apply weights to the extraction using built-in weight files for population, croplands, or the harvested area of various crops (maize, wheat, soybean, and rice are currently available). Alternatively, providing your own shape and weight file is also an option. Once you have your data from various sources aggregated to the same spatio-temporal level, use the toolkit's `integrate` function to merge them into a single dataframe ready to use with Step 2. Pre-processed datasets are also available for users who want to get started quickly.

**For Step 2**, you can use the user interface to visually construct and evaluate different models. Models are automatically evaluated using ten-fold cross-validation, and several metrics are available to compare models directly in the interface. The timeline feature lets you easily return to previous models and see which models are the best performing. After choosing your best model, you can export the coefficients as a CSV file or you can run bootstrapping or Bayesian inference from within the interface to generate coefficient samples that can be used to seed your impact projections with uncertainty. A model cache saves model results and canvas state for each dataset, which persists between sessions. All of these features are also available from the API if you prefer to programmatically construct and evaluate your model.

**For Step 3**, you provide your counterfactual or climate projection data and you get predictions of your dependent variable based on your model. If you ran bootstrapping or Bayesian inference, a prediction for each sample will be calculated (so if you have, for example, 1,000 bootstrap samples, you'll get 1,000 predictions of the dependent variable for each row of climate data input). You can then use these predictions to assess historical attribution, compute historical cumulative sums, or project future impacts.

# Installation

The package is currently hosted on TestPyPi. When installing, use the commands below to install all dependencies from PyPi in addition to the TestPyPi repo. Python 3 must be installed prior to installation.

#### Create and enter new virtual environment (optional)

```
python3 -m venv climate_econometrics_toolkit.venv
source climate_econometrics_toolkit.venv/bin/activate
```

#### Install package and dependencies

```
pip install climate-econometrics-toolkit
```

#### Add environmental variable CET_HOME

In order for all features to work properly, the environmental variable `CET_HOME` must be set. Exported files will be saved to this directory.

#### Start

To start the interface, simply open a Python shell and execute the following commands:

```
from climate_econometrics_toolkit import interface_api as api
api.start_interface()
```

The interface should launch in a separate window.


# Climate Econometrics Toolkit Interface

The Climate Economterics Toolkit interface consists of 4 main sections.

![Interface Overview](figures/overview_fig.png)

1. **Point-and-click model construction:** Using the top right panel, variables from a loaded dataset can be dragged on the canvas. Arrows can be added by first clicking a source node, then clicking a target node. A regression model is built by adding arrows from all the model covariates and effects to a single dependent (target) variable. Arrows can be removed by right-clicking.
2. **Data transformations:** Right clicking on a variable opens up a transformation menu, which enables automatic application of a set of pre-programmed transformations, such as squaring, log transforming, first differencing, and lagging a variable. Fixed-effects, random-effects, and time trends can also be created from this menu.
3. **Out-of-sample evaluation:** Once a model has been constructed on the canvas, the "Evaluate Model with OLS" button triggers an automatic evaluation of the model on a portion of the dataset that is automatically withheld. Once the evaluation is complete, the results of metrics such as $R^2$ and root mean squared error (RMSE), as well as p-values for each covariate, appear in the bottom left of the interface.
4. **Model history timeline:** Once a model has been evaluated, it appears on the timeline panel in the bottom right as a blue dot. The timeline can be adjusted by clicking on any of the metrics boxes, which rebuilds the timeline to show how each model performed based on the selected metric. Clicking any blue dot restores the state of the corresponding model to the canvas.
5. **Model cache (not pictured):** All aspects of the model, including coefficient values, metrics, timeline, and canvas state, are saved in a cache, which automatically reload when the same dataset is re-loaded, even if the program is closed and re-opened.

See the [Interface Quickstart Guide](docs/quickstart.md) for more details.

# Availability of Pre-Processed Data

The toolkit provides a series of pre-processed datasets, weight files, and crop growing-season masks to help you get up and running quickly. Available weight files are shown in the table below. These files can be automatically used with raster extraction, following the code example below. 

| Weight File Source                                                      | Weight File Content                                                                     | Granularity  |
|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------|
| NASA Gridded Population of the World (Doxsey 2015)             | Population density                                                                      | 5 arcminutes |
| EarthStat Cropland Harvested Area Map (Ramankutty 2008)       | Harvested area (all crops)                                                              | 5 arcminutes |
| EarthStat Individual Crops Harvested Area Map (Monfreda 2008) | Maize harvested area, Wheat harvested area, Rice harvested area, Soybean harvested area | 5 arcminutes |

```
# Argument to 'weights' must be one of: "popweighted","cropweighted","maizeweighted","riceweighted","soybeanweighted","wheatweighted"
# This call uses a built-in shape file that countains country shapes. The 'shape_file' argument can be used to pass a custom shape file via its filepath.
extracted = api.extract_raster_data(path_to_raster_file, weights="cropweighted")

# The argument '12' indicates that the raster file contains monthly data and to aggregate it yearly
# The optional argument 'maize' to the 'crop' parameter indicates to only process raster layers that exist over the maize growing season for each country
aggregated = api.aggregate_raster_data(extracted, "temp", "mean", 12, 1948, crop="maize")
```
Note that crop growing season masks, such as used in the above argument to the 'crop' parameter, are from Sacks et al. 2010., _Crop planting dates: an analysis of global patterns_.

For users who wish to avoid the raster extraction process entirely, the toolkit also provides several datasets already extracted and processed as CSV files at the country/year level. The table below shows the available datasets, and the code example below shows how one can integrate two or more of these datasets into a single regression-ready dataset. Asterisks next to the data source indicate that the file was aggregated from gridded data to the country/year level and is available with various weights applied, using arguments to the `weight' parameter.

| Data Source                                                                          | Data Variables                                                                    | Years in Dataset | Num. Countries in Dataset |
|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|------------------|---------------------------|
| NCEP-NCAR Reanalysis* (Kalnay 2018)                                         | Surface Air Temperature, Precipitation, Specific Humidity, Relative Humidity                        | 1948-2024        | 235           
| CPC Global Unified Temperature | Maximum and Minimum Surface Air Temperature | 1979-2024 | 240
| SPEIbase* (Begueria 2010)                                              | Standardized Precipitation-Evapotranspiration Index                               | 1901-2023        | 240                       |
| PKU Global Inventory Modeling and Mapping Studies NDVI* (Li 2023) | Normalized Difference Vegetation Index                                            | 1982-2022        | 162                       |
| FAOStat Database (Kasnakoglu 2006)                                 | Total Food, Primary Cereals, Agriculture, Livestock, and Crops Production Indices | 1961-2023        | 199                       |
| USDA FDA (USDA 2024)                                              | Agricultural Total Factor Productivity                                            | 1961-2021        | 182                       |
| World Bank Development Indicators (World Bank 2024)                             | Per-Capita Gross Domestic Product                                                 | 1961-2022        | 262                       |
| EM-DAT International Disaster Database (Delforge 2023)                         | Drought, Heat Wave, Wildfire, Flood                                               | 1960-2024        | 200                       |
```
ndvi_data = api.load_ndvi_data(weight='cropweighted')
spei_data = api.load_spei_data(weight='cropweighted')
# loads NCEP-NCAR dataeeee
clim_data = api.load_climate_data(weight='cropweighted')
ag_tfp_data = api.load_usda_fda_data()
reg_data = api.integrate([ndvi_data,spei_data,clim_data,ag_tfp_data], keep_na=False)
```

Preprocessed temperature-humidity index and dewpoint temperature datasets are also included. These are derived from the NCEP-NCAR surface air temperature and relative humidity reanalysis data. See the API documentation for details.

# Available Regression Models and Estimators

The toolkit makes several different types of regressions and estimators available, which are designed to suit a variety of different use cases in the field of climate econometrics. The table below summarizes the types of regressions that can be used to fit specified models:

| **Regression Models & Estimators** | **Standard Errors Supported** | **Implementation** |
|-----------------------------------|-------------------------------|---------------------|
| **POLS** (Pooled OLS)  <br> **FEOLS** (Fixed Effects via OLS)  <br> **Random Effects** via REML | - Non-robust SE  <br> - Clustered SE  <br> - Driscoll-Kraay SE  <br> - White-Huber SE  <br> - Newey-West SE | Statsmodels |
| **Quantile Regression** via IWLS | - Non-robust SE  <br> - Greene SE | Statsmodels |
| **Panel Spatial Lag Model** via ML <br> **Panel Spatial Error Model** via ML | - Non-robust SE   | PySAL/spreg |

# Computing Growing and Killing Degree Days

The toolkit provides functionality to automatically compute growing and killing degree days for the entire year as well as various crop growing seasons, and automatically integrates them with existing regression data. See the methods `compute_degree_days` and `add_degree_days_to_dataframe` (which calls `compute_degree_days` under the hood) in the API documentation. 

Degree days can be calculated using one of two methods of computation: `country` and `grid`. `country` computation is less accurate but faster, as it simply uses the mean daily temperature for entire countries. `grid` computation is more time-consuming, but captures heterogeneous temperature effects that are present at the grid level. The computation mode is specified by the `computation` argument to both of the methods mentioned above. In order to use the `grid` computation method, you must download the gridded daily temperature data available [here](https://drive.google.com/drive/folders/1c7bl8vR9HtIUxDTYMFB3HyfJRyqcOQQQ?usp=drive_link) and place it in your `cet_home` directory with the sub-directory structure `daily_temp_mean/gridded/...`.
