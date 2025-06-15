# New Users Guide

This brief guide is intended as a practical introduction to some of the more common practices in climate econometrics, with code snippets showing use of the toolkit's API to implement these techniques are also included. This guide is not intended as an overview of the field, nor as a comprehensive guide to the techniques; see the _Further Reading_ section for resources that can provide a more in-depth introduction to the field.

## Climate Econometrics Basics

Climate econometrics is a field designed to study "how social, economic, and biophysical systems respond to weather" (Rising et al., 2020). To do so, weather data is combined with economic indicator data (i.e., agricultural productivity data, GDP data) and a regression model is fit to these data. Impacts are then computed based on the coefficient estimates derived from the regression output. This section defines a few key terms that newcomers to the field may be unfamiliar with, but which are useful for understanding the basics of a climate econometrics workflow.

* `Panel Data`: The main type of data that is used in climate econometrics analyses. Panel data consists of observations of a set of variables that are tracked across space (typically countries or lower-level administrative units), and across time (typically years, sometimes months or weeks). In the climate econometrics context, such data allows the construction of models that provide insights into how climate change has impacted macroeconomies over time and how the effect may vary based on geography.

As a brief example, consider this small dataset that shows temperature, precipitation, and GDP across two countries over two years:

| Country  | Year | Temperature (°C) | Precipitation (mm) | GDP (bn USD) |
| -------- | ---- | ---------------- | ------------------ | ------------ |
| Country A | 2023 | 15.2             | 820                | 1,800        |
| Country A | 2024 | 15.7             | 790                | 1,850        |
| Country B | 2023 | 22.3             | 560                | 2,300        |
| Country B | 2024 | 23.0             | 610                | 2,334        |
  
* `Gridded Climate Data`: Weather data is often available as a matrix of values corresponding to cells of a fixed dimension that are projected across a representation of the globe. Such data allows for more granular insights regarding climate and weather than if it were reported at the administrative level; however, in climate econometrics, it is often necessary to aggregate this data to the administrative level (see subsequent bullet). Also known as "raster data". Common formats for such data are NetCDF and TIF, and it can often be freely downloaded from public repositories.
  
* `Raster Data Extraction`: In order to accomodate a model that incorporates both economic data, which is often observed at the level of administrative units, and climate data, which is observed at the grid level, it is necessary to "extract" the climate data based on a shape file defining the relevant geography in order to obtain a single value of the climate variable for each administrative unit. This process may mean the gridded observations across the defined geography (for example, to generate the mean temperature for a given country/year), or in some cases sum them (for example, to generate the total precipitation for a given country/year). The toolkit's default strategy is to use area weighting, meaning that each grid cell falling fully within the shape of a given administrative unit is counted with weight of 1, while grid cells falling on the border between administrative units are weighted based on the fraction of the cell that is within the given administrative unit. Additional weighting schemes assign weights to grid cells based on population (often used in studies assessing the impact of climate change on economic output, or GDP) and cropland (often used to assess the impact of climate change on agricultural productiivty).

In order to extract raster data using the toolkit API, consider a gridded climate data file alongside a shape file defining the boundaries of the administrative units. The gridded data contains 4X daily temperature observations (meaning a total of 1460 observations per year), starting in the year 1948:

```
extracted = api.extract_raster_data("path_to_gridded_climate_temperature_data.nc", my_shape_file, weights="cropweighted")
aggregated = aggregate_raster_data_to_year_level(extracted, "temperature", "mean", 1460, 1948)
```
  
* `Fixed Effects`: In the context of climate econometrics, "fixed effects" typically refers to "fixed" (non-random) intercepts that are learned individually for each entity in the panel data. For example, the equation below shows individual intercepts $\alpha$ learned for each country _i_:
  
$GDP_{it} = \beta_1 \cdot Temp_{it} + \beta_2 \cdot Prec_{it} + \alpha_i + \varepsilon_{it}$

Such a model accounts for differences between individual countries that may affect the dependent variable (in this case, GDP) external to the impacts of climate on this variable. In addition, it is also common to use year fixed effects to account for temporary global phenomena that affect the dependent variable. To add fixed effects to a model using the toolkit API, see the code snippets below (for country fixed effects, year fixed effects, and both) which can be applied during model construction.

```
api.add_fixed_effects("Country")
```
```
api.add_fixed_effects("Year")
```
```
api.add_fixed_effects(["Country","Year"])
```
  

* `Random Effects`: In the context of climate econometrics, "random effects" typically refers to learning multiple coefficients for a given model covariate based on groups defined by either the geography or time columns in the panel data. These effects are "random" because, while the coefficients are allowed to vary across time or administrative units, the values of the coefficients are drawn from a single, normal distribution that is learned simultaneously to the group level coefficients. Such a model enables variance in the studied effect across panel entities while still assuming that the studied effect operates according to a similar pattern across all entities.

For example, consider that we want to learn the individual effect of temperature on GDP for each country in our dataset using a random slopes model. The equation below shows how this looks. $\beta_1$ is learned as a global coefficient and $u_{1i}$ is an offset from the global coefficient that exists separately for each country.

$GDP_{it} = (\beta_1 + u_{1i}) \cdot Temp_{it} + \beta_2 \cdot Prec_{it} + \alpha + \varepsilon_{it}$

The code snippet below shows how to apply random effects to the variable "Temperature", broken out by "Country". Note that the toolkit currently only supports one variable with random effects per model.

```
api.add_random_effect("Temperature", "Country")
```

* `Climate Projections`: After fitting a climate econometrics model, researchers are often interested in understanding the implications of the model by estimating future impacts. To do so, General Circulation Models (GCMs), which project future climate conditions under a variety of possible climatic scenarios, are often downloaded and used alongside the model results. These GCMs typically exist at the grid-cell level and thus require an extraction process similar to the historical gridded climate data that was input to the model. 

## Designing an Appropriate Model

There are a wide variety of econometrics models and use cases that can be used by the field. This section seeks to help guide newcomes to the field towards appropriate model choices based on an example use case. Although it is not possible to cover all possible scenarios, I attempted to provide a few pointers towards appropriate modeling decisions to help new users get started. It is also recommended for new users to look through the code in the `notebooks/` directory, specifically the two paper reproductions.

Consider the panel data example shown above, and consider that we would like to study the effect of temperature and precipitation on GDP at the country/year level. Let's assume we've already extracted the gridded climate data and integrated the data into a single regression dataset (see the `api.integrate` method for integration of various datasets that already exist at the same spatiotemporal level).

### Treatment of dependent variable

Consider applying the first difference of the natural log to the dependent variable (GDP) in climate econometrics models. This approach approximates the percentage change in the dependent variable while also accounting for non-stationarity.

```
api.add_transformation("GDP", ["ln","fd"])
```

### Treatment of independent variables

It is common to try models with quadratic or cubic transformations of the independent variables in order to account for non-linear effects. For example, we can make Temperature and Precipitation quadratic in the model by adding squared terms:

```
api.add_transformation("Temperature", "sq")
api.add_transformation("Precipitation", "sq")
```

### Group intercepts (fixed effects)

Adding fixed effects is very common in climate econometrics models, as it is a straightforward way to account for geography- and time- specific effects on the dependent variable that are not directly caused by the independent variables. Let's add both country and year fixed effects to the model:

```
api.add_fixed_effects(["Country","Year"])
```

### Time trends

Finally, we may choose to add time trends to the model, in order to account for country-specific changes over time that affect the dependent variable. For example, technological advances by a country can impact GDP over time. We can add quadratic time trends (the second argument as '2' in the code snippet below) in order to account for non-linear changes.

```
api.add_time_trend("Country", 2)
```

After applying all of these transformations, we might come to a model that looks like this (which is the model used in the paper [Burke et al.](https://www.nature.com/articles/nature15725)).

$\Delta \ln(GDP_{it}) = \beta_1 Temp_{it} + \beta_2 Temp_{it}^2 + \beta_3 Prec_{it} + \beta_4 Prec_{it}^2 + \alpha_i + \delta_t + \gamma_1 t + \gamma_2 t^2 + \varepsilon_{it}$

## Further Reading

The following links provide practical resources for further broadening one's understanding of climate impacts research.

* [_A practical guide to climate econometrics: Navigating key decision points in weather and climate data analysis_](https://climateestimate.net/content/getting-started.html) by James A. Rising et al. Journal of Open Science Education, 2020.
* [_The empirical analysis of climate change impacts and adaptation in agriculture_](https://www.sciencedirect.com/science/article/pii/S1574007221000025) by Ariel Ortiz-Bobea. Chapter 76, Handbook of Agricultural Economics, Volume 5, Pages 3981-4073, 2021.
* [_Climate Econometrics_](https://www.annualreviews.org/content/journals/10.1146/annurev-resource-100815-095343) by Solomon Hsiang. Annual Review of Resource Economics, Volume 8, Pages 43-75, 2016.
* [_Using Weather Data and Climate Model Output in Economic Analyses of Climate Change_](https://www.journals.uchicago.edu/doi/full/10.1093/reep/ret016) by Maximillian Auffhammer et al. Review of Environmental Economics and Policy, Volume 7, Number 2, 2013.
* [_What Do We Learn from the Weather? The New Climate-Economy Literature_](https://www.aeaweb.org/articles?id=10.1257/jel.52.3.740) by Melissa Dell et al. Journal of Economic Literature, Volume 52, Number 3, Pages 740-798, 2014.
