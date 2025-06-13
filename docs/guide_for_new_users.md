# New Users Guide

This brief guide is intended as a practical introduction to some of the more common techniques in climate econometrics, with code snippets showing use of the toolkit's API to implement these techniques are also included. This guide is not intended as an overview of the field, nor as a comprehensive guide to the techniques; see the _Further Reading_ section for resources that can provide a more in-depth introduction to the field.

## Climate Econometrics Basics

Climate econometrics is a field designed to study "how social, economic, and biophysical systems respond to weather" (Rising et al., 2020). To do so, weather data is combined with economic indicator data (i.e., agricultural productivity data, GDP data) and a regression model is fit to these data. Impacts are then computed based on the coefficient estimates derived from the regression output. This section defines a few key terms that newcomers to the field may be unfamiliar with, but which are useful for understanding the basics of a climate econometrics workflow.

* `Gridded Climate Data`
* `Raster Data Extraction`
* `Fixed Effects`
* `Random Effects`
* `Climate Projections`

## Designing an Appropriate Model

There are a wide variety of econometrics models and use cases that can be used by the field. This section seeks to help guide newcomes to the field towards appropriate model choices based on some common use cases. Although it is not possible to cover all possible scenarios, I attempted to provide a few pointers towards appropriate modeling decisions to help new users get started. It is also for new users recommended to look through the code in the `notebooks/` directory, specifically the two paper reproductions.

### Treatment of climate variables

appropriate weighting scheme
quadratic, cubic

### Treatment of economic variables

first difference of natural log (if dependent variable)

### Group intercepts (fixed effects)

geography, time

### Random slopes (random effects)

heterogeneous effects by geography

### Time trends

### Lagged variables

### Statistical tests

* Stationarity checks

panel unit root test

* Cross-sectional dependence checks

* Cointegration checks

### Regression models

* OLS

standard error selection

* Spatial Regression

* Quantile Regression

### Sampling-based inference

* Bayesian Inference

* Block Bootstrap


## Further Reading

The following links provide practical resources for further broadening one's understanding of climate impacts research.

* [_A practical guide to climate econometrics: Navigating key decision points in weather and climate data analysis_](https://climateestimate.net/content/getting-started.html) by James A. Rising et al. Journal of Open Science Education, 2020.
* [_The empirical analysis of climate change impacts and adaptation in agriculture_](https://www.sciencedirect.com/science/article/pii/S1574007221000025) by Ariel Ortiz-Bobea. Chapter 76, Handbook of Agricultural Economics, Volume 5, Pages 3981-4073, 2021.
* [_Climate Econometrics_](https://www.annualreviews.org/content/journals/10.1146/annurev-resource-100815-095343) by Solomon Hsiang. Annual Review of Resource Economics, Volume 8, Pages 43-75, 2016.
* [_Using Weather Data and Climate Model Output in Economic Analyses of Climate Change_](https://www.journals.uchicago.edu/doi/full/10.1093/reep/ret016) by Maximillian Auffhammer et al. Review of Environmental Economics and Policy, Volume 7, Number 2, 2013.
* [_What Do We Learn from the Weather? The New Climate-Economy Literature_](https://www.aeaweb.org/articles?id=10.1257/jel.52.3.740) by Melissa Dell et al. Journal of Economic Literature, Volume 52, Number 3, Pages 740-798, 2014.
