![graphical abstract](figures/graphical_abstract.png)

This project contains both a user interface and a Python API designed for helping researchers with climate impact modeling using econometric-style regression models. This project has been developed as part of my PhD research. It is a work in progress. Please reach out with any questions or make an issue.

Developer contact: Hayden Freedman (hfreedma@uci.edu)

There is an installation and quick start guide in this README. For more detailed instructions, you can reference the full [User Guide](USER_GUIDE.md).

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

**For Step 1**, I simply provided a wrapper for the [exactextract](https://github.com/isciences/exactextract) package with a little bit of extra functionality to help you aggregate/export your data. You provide your GCM data, a shape file, and optionally a weight raster file (in the case that you want to weight the climate data by population, croplands, or whatever else), and you get a CSV file with data aggregated to the level you specify.

**For Step 2**, you can use the user interface to visually construct and evaluate different models. Models are automatically evaluated on a withheld subset of your data and several metrics are available in the interface. There is also a model cache alongside a timeline feature that lets you easily return to previous models and see which models are the best performing. After choosing your best model, you can export the coefficients as a CSV file or you can run bootstrapping or Bayesian inference from within the interface to generate coefficient samples that can be used to seed your impact projections with uncertainty. All of these features are also available from the API if you prefer to programmatically construct and evaluate your model.

**For Step 3**, you provide your counterfactual or climate projection data and you get predictions of your dependent variable based on your model. If you ran bootstrapping or Bayesian inference, a prediction for each sample will be calculated (so if you have, for example, 1,000 bootstrap samples, you'll get 1,000 predictions of the dependent variable for each row of climate data input).

# Climate Econometrics Toolkit Interface

The Climate Economterics Toolkit interface consists of 4 main sections.

![Interface Overview](figures/overview_fig.png)

1. **Point-and-click model construction:** Using the top right panel, variables from a loaded dataset can be dragged on the canvas. Arrows can be added by first clicking a source node, then clicking a target node. A regression model is built by adding arrows from all the model covariates and effects to a single dependent (target) variable. Arrows can be removed by right-clicking.
2. **Data transformations:** Right clicking on a variable opens up a transformation menu, which enables automatic application of a set of pre-programmed transformations, such as squaring, log transforming, first differencing, and lagging a variable. Fixed effects and time trends can also be created from this menu.
3. **Out-of-sample evaluation:** Once a model has been constructed on the canvas, the "Evaluate Model" button triggers an automatic evaluation of the model on a portion of the dataset that is automatically withheld. Once the evaluation is complete, the results of metrics such as $R^2$ and root mean squared error (RMSE), as well as p-values for each covariate, appear in the bottom left of the interface.
4. **Model history timeline:** Once a model has been evaluated, it appears on the timeline panel in the bottom right as a blue dot. The timeline can be adjusted by clicking on any of the metrics boxes, which rebuilds the timeline to show how each model performed based on the selected metric. Clicking any blue dot restores the state of the corresponding model to the canvas.
5. **Model cache (not pictured):** All aspects of the model, including coefficient values, metrics, timeline, and canvas state, are saved in a cache, which automatically reload when the same dataset is re-loaded, even if the program is closed and re-opened.

See the [Interface Quickstart Guide](docs/quickstart.md) for more details.

# Installation

The package is currently hosted on TestPyPi. When installing, use the commands below to install all dependencies from PyPi in addition to the TestPyPi repo. Python 3 must be installed prior to installation.

#### Create and enter new virtual environment (optional)

```
python3 -m venv climate_econometrics_toolkit.venv
source climate_econometrics_toolkit.venv/bin/activate
```

#### Install package and dependencies

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ climate-econometrics-toolkit==0.0.16
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

For further documentation, see the [Interface Quickstart Guide](docs/quickstart.md) and the [API Documentation](docs/api_documentation.pdf).
