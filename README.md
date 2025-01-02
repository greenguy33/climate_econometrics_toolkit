# Climate Econometrics Toolkit

This project has been developed as part of my PhD research. It is a work in progress. Please reach out with any questions or make an issue.

Developer contact: Hayden Freedman (hfreedma@uci.edu)

There is an installation and quick start guide in this README. For more detailed instructions, you can reference the full [User Guide](USER_GUIDE.md).

# Is this the right tool for you?

The Climate Econometrics Toolkit is designed to help researchers interested in using econometric regression models to calculate the downstream societal impacts of climate change.
While the tool is not limited to this use case, and theoretically could be used to analyze data from other domains, certain features have been added with the assumption of the climate data use case.

If this tool will be helpful to you, you most likely...

- Are a climate impacts researcher or statistician interested in econometric-style analyses
- Have some basic knowledge of econometric regression and working with panel data
- Are interested in putting together a workflow for modeling climate impacts without starting completely from scratch

If you are aren't sure what type of analysis this project is designed to help with, I recommend checking out the paper [Anthropogenic climate change has slowed global agricultural productivity growth](https://www.nature.com/articles/s41558-021-01000-1)
by Ortiz-Bobea et al. as an example. I based some of the implementations in this tool off of the codebase attached to this paper and implemented a reproduction of this paper using the tool.

# Overview

After analyzing the workflows of several climate econometric research papers, I identified a three-step workflow which many of these papers have in common.



# Installation

The package is currently hosted on TestPyPi. When installing, use the commands below to install all dependencies from PyPi in addition to the TestPyPi repo.

#### Create and enter new virtual environment (optional)

python3 -m venv climate_econometrics_toolkit.venv

source climate_econometrics_toolkit.venv/bin/activate

#### Install package and dependencies

pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ climate-econometrics-toolkit==0.0.13

# Quick Start

This package contains 
