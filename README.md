# epl
Python code and Jupyter Notebooks for a pet project to predict football match results using various statistical models - both classical and ML

## Overview
Code consists of several main branches:
 - Database Creation/Maintenance: Code to create and regularly maintain an up to date sqlite database of European Football Matches pulled from [here](https://www.football-data.co.uk/)
 - Feature Creation/Maintenance: Code to take this raw match result data and create relevant features for use in predictive modelling
 - Modelling/Evaluation Helpers: Helper functions wrapped around standard libs (statsmodels, sklearn) to improve notebook workflow (e.g. pretty print confusion matrix using seaborn)

## Get Started
__Dependencies__
1. [git](https://git-scm.com/): Version Control System - used here to clone the git repo locally. Pre-installed on most linux/mac, install instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) if required
2. [conda](https://docs.conda.io/en/latest/miniconda.html): Package & environment manager - used to create python environment (populated with required python libs) to run the code. Helpful installation instructions for __miniconda__ [here](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html)

__Instructions__

To run the code on your local machine from scratch:
1. Once in your chosen parent directory, run the following to clone this git repo locally (requires git to be installed - comes as default on mac):
```
git clone https://github.com/mjam03/epl.git
```
2. Create a conda environment to run the code inside - this is initialised by the included `environment.yml` file:
```
conda env create -f environment.yml
```
3. Activate the newly created environment - this is required _before_ the next step so `epl` is initialised within the conda env:
```
conda activate epl
```
4. Inside directory (i.e. in `./epl/`), execute the setup.py script s.t. the python functions within `epl/` can be picked up - explained brilliantly [here](https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/):
```
pip install -e .
```
pip does this for us and will create a `.egg-info/` directory (used internally to identify the files needed), the -e arg ensures local changes to the code are picked up (do not need to pip install again to pick up changes)

5. Create the sqlite database by running the `parse.py` script inside `epl/scripts/`:
```
cd ./epl/scripts/
python parse.py
```
