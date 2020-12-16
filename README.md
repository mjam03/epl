# epl
Python code and Jupyter Notebooks for a pet project to predict football match results using various statistical models - both classical and ML

## Overview
Code consists of several main branches:
 - Database Creation/Maintenance: Code to create and regularly maintain an up to date sqlite database of European Football Matches pulled from [here](https://www.football-data.co.uk/)
 - Feature Creation/Maintenance: Code to take this raw match result data and create relevant features for use in predictive modelling
 - Modelling/Evaluation Helpers: Helper functions wrapped around standard libs (statsmodels, sklearn) to improve notebook workflow (e.g. pretty print confusion matrix using seaborn)

## Get Started
To run the code on your local machine from scratch:
1. Open your cmd line and go to where you want to install e.g. to install like below go to 'dev' as parent directory:
```
../dev/
    project1/
    project2/
    epl/
```
2. Once in your chosen parent directory, run the following to clone this git repo locally (requires git to be installed - comes as default on mac):
```
git clone https://github.com/mjam03/epl.git
```
3. Run the setup.py script to:
 - Add ../parent_directory/epl to your python path (so e.g. from epl.query import blahblahblah will work)
 - Create an 'epl' conda environment (utilises environment.yaml in this git) so you have all the required libraries (statsmodels, numpy etc) to run this code
4. Run the database creation/management script to create your own local sqlite database - the script is in epl/scripts:
```
cd scripts
python parse.py
```
