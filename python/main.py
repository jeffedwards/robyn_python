#from python.helpers import functions_statistics as fs, functions_data as fd

#fd.func_delete_later()

print("Installing All Dependencies")
import pandas as pd
import scipy as sc
import numpy as np

#from fbprophet import Prophet

import multiprocessing
from scipy.spatial.distance import mahalanobis
import scipy as sp

# DAVID comment: Structure for importing features needs to be determined
from python import fb_robyn_func as f
import python.setting as input

# Uploading an Input file.
# Specify the path where the input data file is located
path = "C:\\pytasks\\202103_Robyn_translation\\robyn_python\\source\\"

df = pd.read_csv(path + "de_simulated_data.csv")
input.df_Input = df.set_index('DATE')

'depVar' not in input.df_Input.columns

################################################################
#### set model input variables


# Loading Prophet in Python:
set_country = 'DE'
input.set_depVarName = ['revenue']

# Turn on or off the Prophet feature
activate_prophet = True

# "trend","season", "weekday", "holiday" are provided and case-sensitive. Recommended to at least keep Trend & Holidays
set_prophet  = ["trend", "season", "holiday"] #
set_prophet_sign = ["default","default",'default']

input.activate_baseline = True
#activate_baseline = True

# typically competitors, price & promotion, temperature,  unemployment rate etc
input.set_baseVarName = ["competitor_sales_B"]

# c("default", "positive", and "negative"), control the signs of coefficients for baseline variables
set_baseVarSign = ["negative"]

# Setting Up Media Variables
input.set_mediaVarName = ['tv_S','ooh_S','print_S','facebook_I','search_clicks_P']
input.set_mediaSpendName = ['tv_S','ooh_S','print_S','facebook_S','search_S']
set_mediaVarSign = ["positive","positive","positive","positive","positive"]

input.set_factorVarName = []



################################################################
#### set global model parameters


# Calculate and set core for running Robyn:
print("Total Cores Running on the machine:", (multiprocessing.cpu_count()))

# User needs to set these cores depending upon the cores in local machine
set_cores = 6

## set model core features
adstock = "geometric" # geometric or weibull . weibull is more flexible, yet has one more parameter and thus takes longer
set_iter = 100 # We recommend to run at least 50k iteration at the beginning, when hyperparameter bounds are not optimised



f.plotTrainSize(True)
set_modTrainSize = 0.74

# selected algorithm for Nevergrad, the gradient-free optimisation library https://facebookresearch.github.io/nevergrad/index.html
set_hyperOptimAlgo = "DiscreteOnePlusOne"


# number of allowed iterations per trial. 40 is recommended without calibration, 100 with calibration.
## Time estimation: with geometric adstock, 500 iterations * 40 trials and 6 cores
set_trial = 3
