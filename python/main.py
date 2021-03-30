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

# Import functions and global parameters
from python import fb_robyn_func as f
import python.setting as input



# Uploading an Input file.
# Specify the path where the input data file is located
path = "C:\\Users\\anuragjoshi\Documents\\01 Projects\\04 MMM\Robyn-master2.0\\source\\"
# path = "C:\\pytasks\\202103_Robyn_translation\\robyn_python\\source\\" #delete later. Tmp line for debugging (David)

df = pd.read_csv(path + "de_simulated_data.csv")
df_Input = df.set_index('DATE')


################################################################
#### set model input variables

# Loading Prophet in Python:
input.set_country = 'DE'
input.set_depVarName = ['revenue']

# Turn on or off the Prophet feature
input.activate_prophet = True

# "trend","season", "weekday", "holiday" are provided and case-sensitive. Recommended to at least keep Trend & Holidays
input.set_prophet  = ["trend", "season", "holiday"]
input.set_prophet_sign = ["default","default",'default']

input.activate_baseline = True

# typically competitors, price & promotion, temperature,  unemployment rate etc
input.set_baseVarName = ["competitor_sales_B"]

# c("default", "positive", and "negative"), control the signs of coefficients for baseline variables
input.set_baseVarSign = ["negative"]

# Setting Up Media Variables
input.set_mediaVarName = ['tv_S','ooh_S','print_S','facebook_I','search_clicks_P']
input.set_mediaSpendName = ['tv_S','ooh_S','print_S','facebook_S','search_S']
input.set_mediaVarSign = ["positive","positive","positive","positive","positive"]

input.set_factorVarName = []


################################################################
#### set global model parameters


# Calculate and set core for running Robyn:
print("Total Cores Running on the machine:", (multiprocessing.cpu_count()))

# User needs to set these cores depending upon the cores in local machine
input.set_cores = 6

## set model core features
input.adstock = "geometric" # geometric or weibull . weibull is more flexible, yet has one more parameter and thus takes longer
input.set_iter = 100 # We recommend to run at least 50k iteration at the beginning, when hyperparameter bounds are not optimised

f.plotTrainSize(True)
input.set_modTrainSize = 0.74

# selected algorithm for Nevergrad, the gradient-free optimisation library https://facebookresearch.github.io/nevergrad/index.html
input.set_hyperOptimAlgo = "DiscreteOnePlusOne"


# number of allowed iterations per trial. 40 is recommended without calibration, 100 with calibration.
## Time estimation: with geometric adstock, 500 iterations * 40 trials and 6 cores
input.set_trial = 3

