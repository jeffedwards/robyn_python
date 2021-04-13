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

# Import functions
from python import fb_robyn_func as f

# Uploading an Input file.
# Specify the path where the input data file is located
#path = "C:\\Users\\anuragjoshi\Documents\\01 Projects\\04 MMM\Robyn-master2.0\\source\\"
#path = "C:\\pytasks\\202103_Robyn_translation\\robyn_python\\source\\" #delete later. Tmp line for debugging (David)
path = "/Users/nuochen/Documents/Robyn/robyn_python/source/" #(Nuo debugging)

df = pd.read_csv(path + "de_simulated_data.csv").set_index('DATE')
dt_holidays = pd.read_csv(path + "holidays.csv")

# Create dictionary to initiate parameters
d = {
    # set model input variables
    "df_Input": df,
    "set_country": 'DE',  # only one country allowed once. Including national holidays for 59 countries, whose list can be found on our github guide
    "set_dateVarName": 'DATE',  # date format must be "2020-01-01"
    "set_depVarName": 'revenue',  # there should be only one dependent variable
    "activate_prophet": True,  # Turn on or off the Prophet feature
    "set_prophet": ["trend", "season", "holiday"],  # "trend","season", "weekday", "holiday" are provided and case-sensitive. Recommend at least keeping Trend & Holidays
    "set_prophet_sign": ["default", "default", "default"],  # c("default", "positive", and "negative"). Recommend as default. Must be same length as set_prophet
    "activate_baseline": True,
    "set_baseVarName": ['competitor_sales_B'],  # typically competitors, price & promotion, temperature, unemployment rate etc
    "set_baseVarSign": ['negative'],  # c("default", "positive", and "negative"), control the signs of coefficients for baseline variables
    "set_mediaVarName": ["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"],  # we recommend to use media exposure metrics like impressions, GRP etc for the model. If not applicable, use spend instead
    "set_mediaSpendName": ["tv_S", "ooh_S",	"print_S", "facebook_S", "search_S"],
    "set_mediaVarSign": ["positive", "positive", "positive", "positive", "positive"],
    "set_factorVarName": [], # please specify which variable above should be factor, otherwise leave empty

    # set global model parameters
    "set_cores": 6,  # User needs to set these cores depending upon the cores in local machine
    "adstock": "geometric",  # geometric or weibull. weibull is more flexible, yet has one more parameter and thus takes longer
    "set_iter": 500,  # number of allowed iterations per trial. 500 is recommended
    "set_modTrainSize": 0.74,  # 0.74 means taking 74% of data to train and 30% to test the model.
    "set_hyperOptimAlgo": "DiscreteOnePlusOne",  # selected algorithm for Nevergrad, the gradient-free optimisation library https://facebookresearch.github.io/nevergrad/index.html
    "set_trial": 40, # number of allowed iterations per trial. 40 is recommended without calibration, 100 with calibration.
    ## Time estimation: with geometric adstock, 500 iterations * 40 trials and 6 cores, it takes less than 1 hour. Weibull takes at least twice as much time.

    # define ground truth (e.g. Geo test, FB Lift test, MTA etc.)
    "activate_calibration": False,
}

# lift calibration table
set_lift = pd.DataFrame({'channel': ["facebook_I",  "tv_S", "facebook_I"],
                         'liftStartDate': ["2018-05-01", "2017-11-27", "2018-07-01"],
                         'liftEndDate': ["2018-06-10", "2017-12-03", "2018-07-20"],
                         'liftAbs': [400000, 300000, 200000]})
set_lift['liftStartDate'] = pd.to_datetime(set_lift['liftStartDate'], format='%Y-%m-%d')
set_lift['liftEndDate'] = pd.to_datetime(set_lift['liftStartDate'], format='%Y-%m-%d')

# Calculate and set core for running Robyn:
print("Total Cores Running on the machine:", (multiprocessing.cpu_count()))

f.inputWrangling(df, dt_holidays, d, set_lift)

f.plotTrainSize(True, d)





