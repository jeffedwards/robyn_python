# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri

# LOCAL
from python.v02 import fb_robyn_helpers as h

# FOR TESTING
import importlib as il
il.reload(h)


########################################################################################################################
# MAIN

class Robyn(object):
    """
    Robyn MMM
    """

    def __init__(self):
        self.x_train = np.array([[1, 1, 2], [3, 4, 2], [6, 5, 2], [5, 5, 3]])
        self.y_train = np.array([1, 0, 0, 1])
        self.upper_limits = [10, 10, 10, 10]
        self.lower_limits = [0, 0, 0, 0]
        self.lambda_ = 1

        # Create variables that will updated when running
        # self.mod = None
        # self.hyperBoundLocal = None

        # INPUT VARIABLES
        self.activate_calibration = True  # Switch to True to calibrate model.
        self.activate_baseline = True
        self.activate_prophet = True  # Turn on or off the Prophet feature
        self.baseVarName = ['competitor_sales_B']  # typically competitors, price  promotion, temperature, unemployment
        # rate etc
        self.baseVarSign = ['negative']  # c(default, positive, and negative), control the signs of coefficients for
        # baseline variables
        self.country = 'DE'  # only one country allowed once. Including national holidays for 59 countries, whose list
        # can be found on our github guide
        self.dateVarName = 'DATE'  # date format must be 2020-01-01
        self.depVarType = 'revenue'  # there should be only one dependent variable
        self.factorVarName = []
        self.mediaSpendName = ['tv_S', 'ooh_S', 'print_S', 'facebook_S', 'search_S']
        self.mediaVarName = ['tv_S', 'ooh_S', 'print_S', 'facebook_I', 'search_clicks_P']  # we recommend to use media
        # exposure metrics like impressions, GRP etc for the model. If not applicable, use spend instead
        self.mediaVarSign = ['positive', 'positive', 'positive', 'positive', 'positive']
        self.prophet = ['trend', 'season', 'holiday']  # 'trend','season', 'weekday', 'holiday' are provided and case-
        # sensitive. Recommend at least keeping Trend & Holidays
        self.prophetVarSign = ['default', 'default', 'default']  # c('default', 'positive', and 'negative'). Recommend
        # as default. Must be same length as set_prophet

        # GLOBAL PARAMETERS
        self.adstock = 'geometric'  # geometric or weibull. weibull is more flexible, et has one more parameter and
        # thus takes longer
        self.fixed_lambda = None
        self.fixed_out = False
        self.lambda_n = 100
        self.optimizer_name = 'DiscreteOnePlusOne'
        self.plot_folder = '~ / Documents / GitHub / plots'
        self.cores = 6  # User needs to set these cores depending upon the cores n local machine
        self.hyperBoundLocal = None
        self.hyperOptimAlgo = 'DiscreteOnePlusOne'  # selected algorithm for Nevergrad, the gradient-free optimisation
        # library ttps =//facebookresearch.github.io/nevergrad/self.index.html
        self.iter = 500  # number of allowed iterations per trial. 500 is recommended
        self.modTrainSize = 0.74  # 0.74 means taking 74% of data to train and 0% to test the model.
        self.trial = 80  # number of all

    def set_variables(self):
        country = self.country
        self.country = input('What country should we use for holidays? Enter a two digit code')
        print(f'Country was set to: {country}')
        print(f'Country now set to: {self.country}')

    def refit(self):

        # Call R functions - to match outputs of Robyn in R
        numpy2ri.activate()

        # Define glmnet model in r
        ro.r('''
                r_glmnet <- function (x, y, family, alpha, lambda_, lower_limits, upper_limits, intercept) {

                    library(glmnet)

                    if(intercept == 1){
                    # print("Intercept")
                    mod <- glmnet(
                        x=x,
                        y=y,
                        family=family,
                        alpha=alpha,
                        lambda=lambda_,
                        lower.limits=lower_limits,
                        upper.limits=upper_limits,
                        )
                    } else {
                    # print("No Intercept")
                    mod <- glmnet(
                        x=x,
                        y=y,
                        family=family,
                        alpha=alpha,
                        lambda=lambda_,
                        lower.limits=lower_limits,
                        upper.limits=upper_limits,
                        intercept=FALSE
                        )
                    }  
                }
            ''')
        r_glmnet = ro.globalenv['r_glmnet']

        # Create model
        mod = r_glmnet(x=self.x_train,
                       y=self.y_train,
                       alpha=1,
                       family="gaussian",
                       lambda_=self.lambda_,
                       lower_limits=self.lower_limits,
                       upper_limits=self.upper_limits,
                       intercept=True
                       )

        # Create model without the intercept if negative
        if mod[0][0] < 0:
            mod = r_glmnet(x=self.x_train,
                           y=self.y_train,
                           alpha=1,
                           family="gaussian",
                           lambda_=self.lambda_,
                           lower_limits=self.lower_limits,
                           upper_limits=self.upper_limits,
                           intercept=False
                           )

        # Run model
        ro.r('''
                    r_predict <- function(model, s, newx) {
                        predict(model, s=s, newx=newx)
                    }
                ''')
        r_predict = ro.globalenv['r_predict']
        y_train_pred = r_predict(model=mod, s=1, newx=self.x_train)
        y_train_pred = y_train_pred.reshape(len(y_train_pred), )  # reshape to be of format (n,)

        # Calc r-squared on training set
        rsq_train = h.rsq(true=self.y_train, predicted=y_train_pred)

        # Get coefficients
        coefs = mod[0]

        # Get normalized root mean square error
        nrmse_train = np.sqrt(np.mean((self.y_train - y_train_pred) ** 2) / (max(self.y_train) - min(self.y_train)))

        # Update model outputs to include calculated values
        mod_out = {'rsq_train': rsq_train,
                   'nrmse_train': nrmse_train,
                   'coefs': coefs,
                   'y_pred': y_train_pred,
                   'mod': mod}

        self.mod = mod_out

        return mod_out
