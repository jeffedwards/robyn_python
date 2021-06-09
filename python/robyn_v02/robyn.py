# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri


########################################################################################################################
# MAIN
from python.fb_robyn_func import rsq


class Robyn(object):

    def __init__(self):
        # include all the dictionary items meant for global parameters, this requires good documentation
        self.check_conditions()  # maybe?

        # todo Variables.
        # todo Global variables should be set in class.
        # todo Function specific variables should set in the function.  Should add default values as much as possible.

        # VARIABLES - GLOBAL
        self.df_holidays = pd.read_csv('source/holidays.csv')
        self.adstock_type = 'geometric'  # geometric or weibull. weibull is more flexible, et has one more parameter and thus takes longer
        self.hyperBoundLocal = None
        self.names_media_spend = ['tv_S', 'ooh_S', 'print_S', 'facebook_S', 'search_S']

        # todo Does it make sense to just get these from the column names in a data frame?

        # VARIABLES - WILL BE UPDATED
        self.mod = None

        # VARIABLES - FOR TESTING
        # TODO remove this section for production
        self.test_activate_baseline = True
        self.test_activate_calibration = True  # Switch to True to calibrate model.
        self.test_activate_prophet = True  # Turn on or off the Prophet feature
        self.test_baseVarName = ['competitor_sales_B']  # typically competitors, price  promotion, temperature, unemployment rate etc
        self.test_baseVarSign = ['negative']  # c(default, positive, and negative), control the signs of coefficients for baseline variables
        self.test_cores = 6  # User needs to set these cores depending upon the cores n local machine
        self.test_country = 'DE'  # only one country allowed once. Including national holidays for 59 countries, whose list can be found on our github guide
        self.test_dateVarName = 'DATE'  # date format must be 2020-01-01
        self.test_depVarType = 'revenue'  # there should be only one dependent variable
        self.test_factorVarName = []
        self.test_fixed_lambda = None
        self.test_fixed_out = False
        self.test_hyperOptimAlgo = 'DiscreteOnePlusOne'  # selected algorithm for Nevergrad, the gradient-free optimisation library ttps =//facebookresearch.github.io/nevergrad/self.test_index.html
        self.test_iter = 500  # number of allowed iterations per trial. 500 is recommended
        self.test_lambda_ = 1
        self.test_lambda_n = 100
        self.test_lower_limits = [0, 0, 0, 0]
        self.test_mediaVarName = ['tv_S', 'ooh_S', 'print_S', 'facebook_I', 'search_clicks_P']  # we recommend to use media exposure metrics like impressions, GRP etc for the model. If not applicable, use spend instead
        self.test_mediaVarSign = ['positive', 'positive', 'positive', 'positive', 'positive']
        self.test_modTrainSize = 0.74  # 0.74 means taking 74% of data to train and 0% to test the model.
        self.test_optimizer_name = 'DiscreteOnePlusOne'
        self.test_plot_folder = '~/Documents/GitHub/plots'
        self.test_prophet = ['trend', 'season', 'holiday']  # 'trend','season', 'weekday', 'holiday' are provided and case-sensitive. Recommend at least keeping Trend & Holidays
        self.test_prophetVarSign = ['default', 'default', 'default']  # c('default', 'positive', and 'negative'). Recommend as default. Must be same length as set_prophet
        self.test_trial = 80  # number of all
        self.test_upper_limits = [10, 10, 10, 10]
        self.test_x_train = np.array([[1, 1, 2], [3, 4, 2], [6, 5, 2], [5, 5, 3]])
        self.test_y_train = np.array([1, 0, 0, 1])

    def check_conditions(self):
        pass

    def input_wrangling(self, df): # where should we put this??
        # check_conditions()
        pass

    @staticmethod
    def get_hypernames(names_media_variables: str, adstock_type='geometric'):
        """
        :param names_media_variables: list of media channels
        :param adstock_type: chosen adstock (geometric or weibull)
        :return: crossed list of adstock parameters and media channels
        """

        if adstock_type == "geometric":
            global_name = ["thetas", "alphas", "gammas"]
            local_name = sorted(list([i + "_" + str(j) for i in names_media_variables for j in global_name]))
        elif adstock_type == "weibull":
            global_name = ["shapes", "scales", "alphas", "gammas"]
            local_name = sorted(list([i + "_" + str(j) for i in names_media_variables for j in global_name]))

        return local_name

    @staticmethod
    def michaelis_menten(spend, vmax, km):
        pass

    @staticmethod
    def adstockGeometric(x, theta):
        pass

    @staticmethod
    def helperWeibull(x, y, vec_cum, n):
        pass

    @staticmethod
    def adstockWeibull(x, shape, scale):
        pass

    @staticmethod
    def transformation(x, adstock, theta=None, shape=None, scale=None, alpha=None, gamma=None, stage=3):
        pass

    @staticmethod
    def unit_format(x_in):
        pass

    @staticmethod
    def rsq(val_actual, val_predicted):
        # Changed "true" to val_actual because Python could misinterpret True
        """
        :param val_actual: actual value
        :param val_predicted: predicted value
        :return: r-squared
        """
        sse = sum((val_predicted - val_actual) ** 2)
        sst = sum((val_actual - sum(val_actual) / len(val_actual)) ** 2)
        return 1 - sse / sst

    def lambdaRidge(self, x, y, seq_len=100, lambda_min_ratio=0.0001):
        pass

    def decomp(self, coefs, dt_modAdstocked, x, y_pred, i, d):
        pass

    def refit(self, x_train, y_train, lambda_: int, lower_limits: list, upper_limits: list):

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
        mod = r_glmnet(x=x_train,
                       y=y_train,
                       alpha=1,
                       family="gaussian",
                       lambda_=lambda_,
                       lower_limits=lower_limits,
                       upper_limits=upper_limits,
                       intercept=True
                       )

        # Create model without the intercept if negative
        if mod[0][0] < 0:
            mod = r_glmnet(x=x_train,
                           y=y_train,
                           alpha=1,
                           family="gaussian",
                           lambda_=lambda_,
                           lower_limits=lower_limits,
                           upper_limits=upper_limits,
                           intercept=False
                           )

        # Run model
        ro.r('''
                    r_predict <- function(model, s, newx) {
                        predict(model, s=s, newx=newx)
                    }
                ''')
        r_predict = ro.globalenv['r_predict']
        y_train_pred = r_predict(model=mod, s=1, newx=x_train)
        y_train_pred = y_train_pred.reshape(len(y_train_pred), )  # reshape to be of format (n,)

        # Calc r-squared on training set
        rsq_train = self.rsq(val_actual=y_train, val_predicted=y_train_pred)

        # Get coefficients
        coefs = mod[0]

        # Get normalized root mean square error
        nrmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2) / (max(y_train) - min(y_train)))

        # Update model outputs to include calculated values
        mod_out = {'rsq_train': rsq_train,
                   'nrmse_train': nrmse_train,
                   'coefs': coefs,
                   'y_pred': y_train_pred,
                   'mod': mod}

        self.mod = mod_out

    def fit(self,
            df,
            adstock_type='geometric',
            optimizer_name='DiscreteOnePlusOne',
            set_iter=100,
            set_cores=6,
            lambda_n=100,
            fixed_out=False,
            fixed_lambda=None):  # This replaces the original mmm + Robyn functions

        ################################################
        # Collect hyperparameters

        # Expand media spend names to to have the hyperparameter names needed based on adstock type
        names_hyper_parameter_sample_names = \
            self.get_hypernames(names_media_variables=self.names_media_spend, adstock_type=adstock_type)
        # names_hyper_parameter_sample_names = \
        #     get_hypernames(names_media_variables=names_media_spend, adstock_type=adstock_type)

        if not fixed_out:
            # input_collect = # todo not sure what this is.  Finish it.
            # todo collects results for parameters?
            input_collect = self.hyperBoundLocal
            # input_collect = None

        ################################################
        # Get spend share

        ################################################
        # Setup environment

        # Get environment for parallel backend

        # available optimizers in ng
        # optimizer_name <- "DoubleFastGADiscreteOnePlusOne"
        # optimizer_name <- "OnePlusOne"
        # optimizer_name <- "DE"
        # optimizer_name <- "RandomSearch"
        # optimizer_name <- "TwoPointsDE"
        # optimizer_name <- "Powell"
        # optimizer_name <- "MetaModel"  CRASH !!!!
        # optimizer_name <- "SQP"
        # optimizer_name <- "Cobyla"
        # optimizer_name <- "NaiveTBPSA"
        # optimizer_name <- "DiscreteOnePlusOne"
        # optimizer_name <- "cGA"
        # optimizer_name <- "ScrHammersleySearch"

        ################################################
        # Start Nevergrad loop

        # Set iterations

        # Start Nevergrad optimiser

        # Start loop

        # Get hyperparameter sample with ask

        # Scale sample to given bounds

        # Add fixed hyperparameters

        # Parallel start

        #####################################
        # Get hyperparameter sample

        # Tranform media with hyperparameters

        #####################################
        # Split and prepare data for modelling

        # Contrast matrix because glmnet does not treat categorical variables

        # Define sign control

        #####################################
        # Dit ridge regression with x-validation

        #####################################
        # Refit ridge regression with selected lambda from x-validation

        # If no lift calibration, refit using best lambda

        #####################################
        # Get calibration mape

        #####################################
        # Calculate multi-objectives for pareto optimality

        # Decomp objective: sum of squared distance between decomp share and spend share to be minimised

        # Adstock objective: sum of squared infinite sum of decay to be minimised? maybe not necessary

        # Calibration objective: not calibration: mse, decomp.rssd, if calibration: mse, decom.rssd, mape_lift

        #####################################
        # Collect output

        # End dopar
        # End parallel

        #####################################
        # Nevergrad tells objectives

        # End NG loop
        # End system.time

        #####################################
        # Get nevergrad pareto results

        #####################################
        # Final result collect

    def budget_allocator(self, model_id):  # This is the last step_model allocation
        pass
