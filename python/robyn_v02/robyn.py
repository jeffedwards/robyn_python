# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from collections import defaultdict
from datetime import timedelta
import matplotlib.pyplot as plt
import math
import os
from prophet import Prophet
# import weibull as weibull
from scipy import stats
from scipy.optimize import curve_fit
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


########################################################################################################################
# MAIN


class Robyn(object):

    def __init__(self, country, dateVarName, depVarName, mediaVarName):

        # todo comprehensive documentation on each variable

        self.country = country
        self.dateVarName = dateVarName
        self.depVarName = depVarName
        self.activate_prophet = True
        self.prophet = ["trend", "season", "holiday"]
        self.prophetVarSign = ["default", "default", "default"]
        self.activate_baseline = True
        self.baseVarName = ['competitor_sales_B']
        self.baseVarSign = ['negative']
        self.mediaVarName = mediaVarName
        self.mediaSpendName = ["tv_S", "ooh_S",	"print_S", "facebook_S", "search_S"]
        self.mediaVarSign = ["positive", "positive", "positive", "positive", "positive"]
        self.factorVarName = []
        self.cores = 6
        self.adstock_type = 'geometric'
        self.iter = 500
        self.hyperOptimAlgo = 'DiscreteOnePlusOne'
        self.trial = 40
        self.dayInterval = 7
        self.activate_calibration = False
        self.lift = pd.DataFrame()
        self.hyperBounds = {}
        self.df_holidays = pd.read_csv('source/holidays.csv')

        self.plotNLSCollect = None
        self.modNLSCollect = None
        # self.hatNLSCollect = None
        self.mediaCostFactor = None
        self.costSelector = None
        self.getSpendSum = None
        self.modelRecurrance = None
        self.forecastRecurrance = None


        # self.set_hyperparmeter_bounds()
        # self.check_conditions()

        # todo Variables.
        # todo Global variables should be set in class.
        # todo Function specific variables should set in the function.  Should add default values as much as possible.

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

    def check_conditions(self, dt_transform):
        """
            check all conditions 1 by 1; terminate and raise errors if conditions are not met
            :param dt_transformations:
            :return: dict
            """
        #try:
        #    d['set_mediaVarName']
        #except NameError:
        #    print('set_mediaVarName must be specified')

        if self.activate_prophet and not set(self.prophet).issubset({'trend', 'season', 'weekday', 'holiday'}):
            raise ValueError('set_prophet must be "trend", "season", "weekday" or "holiday"')
        if self.activate_baseline:
            if len(self.baseVarName) != len(self.baseVarSign):
                raise ValueError('set_baseVarName and set_baseVarSign have to be the same length')

        if len(self.mediaVarName) != len(self.mediaVarSign):
            raise ValueError('set_mediaVarName and set_mediaVarSign have to be the same length')
        if not (set(self.prophetVarSign).issubset({"positive", "negative", "default"}) and
                set(self.baseVarSign).issubset({"positive", "negative", "default"}) and
                set(self.mediaVarSign).issubset({"positive", "negative", "default"})):
            raise ValueError('set_prophetVarSign, '
                             'set_baseVarSign & set_mediaVarSign must be "positive", "negative" or "default"')
        if self.activate_calibration:
            if self.lift.shape[0] == 0:
                raise ValueError('please provide lift result or set activate_calibration = FALSE')
            if (min(self.lift['liftStartDate']) < min(dt_transform['ds'])
                    or (max(self.lift['liftEndDate']) > max(dt_transform['ds']) + timedelta(days=self.dayInterval - 1))):
                raise ValueError(
                    'we recommend you to only use lift results conducted within your MMM input data date range')

            if self.iter < 500 or self.trial < 80:
                raise ValueError('you are calibrating MMM. we recommend to run at least 500 iterations '
                                 'per trial and at least 80 trials at the beginning')

        if self.adstock_type not in ['geometric', 'weibull']:
            raise ValueError('adstock must be "geometric" or "weibull"')
        if self.adstock_type == 'geometric':
            num_hp_channel = 3
        else:
            num_hp_channel = 4
        # TODO: check hyperparameter names?
        if set(self.get_hypernames()) != set(list(self.hyperBounds.keys())):
            raise ValueError('set_hyperBoundLocal has incorrect hyperparameters')
        if dt_transform.isna().any(axis=None):
            raise ValueError('input data includes NaN')
        if np.isinf(dt_transform).any():
            raise ValueError('input data includes Inf')

        return None

    def input_wrangling(self, dt):
        """

            :param dt:
            :param dt_holiday:
            :param d:
            :param set_lift:
            :param set_hyperBoundLocal:
            :return: (DataFrame, dict)
            """
        dt_transform = dt.copy().reset_index()
        dt_transform = dt_transform.rename({self.dateVarName: 'ds'}, axis=1)
        dt_transform['ds'] = pd.to_datetime(dt_transform['ds'], format='%Y-%m-%d')
        dt_transform = dt_transform.rename({self.depVarName: 'depVar'}, axis=1)
        self.df_holidays['ds'] = pd.to_datetime(self.df_holidays['ds'], format='%Y-%m-%d')
        # check date format
        try:
            pd.to_datetime(dt_transform['ds'], format='%Y-%m-%d', errors='raise')
        except ValueError:
            print('input date variable should have format "yyyy-mm-dd"')

        # check variable existence
        if not self.activate_prophet:
            self.prophet = None
            self.prophetVarSign = None

        if not self.activate_baseline:
            self.baseVarName = None
            self.baseVarSign = None

        if not self.activate_calibration:
            self.lift = None

        try:
            self.mediaSpendName
        except NameError:
            print('set_mediaSpendName must be specified')

        if len(self.mediaVarName) != len(self.mediaVarSign):
            raise ValueError('set_mediaVarName and set_mediaVarSign have to be the same length')

        #TODO new R version is now different
        trainSize = round(dt_transform.shape[0] * d['set_modTrainSize'])
        dt_train = dt_transform[self.mediaVarName].iloc[:trainSize, :]
        train_all0 = dt_train.loc[:, dt_train.sum(axis=0) == 0]
        if train_all0.shape[1] != 0:
            raise ValueError('These media channels contains only 0 within training period. '
                             'Recommendation: increase set_modTrainSize, remove or combine these channels')

        dayInterval = dt_transform['ds'].nlargest(2)
        dayInterval = (dayInterval.iloc[0] - dayInterval.iloc[1]).days
        if dayInterval == 1:
            intervalType = 'day'
        elif dayInterval == 7:
            intervalType = 'week'
        elif 28 <= dayInterval <= 31:
            intervalType = 'month'
        else:
            raise ValueError('input data has to be daily, weekly or monthly')
        self.dayInterval = dayInterval
        mediaVarCount = len(self.mediaVarName)

        ################################################################
        #### model reach metric from spend
        mediaCostFactor = pd.DataFrame(dt_transform[self.mediaSpendName].sum(axis=0),
                                       columns=['total_spend']).reset_index()
        var_total = pd.DataFrame(dt_transform[self.mediaVarName].sum(axis=0), columns=['total_var']).reset_index()
        mediaCostFactor['mediaCostFactor'] = mediaCostFactor['total_spend'] / var_total['total_var']
        mediaCostFactor = mediaCostFactor.drop(columns=['total_spend'])
        costSelector = pd.Series(self.mediaSpendName) != pd.Series(self.set_mediaVarName)

        if len(costSelector) != 0:
            modNLSCollect = defaultdict()
            yhatCollect = []
            plotNLSCollect = []
            for i in range(mediaVarCount - 1):
                if costSelector[i]:
                    dt_spendModInput = pd.DataFrame(dt_transform.loc[:, self.mediaSpendName[i]])
                    dt_spendModInput['reach'] = dt_transform.loc[:, self.mediaVarName[i]]
                    dt_spendModInput.loc[
                        dt_spendModInput[self.mediaSpendName[i]] == 0, self._mediaSpendName[i]] = 0.01
                    dt_spendModInput.loc[dt_spendModInput['reach'] == 0, 'reach'] = \
                        dt_spendModInput[dt_spendModInput['reach'] == 0][self.mediaSpendName[i]] / \
                        mediaCostFactor['mediaCostFactor'][i]

                    # Michaelis-Menten model
                    # vmax = max(dt_spendModInput['reach'])/2
                    # km = max(dt_spendModInput['reach'])
                    # y = michaelis_menten(dt_spendModInput[d['set_mediaSpendName'][i]], vmax, km)
                    popt, pcov = curve_fit(self.michaelis_menten, dt_spendModInput[self.mediaSpendName[i]],
                                           dt_spendModInput['reach'])

                    yhatNLS = self.michaelis_menten(dt_spendModInput[self.mediaSpendName[i]], *popt)
                    # nls_pred = yhatNLS.predict(np.array(dt_spendModInput[d['set_mediaSpendName'][i]]).reshape(-1, 1))

                    # linear model
                    lm = LinearRegression().fit(np.array(dt_spendModInput[self.mediaSpendName[i]])
                                                .reshape(-1, 1), np.array(dt_spendModInput['reach']).reshape(-1, 1))
                    lm_pred = lm.predict(np.array(dt_spendModInput[self.mediaSpendName[i]]).reshape(-1, 1))

                    # compare NLS & LM, takes LM if NLS fits worse
                    rsq_nls = r2_score(dt_spendModInput['reach'], yhatNLS)
                    rsq_lm = r2_score(dt_spendModInput['reach'], lm_pred)
                    costSelector[i] = rsq_nls > rsq_lm

                    modNLSCollect[self.mediaSpendName[i]] = {'vmax': popt[0], 'km': popt[1], 'rsq_lm': rsq_lm,
                                                             'rsq_nls': rsq_nls, 'coef_lm': lm.coef_}

                    yhat_dt = pd.DataFrame(dt_spendModInput['reach']).rename(columns={'reach': 'y'})
                    yhat_dt['channel'] = self.mediaVarName[i]
                    yhat_dt['x'] = dt_spendModInput[self.mediaSpendName[i]]
                    yhat_dt['yhat'] = yhatNLS if costSelector[i] else lm_pred
                    yhat_dt['models'] = 'nls' if costSelector[i] else 'lm'
                    yhatCollect.append(yhat_dt)

                    # TODO: generate plots

        self.plotNLSCollect = plotNLSCollect
        self.modNLSCollect = modNLSCollect
        # d['yhatNLSCollect'] = yhatNLSCollect

        getSpendSum = pd.DataFrame(dt_transform[self.mediaSpendName].sum(axis=0), columns=['total_spend']).T

        self.mediaCostFactor = mediaCostFactor
        self.costSelector = costSelector
        self.getSpendSum = getSpendSum

        ################################################################
        #### clean & aggregate data
        all_name = [['ds'], ['depVar'], self.prophet, self.baseVarName, self.mediaVarName]
        all_name = set([item for sublist in all_name for item in sublist])
        # all_mod_name = [['ds'], ['depVar'], d['set_prophet'], d['set_baseVarName'], d['set_mediaVarName']]
        # all_mod_name = [item for sublist in all_mod_name for item in sublist]
        if len(all_name) != len(set(all_name)):
            raise ValueError('Input variables must have unique names')

        ## transform all factor variables
        try:
            self.factorVarName
        except:
            pass
        finally:
            if len(self.factorVarName) > 0:
                dt_transform[self.factorVarName].apply(lambda x: x.astype('category'))
            else:
                self.factorVarName = None

        ################################################################
        #### Obtain prophet trend, seasonality and changepoints

        if self.activate_prophet:
            if len(self.prophet) != len(self.prophet):
                raise ValueError('set_prophet and set_prophetVarSign have to be the same length')
            if len(self.prophet) == 0 or len(self.prophetVarSign) == 0:
                raise ValueError('if activate_prophet == TRUE, set_prophet and set_prophetVarSign must to specified')
            if self.country not in self.df_holidays['country'].values:
                raise ValueError(
                    'set_country must be already included in the holidays.csv and as ISO 3166-1 alpha-2 abbreviation')

            recurrence = dt_transform.copy().rename(columns={'depVar': 'y'})
            use_trend = True if 'trend' in self.prophet else False
            use_season = True if 'season' in self.prophet else False
            use_weekday = True if 'weekday' in self.prophet else False
            use_holiday = True if 'holiday' in self.prophet else False

            if intervalType == 'day':
                holidays = self.df_holidays
            elif intervalType == 'week':
                weekStartInput = dt_transform['ds'][0].weekday()
                if weekStartInput == 0:
                    weekStartMonday = True
                elif weekStartInput == 6:
                    weekStartMonday = False
                else:
                    raise ValueError('week start has to be Monday or Sunday')
                self.df_holidays['weekday'] = self.df_holidays['ds'].apply(lambda x: x.weekday())
                self.df_holidays['dsWeekStart'] = self.df_holidays.apply(lambda x: x['ds'] - timedelta(days=x['weekday']), axis=1)
                self.df_holidays['ds'] = self.df_holidays['dsWeekStart']
                self.df_holidays = self.df_holidays.drop(['dsWeekStart', 'weekday'], axis=1)
                holidays = self.df_holidays.groupby(['ds', 'country', 'year'])['holiday'].apply(
                    lambda x: '#'.join(x)).reset_index()

            elif intervalType == 'month':
                monthStartInput = dt_transform['ds'][0].strftime("%d")
                if monthStartInput != '01':
                    raise ValueError("monthly data should have first day of month as datestampe, e.g.'2020-01-01'")
                self.df_holidays['month'] = self.df_holidays['ds'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(normalize=True)
                self.df_holidays['ds'] = self.df_holidays['month']
                self.df_holidays.drop(['month'], axis=1)
                holidays = self.df_holidays.groupby(['ds', 'country', 'year'])['holiday'].apply(
                    lambda x: '#'.join(x)).reset_index()
            h = holidays[holidays['country'] == self.country] if use_holiday else None
            modelRecurrance = Prophet(holidays=h, yearly_seasonality=use_season, weekly_seasonality=use_weekday,
                                      daily_seasonality=False)
            modelRecurrance.fit(recurrence)
            forecastRecurrance = modelRecurrance.predict(dt_transform)

            self.modelRecurrance = modelRecurrance
            self.forecastRecurrance = forecastRecurrance

            # python implementation of scale() is different from R, may need to hard-code the R equivalent
            if use_trend:
                fc_trend = forecastRecurrance['trend'][:recurrence.shape[0]]
                fc_trend = preprocessing.scale(fc_trend)
                dt_transform['trend'] = fc_trend
            if use_season:
                fc_season = forecastRecurrance['yearly'][:recurrence.shape[0]]
                fc_season = preprocessing.scale(fc_season)
                dt_transform['seasonal'] = fc_season
            if use_weekday:
                fc_weekday = forecastRecurrance['weekly'][:recurrence.shape[0]]
                fc_weekday = preprocessing.scale(fc_weekday)
                dt_transform['weekday'] = fc_weekday
            if use_holiday:
                fc_holiday = forecastRecurrance['holidays'][:recurrence.shape[0]]
                fc_holiday = preprocessing.scale(fc_holiday)
                dt_transform['trend'] = fc_holiday

        ################################################################
        #### Finalize input

        self.check_conditions(dt_transform)

        return dt_transform



    def set_param_bounds(self):
        """

        :return:
        """

        pass

    def get_hypernames(self):

        if self.adstock_type == "geometric":
            global_name = ["thetas", "alphas", "gammas"]
            local_name = sorted(list([i + "_" + str(j) for i in self.mediaVarName for j in global_name]))
        elif self.adstock_type == "weibull":
            global_name = ["shapes", "scales", "alphas", "gammas"]
            local_name = sorted(list([i + "_" + str(j) for i in self.mediaVarName for j in global_name]))

        return local_name

    @staticmethod
    def michaelis_menten(spend, vmax, km):
        """
            :param vmax:
            :param spend:
            :param km:
            :return: float
            """

        return vmax * spend / (km + spend)

    @staticmethod
    def adstockGeometric(x, theta):
        """
            :param x:
            :param theta:
            :return: numpy
            """

        x_decayed = [x[0]] + [0] * (len(x) - 1)
        for i in range(1, len(x_decayed)):
            x_decayed[i] = x[i] + theta * x_decayed[i - 1]

        return x_decayed

    @staticmethod
    def helperWeibull(x, y, vec_cum, n):
        """
            :param x:
            :param y:
            :param vec_cum:
            :param n:
            :return: numpy
            """

        x_vec = np.array([0] * (y - 1) + [x] * (n - y + 1))
        vec_lag = np.roll(vec_cum, y - 1)
        vec_lag[: y - 1] = 0
        x_matrix = np.c_[x_vec, vec_lag]
        x_prod = np.multiply.reduce(x_matrix, axis=1)
        print(x_prod)
        return x_prod

    def adstockWeibull(self, x, shape, scale):
        """
            Parameters
            ----------
            x: numpy array
            shape: shape parameter for Weibull
            scale: scale parameter for Weibull
            Returns
            -------
            (list, list)
            """
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html

        n = len(x)
        bin = list(range(1, n + 1))
        scaleTrans = round(np.quantile(bin, scale))
        thetaVec = 1 - stats.weibull_min.cdf(bin[:-1], shape, scale=scaleTrans)
        thetaVec = np.concatenate(([1], thetaVec))
        thetaVecCum = np.cumprod(thetaVec).tolist()

        x_decayed = list(map(lambda i, j: self.helperWeibull(i, j, vec_cum=thetaVecCum, n=n), x, bin))
        x_decayed = np.concatenate(x_decayed, axis=0).reshape(n, n)
        x_decayed = np.transpose(x_decayed)
        x_decayed = np.sum(x_decayed, axis=1).tolist()

        return x_decayed, thetaVecCum

    def transformation(self, x, adstock, theta=None, shape=None, scale=None, alpha=None, gamma=None, stage=3):
        """
            ----------
            Parameters
            ----------
            x: vector
            adstock: chosen adstock (geometric or weibull)
            theta: decay coefficient
            shape: shape parameter for weibull
            scale: scale parameter for weibull
            alpha: hill function parameter
            gamma: hill function parameter
            Returns
            -------
            s-curve transformed vector
            """

        ## step 1: add decay rate
        if adstock == "geometric":
            x_decayed = self.adstockGeometric(x, theta)

            if stage == "thetaVecCum":
                thetaVecCum = theta
            for t in range(1, len(x) - 1):
                thetaVecCum[t] = thetaVecCum[t - 1] * theta
            # thetaVecCum.plot()

        elif adstock == "weibull":
            x_list = self.adstockWeibull(x, shape, scale)
            x_decayed = x_list['x_decayed']
            # x_decayed.plot()

            if stage == "thetaVecCum":
                thetaVecCum = x_list['thetaVecCum']
            # thetaVecCum.plot()

        else:
            print("alternative must be geometric or weibull")

        ## step 2: normalize decayed independent variable # deprecated
        # x_normalized = x_decayed

        ## step 3: s-curve transformation
        gammaTrans = round(np.quantile(np.linspace(min(x_decayed), max(x_decayed), 100), gamma), 4)
        x_scurve = x_decayed ** alpha / (x_decayed ** alpha + gammaTrans ** alpha)
        # x_scurve.plot()
        if stage in [1, 2]:
            x_out = x_decayed
        # elif stage == 2:
        # x_out = x_normalized
        elif stage == 3:
            x_out = x_scurve
        elif stage == "thetaVecCum":
            x_out = thetaVecCum
        else:
            raise ValueError(
                "hyperparameters out of range. theta range: 0-1 (excl.1), shape range: 0-5 (excl.0), alpha range: 0-5 (excl.0),  gamma range: 0-1 (excl.0)")

        return x_out

    @staticmethod
    def unit_format(x_in):
        """
            Define helper unit format function for axis
            :param x_in: a number in decimal or float format
            :return: the number rounded and in certain cases abbreviated in the thousands, millions, or billions
            """

        # suffixes = ["", "Thousand", "Million", "Billion", "Trillion", "Quadrillion"]
        number = str("{:,}".format(x_in))
        n_commas = number.count(',')
        # print(number.split(',')[0], suffixes[n_commas])

        if n_commas >= 3:
            x_out = f'{round(x_in / 1000000000, 1)} bln'
        elif n_commas == 2:
            x_out = f'{round(x_in / 1000000, 1)} mio'
        elif n_commas == 1:
            x_out = f'{round(x_in / 1000, 1)} tsd'
        else:
            x_out = str(int(round(x_in, 0)))

        return x_out

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

    @staticmethod
    def lambdaRidge(x, y, seq_len=100, lambda_min_ratio=0.0001):
        """
            ----------
            Parameters
            ----------
            x: matrix
            y: vector
            Returns
            -------
            lambda sequence
            """

        def mysd(y):
            return math.sqrt(sum((y - sum(y) / len(y)) ** 2) / len(y))

        sx = x.apply(mysd)
        sx = preprocessing.scale(x).T
        sy = y.to_numpy()
        sxy = sx * sy
        sxy = sxy.T
        # return sxy
        lambda_max = max(abs(sxy.sum(axis=0)) / (
                0.001 * x.shape[0]))  # 0.001 is the default smalles alpha value of glmnet for ridge (alpha = 0)
        lambda_max_log = math.log(lambda_max)

        log_step = (math.log(lambda_max) - math.log(lambda_max * lambda_min_ratio)) / (seq_len - 1)
        log_seq = np.linspace(math.log(lambda_max), math.log(lambda_max * lambda_min_ratio), seq_len)
        lambda_seq = np.exp(log_seq)
        return lambda_seq

    def decomp(self, coefs, dt_mod, dt_modAdstocked, x, y_pred, i):
        """
            ----------
            Parameters
            ----------
            coef: Pandas Series with index name
            dt_modAdstocked: Pandas Dataframe
            x: Pandas Dataframe
            y_pred: Pandas Series
            i: interger
            d: master dictionary
            Returns
            -------
            decompCollect
            """

        ## input for decomp
        y = dt_modAdstocked["depVar"]
        indepVar = dt_modAdstocked.loc[:, dt_modAdstocked.columns != 'depVar']
        intercept = coefs.iloc[0]
        indepVarName = indepVar.columns.tolist()
        indepVarCat = indepVar.select_dtypes(['category']).columns.tolist()

        ## decomp x
        xDecomp = x * coefs.iloc[1:]
        xDecomp.insert(loc=0, column='intercept', value=[intercept] * len(x))
        xDecompOut = pd.concat([dt_mod['ds'], xDecomp], axis=1)

        ## QA decomp
        y_hat = xDecomp.sum(axis=1)
        errorTerm = y_hat - y_pred
        if np.prod(round(y_pred) == round(y_hat)) == 0:
            print(
                "### attention for loop " + str(i) + \
                ": manual decomp is not matching linear model prediction. Deviation is " + \
                str(np.mean(errorTerm / y) * 100) + "% ###"
            )

        ## output decomp
        y_hat_scaled = abs(xDecomp).sum(axis=1)
        xDecompOutPerc_scaled = abs(xDecomp).div(y_hat_scaled, axis=0)
        xDecompOut_scaled = xDecompOutPerc_scaled.multiply(y_hat, axis=0)

        xDecompOutAgg = xDecompOut[['intercept'] + indepVarName].sum(axis=0)
        xDecompOutAggPerc = xDecompOutAgg / sum(y_hat)
        xDecompOutAggMeanNon0 = xDecompOut.mean(axis=0).clip(lower=0)
        xDecompOutAggMeanNon0Perc = xDecompOutAggMeanNon0 / sum(xDecompOutAggMeanNon0)

        coefsOut = coefs.reset_index(inplace=False)
        coefsOut = coefsOut.rename(columns={'index': 'rn'})
        if len(indepVarCat) == 0:
            pass
        else:
            for var in indepVarCat:
                coefsOut.rn.replace(r'(^.*' + var + '.*$)', var, regex=True, inplace=True)
        coefsOut = coefsOut.groupby(coefsOut['rn'], sort=False).mean().reset_index()

        frame = {'xDecompAgg': xDecompOutAgg,
                 'xDecompPerc': xDecompOutAggPerc,
                 'xDecompMeanNon0': xDecompOutAggMeanNon0,
                 'xDecompMeanNon0Perc': xDecompOutAggMeanNon0Perc}
        frame.index = coefsOut.index
        decompOutAgg = pd.merge(coefsOut, frame, left_index=True, right_index=True)
        decompOutAgg['pos'] = decompOutAgg['xDecompAgg'] >= 0

        decompCollect = {'xDecompVec': xDecompOut,
                         'xDecompVec_scaled': xDecompOut_scaled,
                         'xDecompAgg': decompOutAgg}

        return decompCollect

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

    def mmm(self,
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
        self.iterations = set_iter
        x = None
        if x == 'fake':
            iter = self.iterations

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

    def fit(self,
            df,
            optimizer_name=set_hyperOptimAlgo,
            set_trial=100,
            set_cores=12,
            fixed_out=False,
            fixed_hyppar_dt=None
            ):

        ## todo need to figure out what fixed is
        plot_folder = getwd()
        pareto_fronts = np.array[1, 2, 3]

        ### start system time

        # t0 <- Sys.time()

        ### check if plotting directory exists

        # if (!dir.exists(plot_folder)) {
        # plot_folder < - getwd()
        # message("provided plot_folder doesn't exist. Using default plot_folder = getwd(): ", getwd())
        # }

        ### run mmm function on set_trials

        hyperparameter_fixed = pd.DataFrame.from_dict(set_hyperBoundLocal)
        hypParamSamName = self.get_hypernames()

        if fixed_out:

            ### run mmm function if using old model result tables

            if fixed.hyppar.dt.isna().any(axis=None):
                raise ValueError(
                    'when fixed_out=T, please provide the table model_output_resultHypParam from previous runs or pareto_hyperparameters.csv with desired model IDs')

            ### check if hypParamSamName + 'lambda' is in fixed.hyppar.dt columns

            # if (!all(c(hypParamSamName, "lambda") %in% names(fixed.hyppar.dt))) {stop("fixed.hyppar.dt is provided with wrong input. please provide the table model_output_collect$resultHypParam from previous runs or pareto_hyperparameters.csv with desired model ID")}
            # if any('lambdas' in s for s in hypParamSamName):
            #    raise ValueError('fixed.hyppar.dt is provided with wrong input. please provide the table model_output_collect$resultHypParam from previous runs or pareto_hyperparameters.csv with desired model ID')

            model_output_collect = []

            ### call mmm function with inputs

            model_output_collect[[1]] = self.mmm(fixed.hyppar.dt[, hypParamSamName, with = F],
            set_iter = set_iter
                                           ,set_cores = set_cores
                                           ,optimizer_name = optimizer_name
                                           ,fixed.out = T
                                           ,fixed.lambda = unlist(fixed.hyppar.dt$lambda))


def budget_allocator(self, model_id):  # This is the last step_model allocation
        pass
