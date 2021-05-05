# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# Data transformation and helper functions
########################################################################################################################


########################################################################################################################
# IMPORTS

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
# import weibull as weibull
from sklearn import preprocessing
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
# from prophet import Prophet
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from prophet import Prophet
from datetime import datetime, timedelta
from collections import defaultdict



########################################################################################################################
# FUNCTIONS


def initiate_dictionary():
    """
    Creates a dictionary with variables that are set or updated withing functions
    :return: the dictionary
    """
    dict_vars = {
        'variable_01': dict(),
        'variable_02': pd.DataFrame()
    }

    return dict_vars


def plotTrainSize(plotTrainSize, d):

    """
    Plot Bhattacharyya coef. of train/test split
    ----------
    Parameters
    ----------
    True: Return the Bhattacharyya coef plot
    Fales: Do nothing
    ----------
    """

    if plotTrainSize:
        if (d['activate_baseline']) and (d['set_baseVarName']):
            bhattaVar = list(
                set(d['set_baseVarName'] + [d['set_depVarName']] + d['set_mediaVarName'] + d['set_mediaSpendName']))
        else:
            exit("either set activate_baseline = F or fill set_baseVarName")
        bhattaVar = list(set(bhattaVar) - set(d['set_factorVarName']))
        if 'depVar' not in d['df_Input'].columns:
            dt_bhatta = d['df_Input'][bhattaVar]
        else:
            bhattaVar = ['depVar' if i == d['set_depVarName'][0] else i for i in bhattaVar]
            dt_bhatta = d['df_Input'][bhattaVar]

        ## define bhattacharyya distance function
        def f_bhattaCoef(mu1, mu2, Sigma1, Sigma2):

            from scipy.spatial import distance

            Sig = (Sigma1 + Sigma2) / 2
            ldet_s = np.linalg.slogdet(Sig)[1]
            ldet_s1 = np.linalg.slogdet(Sigma1)[1]
            ldet_s2 = np.linalg.slogdet(Sigma2)[1]
            d1 = distance.mahalanobis(mu1, mu2, np.linalg.inv(Sig)) / 8
            d2 = 0.5 * ldet_s - 0.25 * ldet_s1 - 0.25 * ldet_s2
            d = d1 + d2
            bhatta_coef = 1 / np.exp(d)

            return bhatta_coef

        ## loop all train sizes
        bcCollect = []
        sizeVec = np.linspace(0.5, 0.9, 41)

        for size in sizeVec:
            test1 = dt_bhatta[0:(math.floor(len(dt_bhatta) * size))]
            test2 = dt_bhatta[(math.floor(len(dt_bhatta) * size)):]
            bcCollect.append(f_bhattaCoef(test1.mean(), test2.mean(), test1.cov(), test2.cov()))

        plt.plot(sizeVec, bcCollect)
        plt.xlabel("train_size")
        plt.ylabel("bhatta_coef")
        plt.title("Bhattacharyya coef. of train/test split \n- Select the training size with larger bhatta_coef",
                  loc='left')
        plt.show()


########################
# TODO plotAdstockCurves


########################
# TODO plotResponseCurves


def checkConditions(dt_transform, d: dict, set_hyperBoundLocal, set_lift=None):
    """
    check all conditions 1 by 1; terminate and raise errors if conditions are not met
    :param dt_transformations:
    :return: dictionary
    """
    try:
        d['set_mediaVarName']
    except NameError:
        print('set_mediaVarName must be specified')

    if d['activate_prophet'] and not set(d['set_prophet']).issubset({'trend', 'season', 'weekday', 'holiday'}):
        raise ValueError('set_prophet must be "trend", "season", "weekday" or "holiday"')
    if d['activate_baseline']:
        if len(d['set_baseVarName']) != len(d['set_baseVarSign']):
            raise ValueError('set_baseVarName and set_baseVarSign have to be the same length')

    if len(d['set_mediaVarName']) != len(d['set_mediaVarSign']):
        raise ValueError('set_mediaVarName and set_mediaVarSign have to be the same length')
    if not (set(d['set_prophetVarSign']).issubset({"positive", "negative", "default"}) and
            set(d['set_baseVarSign']).issubset({"positive", "negative", "default"}) and
            set(d['set_mediaVarSign']).issubset({"positive", "negative", "default"})):
        raise ValueError('set_prophetVarSign, '
                         'set_baseVarSign & set_mediaVarSign must be "positive", "negative" or "default"')
    if d['activate_calibration']:
        if set_lift.shape[0] == 0:
            raise ValueError('please provide lift result or set activate_calibration = FALSE')
        if (min(set_lift['liftStartDate']) < min(dt_transform['ds'])
                or (max(set_lift['liftEndDate']) > max(dt_transform['ds']) + timedelta(days=d['dayInterval'] - 1))):
            raise ValueError(
                'we recommend you to only use lift results conducted within your MMM input data date range')

        if d['set_iter'] < 500 or d['set_trial'] < 80:
            raise ValueError('you are calibrating MMM. we recommend to run at least 500 iterations '
                             'per trial and at least 80 trials at the beginning')

    if d['adstock'] not in ['geometric', 'weibull']:
        raise ValueError('adstock must be "geometric" or "weibull"')
    if d['adstock'] == 'geometric':
        num_hp_channel = 3
    else:
        num_hp_channel = 4
    # TODO: check hyperparameter names?
    if set(gethypernames(d)) != set(list(set_hyperBoundLocal.keys())):
        raise ValueError('set_hyperBoundLocal has incorrect hyperparameters')
    if dt_transform.isna().any(axis=None):
        raise ValueError('input data includes NaN')
    if np.isinf(dt_transform).any():
        raise ValueError('input data includes Inf')

    return d


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
        x_out = f'{round(x_in/1000000000, 1)} bln'
    elif n_commas == 2:
        x_out = f'{round(x_in/1000000, 1)} mio'
    elif n_commas == 1:
        x_out = f'{round(x_in/1000, 1)} tsd'
    else:
        x_out = str(int(round(x_in, 0)))

    return x_out


def michaelis_menten(spend, vmax, km):
    """

    :param vmax:
    :param spend:
    :param km:
    :return:
    """

    return vmax * spend/(km + spend)


def inputWrangling(dt, dt_holiday, d, set_lift, set_hyperBoundLocal):
    dt_transform = dt.copy().reset_index()
    dt_transform = dt_transform.rename({d['set_dateVarName']: 'ds'}, axis=1)
    dt_transform['ds'] = pd.to_datetime(dt_transform['ds'], format='%Y-%m-%d')
    dt_transform = dt_transform.rename({d['set_depVarName']: 'depVar'}, axis=1)
    dt_holiday['ds'] = pd.to_datetime(dt_holiday['ds'], format='%Y-%m-%d')
    # check date format
    try:
        pd.to_datetime(dt_transform['ds'], format='%Y-%m-%d', errors='raise')
    except ValueError:
        print('input date variable should have format "yyyy-mm-dd"')

    # check variable existence
    if not d['activate_prophet']:
        d['set_prophet'] = None
        d['set_prophetVarSign'] = None

    if not d['activate_baseline']:
        d['set_baseVarName'] = None
        d['set_baseVarSign'] = None

    if not d['activate_calibration']:
        d['set_lift'] = None

    try:
        d['set_mediaSpendName']
    except NameError:
        print('set_mediaSpendName must be specified')

    if len(d['set_mediaVarName']) != len(d['set_mediaVarSign']):
        raise ValueError('set_mediaVarName and set_mediaVarSign have to be the same length')

    trainSize = round(dt_transform.shape[0] * d['set_modTrainSize'])
    dt_train = dt_transform[d['set_mediaVarName']].iloc[:trainSize, :]
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
    d['dayInterval'] = dayInterval
    mediaVarCount = len(d['set_mediaVarName'])

    ################################################################
    #### model reach metric from spend
    mediaCostFactor = pd.DataFrame(dt_transform[d['set_mediaSpendName']].sum(axis=0), columns=['total_spend']).reset_index()
    var_total = pd.DataFrame(dt_transform[d['set_mediaVarName']].sum(axis=0), columns=['total_var']).reset_index()
    mediaCostFactor['mediaCostFactor'] = mediaCostFactor['total_spend']/var_total['total_var']
    mediaCostFactor = mediaCostFactor.drop(columns=['total_spend'])
    costSelector = pd.Series(d['set_mediaSpendName']) != pd.Series(d['set_mediaVarName'])

    if len(costSelector) != 0:
        modNLSCollect = defaultdict()
        yhatCollect = []
        plotNLSCollect = []
        for i in range(mediaVarCount - 1):
            if costSelector[i]:
                dt_spendModInput = pd.DataFrame(dt_transform.loc[:, d['set_mediaSpendName'][i]])
                dt_spendModInput['reach'] = dt_transform.loc[:, d['set_mediaVarName'][i]]
                dt_spendModInput.loc[dt_spendModInput[d['set_mediaSpendName'][i]] == 0, d['set_mediaSpendName'][i]] = 0.01
                dt_spendModInput.loc[dt_spendModInput['reach'] == 0, 'reach'] = \
                    dt_spendModInput[dt_spendModInput['reach'] == 0][d['set_mediaSpendName'][i]]/mediaCostFactor['mediaCostFactor'][i]

                #Michaelis-Menten model
                #vmax = max(dt_spendModInput['reach'])/2
                #km = max(dt_spendModInput['reach'])
                #y = michaelis_menten(dt_spendModInput[d['set_mediaSpendName'][i]], vmax, km)
                popt, pcov = curve_fit(michaelis_menten, dt_spendModInput[d['set_mediaSpendName'][i]], dt_spendModInput['reach'])

                yhatNLS = michaelis_menten(dt_spendModInput[d['set_mediaSpendName'][i]], *popt)
                #nls_pred = yhatNLS.predict(np.array(dt_spendModInput[d['set_mediaSpendName'][i]]).reshape(-1, 1))

                #linear model
                lm = LinearRegression().fit(np.array(dt_spendModInput[d['set_mediaSpendName'][i]])
                                            .reshape(-1, 1), np.array(dt_spendModInput['reach']).reshape(-1, 1))
                lm_pred = lm.predict(np.array(dt_spendModInput[d['set_mediaSpendName'][i]]).reshape(-1, 1))

                # compare NLS & LM, takes LM if NLS fits worse
                rsq_nls = r2_score(dt_spendModInput['reach'], yhatNLS)
                rsq_lm = r2_score(dt_spendModInput['reach'], lm_pred)
                costSelector[i] = rsq_nls > rsq_lm

                modNLSCollect[d['set_mediaSpendName'][i]] = {'vmax': popt[0], 'km': popt[1], 'rsq_lm': rsq_lm
                                                             , 'rsq_nls': rsq_nls, 'coef_lm': lm.coef_}

                yhat_dt = pd.DataFrame(dt_spendModInput['reach']).rename(columns={'reach': 'y'})
                yhat_dt['channel'] = d['set_mediaVarName'][i]
                yhat_dt['x'] = dt_spendModInput[d['set_mediaSpendName'][i]]
                yhat_dt['yhat'] = yhatNLS if costSelector[i] else lm_pred
                yhat_dt['models'] = 'nls' if costSelector[i] else 'lm'
                yhatCollect.append(yhat_dt)

                # TODO: generate plots


    d['plotNLSCollect'] = plotNLSCollect
    d['modNLSCollect'] = modNLSCollect
    #d['yhatNLSCollect'] = yhatNLSCollect

    getSpendSum = pd.DataFrame(dt_transform[d['set_mediaSpendName']].sum(axis=0), columns=['total_spend']).T

    d['mediaCostFactor'] = mediaCostFactor
    d['costSelector'] = costSelector
    d['getSpendSum'] = getSpendSum

    ################################################################
    #### clean & aggregate data
    all_name = [['ds'], ['depVar'], d['set_prophet'], d['set_baseVarName'], d['set_mediaVarName']]
    all_name = set([item for sublist in all_name for item in sublist])
    #all_mod_name = [['ds'], ['depVar'], d['set_prophet'], d['set_baseVarName'], d['set_mediaVarName']]
    #all_mod_name = [item for sublist in all_mod_name for item in sublist]
    if len(all_name) != len(set(all_name)):
        raise ValueError('Input variables must have unique names')

    ## transform all factor variables
    try:
        d['set_factorVarName']
    except:
        pass
    finally:
        if len(d['set_factorVarName']) > 0:
            dt_transform[d['set_factorVarName']].apply(lambda x: x.astype('category'))
        else:
            d['set_factorVarName'] = None

    ################################################################
    #### Obtain prophet trend, seasonality and changepoints

    if d['activate_prophet']:
        if len(d['set_prophet']) != len(d['set_prophetVarSign']):
            raise ValueError('set_prophet and set_prophetVarSign have to be the same length')
        if len(d['set_prophet']) == 0 or len(d['set_prophetVarSign']) == 0:
            raise ValueError('if activate_prophet == TRUE, set_prophet and set_prophetVarSign must to specified')
        if d['set_country'] not in dt_holiday['country'].values:
            raise ValueError(
                'set_country must be already included in the holidays.csv and as ISO 3166-1 alpha-2 abbreviation')

        recurrence = dt_transform.copy().rename(columns={'depVar': 'y'})
        use_trend = True if 'trend' in d['set_prophet'] else False
        use_season = True if 'season' in d['set_prophet'] else False
        use_weekday = True if 'weekday' in d['set_prophet'] else False
        use_holiday = True if 'holiday' in d['set_prophet'] else False

        if intervalType == 'day':
            holidays = dt_holiday
        elif intervalType == 'week':
            weekStartInput = dt_transform['ds'][0].weekday()
            if weekStartInput == 0:
                weekStartMonday = True
            elif weekStartInput == 6:
                weekStartMonday = False
            else:
                raise ValueError('week start has to be Monday or Sunday')
            dt_holiday['weekday'] = dt_holiday['ds'].apply(lambda x: x.weekday())
            dt_holiday['dsWeekStart'] = dt_holiday.apply(lambda x: x['ds'] - timedelta(days=x['weekday']), axis=1)
            dt_holiday['ds'] = dt_holiday['dsWeekStart']
            dt_holiday = dt_holiday.drop(['dsWeekStart', 'weekday'], axis=1)
            holidays = dt_holiday.groupby(['ds', 'country', 'year'])['holiday'].apply(
                lambda x: '#'.join(x)).reset_index()

        elif intervalType == 'month':
            monthStartInput = dt_transform['ds'][0].strftime("%d")
            if monthStartInput != '01':
                raise ValueError("monthly data should have first day of month as datestampe, e.g.'2020-01-01'")
            dt_holiday['month'] = dt_holiday['ds'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(normalize=True)
            dt_holiday['ds'] = dt_holiday['month']
            dt_holiday.drop(['month'], axis=1)
            holidays = dt_holiday.groupby(['ds', 'country', 'year'])['holiday'].apply(
                lambda x: '#'.join(x)).reset_index()
        h = holidays[holidays['country'] == d['set_country']] if use_holiday else None
        modelRecurrance = Prophet(holidays=h, yearly_seasonality=use_season, weekly_seasonality=use_weekday,
                                  daily_seasonality=False)
        modelRecurrance.fit(recurrence)
        forecastRecurrance = modelRecurrance.predict(dt_transform)

        d['modelRecurrance'] = modelRecurrance
        d['forecastRecurrance'] = forecastRecurrance

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

    checkConditions(dt_transform, d,set_hyperBoundLocal, set_lift)

    return dt_transform, d



def gethypernames(adstock, set_mediaVarName):
    """
    Parameters
    ----------
    adstock: chosen adstock (geometric or weibull)
    set_mediaVarName: list of media channels
    Returns
    -------
    crossed list of adstock parameters and media channels
    """

    if adstock == "geometric":
        global_name = ["thetas", "alphas", "gammas"]
        local_name = sorted(list([i+"_"+str(j) for i in set_mediaVarName for j in global_name]))
    elif adstock == "weibull":
        global_name = ["shapes", "scales", "alphas", "gammas"]
        local_name = sorted(list([i+"_"+str(j) for i in set_mediaVarName for j in global_name]))

    return local_name


def adstockGeometric(x, theta):
    """
    Parameters
    ----------
    x: vector
    theta: decay coefficient
    Returns
    -------
    transformed vector
    """
    x_decayed = [x[0]] + [0] * (len(x) - 1)
    for i in range(1, len(x_decayed)):
        x_decayed[i] = x[i] + theta * x_decayed[i - 1]

    return x_decayed


def helperWeibull(x, y, vec_cum, n):
    """

    :param x:
    :param y:
    :param vec_cum:
    :param n:
    :return:
    """

    x_vec = np.array([0] * (y - 1) + [x] * (n - y + 1))
    vec_lag = np.roll(vec_cum, y - 1)
    vec_lag[: y - 1] = 0
    x_matrix = np.c_[x_vec, vec_cum]
    x_prod = np.multiply.reduce(x_matrix, axis=1)

    return x_prod


def adstockWeibull(x, shape, scale):
    """
    Parameters
    ----------
    x: vector
    shape: shape parameter for Weibull
    scale: scale parameter for Weibull
    Returns
    -------
    tuple
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html

    n = len(x)
    bin =list(range(1, n + 1))
    scaleTrans = round(np.quantile(bin, scale))
    thetaVec = 1 - stats.weibull_min.cdf(bin[:-1], shape, scale=scaleTrans)
    thetaVec = np.concatenate(([1], thetaVec))
    thetaVecCum = np.cumprod(thetaVec).tolist()

    x_decayed = list(map(lambda i, j: helperWeibull(i, j, vec_cum=thetaVecCum, n=n), x, bin))
    x_decayed = np.concatenate(x_decayed, axis=0).reshape(7, 7)
    x_decayed = np.sum(x_decayed, axis=1).tolist()

    return x_decayed, thetaVecCum


def transformation(x, adstock, theta=None, shape=None, scale=None, alpha=None, gamma=None, stage=3):
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
        x_decayed = adstockGeometric(x, theta)

        if stage == "thetaVecCum":
            thetaVecCum = theta
        for t in range(1, len(x) - 1):
            thetaVecCum[t] = thetaVecCum[t - 1] * theta
        # thetaVecCum.plot()

    elif adstock == "weibull":
        x_list = adstockWeibull(x, shape, scale)
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
    gammaTrans = round(np.quantile(np.linspace(min(x_normalized), max(normalized), 100), gamma), 4)
    x_scurve = x_normalized ** alpha / (x_normalized ** alpha + gammaTrans ** alpha)
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


def rsq(true, predicted):
    """
    ----------
    Parameters
    ----------
    true: true value
    predicted: predicted value
    Returns
    -------
    r-squared
    """
    sse = sum((predicted - true) ** 2)
    sst = sum((true - sum(true)/len(true)) ** 2)
    return 1 - sse / sst


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


########################
# TODO decomp


########################
# TODO calibrateLift

# def calibrateLift(xDecompOut, xDecompOut.scaled, decompOutAgg, set_lift):
#     """
#
#     :param xDecompOut:
#     :param xDecompOut.scaled:
#     :param decompOutAgg:
#     :param set_lift:
#     :return:
#     """
#
#     lift_channels = list(set_lift.channel)
#     check_set_lift = all(item in set_mediaVarName for item in lift_channels)
#     if check_set_lift:
#         getLiftMedia = list(set(lift_channels))
#         getDecompVec = xDecompOut
#     else:
#         exit("set_lift channels must have media variable")
#
#     # loop all lift input
#     liftCollect = pd.DataFrame(columns = ['liftMedia', 'liftStart', 'liftEnd' ,
#                                           'liftAbs', 'decompAbsScaled', 'dependent'])
#     for m in getLiftMedia: # loop per lift channel
#         liftWhich = list(set_lift.loc[set_lift.channel.isin([m])].index)
#         liftCollect2 = pd.DataFrame(columns = ['liftMedia', 'liftStart', 'liftEnd' ,
#                                                'liftAbs', 'decompAbsScaled', 'dependent'])
#         for lw in liftWhich: # loop per lift test per channel
#             # get lift period subset
#             liftStart = set_lift['liftStartDate'].iloc[lw]
#             liftEnd = set_lift['liftEndDate'].iloc[lw]
#             liftAbs = set_lift['liftAbs'].iloc[lw]
#             liftPeriodVec = getDecompVec[['ds', m]][(getDecompVec.ds >= liftStart) & (getDecompVec.ds <= liftEnd)]
#             liftPeriodVecDependent = getDecompVec[['ds', 'y']][(getDecompVec.ds >= liftStart) & (getDecompVec.ds <= liftEnd)]
#
#             # scale decomp
#             mmmDays = len(liftPeriodVec)*7
#             liftDays = abs((liftEnd - liftStart).days) + 1
#             y_hatLift = getDecompVec['y_hat'].sum() # total predicted sales
#             x_decompLift = liftPeriodVec.iloc[:1].sum()
#             x_decompLiftScaled = x_decompLift / mmmDays * liftDays
#             y_scaledLift = liftPeriodVecDependent['y'].sum() / mmmDays * liftDays
#
#             # output
#             list_to_append = [[getLiftMedia[m], liftStart, liftEnd, liftAbs, x_decompLiftScaled, y_scaledLift]]
#             liftCollect2 = liftCollect2.append(pd.DataFrame(list_to_append,
#                                                             columns = ['liftMedia', 'liftStart', 'liftEnd' ,
#                                                                        'liftAbs', 'decompAbsScaled', 'dependent'],
#                                                             ignore_index = True))
#         liftCollect = liftCollect.append(liftCollect2, ignore_index = True)
#     #get mape_lift
#     liftCollect['mape_lift'] = abs((liftCollect['decompAbsScaled'] - liftCollect['liftAbs']) / liftCollect['liftAbs'])
#
#     return liftCollect

########################
# TODO refit

def refit(x_train: np.array(), y_train: np.array(), lambda_: int, lower_limits: list, upper_limits: list):
    """

    :param x_train: numpy array; rows = record; columns = betas
        e.g. np.array([[1, 1, 2], [3, 4, 2], [6, 5, 2], [5, 5, 3]])
    :param y_train: numpy array (1xn) of outcomes
        e.g. np.array([1, 0, 0, 1])
    :param lambda_: expects integer
        e.g. 1
    :param lower_limits:
        e.g. [0, 0, 0, 0]
    :param upper_limits:
        e.g. [10, 10, 10, 10]
    :return: dictionary of model outputs including rsq_train, nrmse_train, coefs, y_pred, model
    """

    # # FOR TESTING #TODO GET RID OF THE TESTING PARAMETERS
    # import numpy as np
    # x_train = np.array([[1, 1, 2], [3, 4, 2], [6, 5, 2], [5, 5, 3]])
    # y_train = np.array([1, 0, 0, 1])
    # upper_limits = [10, 10, 10, 10]
    # lower_limits = [0, 0, 0, 0]
    # lambda_ = 1

    # GREAT PACKAGE, BUT REQUIRES LINUX
    # https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html
    # https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#intro
    # https://pypi.org/project/glmnet-python/
    # import glmnet_python
    # from glmnet import glmnet

    # WORKS BUT DOES NOT ALLOW LIMITS TO BE PASSED IN
    # from sklearn.linear_model import Ridge
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
    # https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
    # https://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons
    # NO UPPER AND LOWER LIMITS PASSED INTO RIDGE REGRESSION.
    # mod = Ridge(alpha=1,
    #             fit_intercept=True,
    #             normalize=False,
    #             tol=1e-3,
    #             solver='auto',
    #             random_state=None)
    # mod.fit(X=x, y=y)
    # mod.intercept_

    # WORKS BUT DOES NOT ALLOW LIMITS TO BE PASSED IN
    # from pyglmnet import GLM
    # NO UPPER AND LOWER LIMITS PASSED INTO RIDGE REGRESSION.
    # glm = GLM(distr='poisson',
    #           alpha=0,
    #           Tau=None,
    #           group=None,
    #           reg_lambda=0.1,
    #           solver='batch-gradient',
    #           learning_rate=2e-1,
    #           max_iter=1000,
    #           tol=1e-6,
    #           eta=2.0,
    #           score_metric='deviance',
    #           fit_intercept=True,
    #           random_state=0,
    #           callback=None,
    #           verbose=False)
    # glm.get_params()

    # R GLMNET R GLMNET R GLMNET R GLMNET R GLMNET R GLMNET
    # https://rpy2.github.io/
    # https://github.com/conda-forge/r-glmnet-feedstock/issues/1
    # conda install -c conda-forge rpy2
    # conda install -c conda-forge r r-essentials
    # conda install -c conda-forge r glmnet

    # WORKING SOLUTION TO CALL R FUNCTIONS
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    # define glmnet model in r
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

    # create model
    mod = r_glmnet(x=x_train,
                   y=y_train,
                   alpha=1,
                   family="gaussian",
                   lambda_=lambda_,
                   lower_limits=lower_limits,
                   upper_limits=upper_limits,
                   intercept=True
                   )

    # could use if we can get lambda to work
    # from rpy2.robjects.packages import importr
    # glmnet = importr('glmnet')
    # mod = glmnet.glmnet(x_train,
    #                     y_train,
    #                     alpha=1,
    #                     family='gaussian',
    #                     # lambda=lambda_,
    #                     lower_limits=lower_limits,
    #                     upper_limits=upper_limits
    #                     )

    # create model without the intercept if negative
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

    # run model
    ro.r('''
                r_predict <- function(model, s, newx) {
                    predict(model, s=s, newx=newx)
                }
            ''')
    r_predict = ro.globalenv['r_predict']
    y_train_pred = r_predict(model=mod, s=1, newx=x_train)
    y_train_pred = y_train_pred.reshape(len(y_train_pred), )  # reshape to be of format (n,)

    # calc r-squared on training set
    rsq_train = rsq(true=y_train, predicted=y_train_pred)

    # get coefficients
    coefs = mod[0]

    # get normalized root mean square error
    nrmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2) / (max(y_train) - min(y_train)))

    # update model outputs to include calculated values
    mod_out = {'rsq_train': rsq_train,
               'nrmse_train': nrmse_train,
               'coefs': coefs,
               'y_pred': y_train_pred,
               'mod': mod}

    return mod_out


########################
# TODO mmm


########################
# TODO robyn


















