import pandas as pd
import math
import numpy as np
#import weibull as weibull
from sklearn import preprocessing
from scipy import stats


def rsq(true,predicted):
    """
    Define r-squared function
    """
    sse = sum((predicted - true) ** 2)
    sst = sum((true - sum(true)/len(true)) ** 2)
    return 1 - sse / sst


def lambdaRidge(x, y, seq_len=100, lambda_min_ratio=0.0001):
    """
    Define ridge lambda sequence function
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


def check_conditions(dt_transform):
    """
    check all conditions 1 by 1; terminate and raise errors if conditions are not met
    :param dt_transformations:
    :return:
    """
    try:
        set_mediaVarName
    except NameError:
        print('set_mediaVarName must be specified')

    if activate_prophet and set_prophet not in ['trend', 'season', 'weekday', 'holiday']:
        raise ValueError('set_prophet must be "trend", "season", "weekday" or "holiday"')
    elif activate_baseline:
        if len(set_baseVarName) != len(set_baseVarSign):
            raise ValueError('set_baseVarName and set_baseVarSign have to be the same length')
    elif len(set_mediaVarName) != len(set_mediaVarSign):
        raise ValueError('set_mediaVarName and set_mediaVarSign have to be the same length')
    elif not all(x in ["positive", "negative", "default"]
                 for x in ['set_prophetVarSign', 'set_baseVarSign', 'set_mediaVarSign']):
        raise ValueError('set_prophetVarSign, '
                         'set_baseVarSign & set_mediaVarSign must be "positive", "negative" or "default"')
    elif activate_calibration:
        if set_lift.shape[0] == 0:
            raise ValueError('please provide lift result or set activate_calibration = FALSE')
        elif:
            pass
        elif set_iter < 500 or set_trial < 80:
            raise ValueError('you are calibrating MMM. we recommend to run at least 500 iterations '
                             'per trial and at least 80 trials at the beginning')

    elif adstock not in ['geometric', 'weibull']:
        raise ValueError('adstock must be "geometric" or "weibull"')
    elif dt_transform.isna().any(axis = None):
        raise ValueError('input data includes NaN')
    elif dt_transform.isinf().any(axis = None):
        raise ValueError('input data includes Inf')
