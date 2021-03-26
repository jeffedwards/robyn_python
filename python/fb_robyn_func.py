import pandas as pd
import math
import numpy as np
#import weibull as weibull
from sklearn import preprocessing
from scipy import stats


def rsq(true,predicted):
    """
    Define r-squared function
    ----------
    true: true value
    predicted: predicted value
    Returns r-squared
    -------
    """
    sse = sum((predicted - true) ** 2)
    sst = sum((true - sum(true)/len(true)) ** 2)
    return 1 - sse / sst


def lambdaRidge(x, y, seq_len=100, lambda_min_ratio=0.0001):
    """
    Define ridge lambda sequence function
    ----------
    x: matrix
    y: vector
    Returns lambda sequence
    -------
    array
    """
    def mysd(x):
        return math.sqrt(sum((x - sum(x) / len(x)) ** 2) / len(x))

    sx = x.apply(mysd)
    sx = preprocessing.scale(x).T
    sy = y.to_numpy()
    sxy = sx * sy
    sxy = sxy.T

    lambda_max = max(abs(sxy.sum(axis=0)) / (
                0.001 * x.shape[0]))  # 0.001 is the default smalles alpha value of glmnet for ridge (alpha = 0)

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

