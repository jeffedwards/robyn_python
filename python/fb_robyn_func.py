import pandas as pd
import math
import numpy as np
#import weibull as weibull
from sklearn import preprocessing
from scipy import stats
import matplotlib.pyplot as plt

import python.setting as input


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


def plotTrainSize(plotTrainSize):

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
        if (input.activate_baseline) and (input.set_baseVarName):
            bhattaVar = list(set(input.set_baseVarName + input.set_depVarName + input.set_mediaVarName + input.set_mediaSpendName))
        else:
            exit("either set activate_baseline = F or fill set_baseVarName")
        bhattaVar = list(set(bhattaVar) - set(input.set_factorVarName))
        if 'depVar' not in input.df_Input.columns:
            dt_bhatta = input.df_Input[bhattaVar]
        else:
            bhattaVar = ['depVar' if i == input.set_depVarName[0] else i for i in bhattaVar]
            dt_bhatta = input.df_Input[bhattaVar]

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
        plt.title("Bhattacharyya coef. of train/test split \n- Select the training size with larger bhatta_coef", loc='left')
        plt.show()
