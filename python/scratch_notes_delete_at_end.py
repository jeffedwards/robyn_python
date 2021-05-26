# REFIT
# # FOR TESTING #TODO GET RID OF THE TESTING PARAMETERS
import numpy as np
x_train = np.array([[1, 1, 2], [3, 4, 2], [6, 5, 2], [5, 5, 3]])
y_train = np.array([1, 0, 0, 1])
upper_limits = [10, 10, 10, 10]
lower_limits = [0, 0, 0, 0]
lambda_ = 1

# GREAT PACKAGE, BUT REQUIRES LINUX
# https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html
# https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#intro
# https://pypi.org/project/glmnet-python/
import glmnet_python
from glmnet import glmnet

# WORKS BUT DOES NOT ALLOW LIMITS TO BE PASSED IN
# from sklearn.linear_model import Ridge
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
# https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
# https://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons
# NO UPPER AND LOWER LIMITS PASSED INTO RIDGE REGRESSION.
mod = Ridge(alpha=1,
            fit_intercept=True,
            normalize=False,
            tol=1e-3,
            solver='auto',
            random_state=None)
mod.fit(X=x, y=y)
mod.intercept_

# WORKS BUT DOES NOT ALLOW LIMITS TO BE PASSED IN
from pyglmnet import GLM
# NO UPPER AND LOWER LIMITS PASSED INTO RIDGE REGRESSION.
glm = GLM(distr='poisson',
          alpha=0,
          Tau=None,
          group=None,
          reg_lambda=0.1,
          solver='batch-gradient',
          learning_rate=2e-1,
          max_iter=1000,
          tol=1e-6,
          eta=2.0,
          score_metric='deviance',
          fit_intercept=True,
          random_state=0,
          callback=None,
          verbose=False)
glm.get_params()
#
# R GLMNET R GLMNET R GLMNET R GLMNET R GLMNET R GLMNET
# https://rpy2.github.io/
# https://github.com/conda-forge/r-glmnet-feedstock/issues/1
# conda install -c conda-forge rpy2
# conda install -c conda-forge r r-essentials
# conda install -c conda-forge r glmnet

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