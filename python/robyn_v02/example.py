# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS
import pandas as pd
from python.robyn_v02 import robyn as r

# FOR TESTING
import importlib as il
il.reload(r)

########################################################################################################################
# EXAMPLES - HOW TO USE THE CLASS

# Initialize object
robyn = r.Robyn()

# See a parameter
robyn.test_y_train
robyn.iterations

# See all variables
robyn.__dict__

# See an empty class variable
robyn.mod

# Run something
robyn.refit(x_train=robyn.test_x_train,
            y_train=robyn.test_y_train,
            lambda_=robyn.test_lambda_,
            lower_limits=robyn.test_lower_limits,
            upper_limits=robyn.test_upper_limits)

# See the class variable that was just updated
robyn.mod


########################################################################################################################
# SCRIPT

# INITIALIZE OBJECT
robyn = r.Robyn()

# IMPORT DATA SET FOR PREDICTIONS
df = pd.read_csv('source/de_simulated_data.csv')

# PREPARE DATA FOR MODELING
df_transformed = robyn.input_wrangling(df)

# FIT MODEL
robyn.fit(df=df_transformed)

# BUDGE ALLOCATOR
robyn.allocate_budget(modID="3_10_2",
                      scenario="max_historical_response",
                      channel_constr_low=[0.7, 0.75, 0.60, 0.8, 0.65],
                      channel_constr_up=[1.2, 1.5, 1.5, 2, 1.5])