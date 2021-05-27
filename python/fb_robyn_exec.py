# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS

from python import fb_robyn_func as frf

# Used to reload an import during development
# TODO delete after development is complete
import importlib as il
il.reload(frf)


########################################################################################################################
# LOAD DATA & SCRIPTS

# todo Remove testing dictionary when done
dict_vars = frf.initiate_testing_dictionary()
# dict_vars = frf.initiate_dictionary()

robyn = frf.Robyn()
robyn.upper_limits


########################################################################################################################
# SET MODEL INPUT VARIABLES


########################################################################################################################
# SET GLOBAL MODEL PARAMETERS


########################################################################################################################
# TUNE CHANNEL HYPER-PARAMETERS BOUNDS


########################################################################################################################
# DEFINE GROUND TRUTH (E.G. GEO TEST, FB LIFT TEST, MTA ETC.)


########################################################################################################################
# PREPARE INPUT DATA


########################################################################################################################
# RUN MODELS

########################################################################################################################
# BUDGET ALLOCATOR - BETA

