# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# IMPORTS
from python.v02 import fb_robyn as r

# FOR TESTING
import importlib as il
il.reload(r)

########################################################################################################################
# SCRIPT

# Initialize object
robyn = r.Robyn()

# See a parameter
robyn.y_train

# See variables
robyn.__dict__

# Run something
mod_sample = robyn.refit()  # don't need to return model if you do not want to
robyn.mod

# Example of update variables
robyn.set_variables()

