# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################
###### Data transformation and helper functions
########################################################################

import python.setting as input

################################################################
#### Define training size guidance plot using Bhattacharyya coefficient




def plotTrainSize(plotTrainSize):

    if plotTrainSize:
        if (input.activate_baseline) and (input.set_baseVarName):
            bhattaVar = list(set(input.set_baseVarName + input.set_depVarName + input.set_mediaVarName + input.set_mediaSpendName))
            print(bhattaVar)