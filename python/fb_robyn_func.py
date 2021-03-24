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
        else:
            exit("either set activate_baseline = F or fill set_baseVarName")

        bhattaVar = list(set(bhattaVar) - set(input.set_factorVarName))

        if 'depVar' not in input.df_Input.columns:
            dt_bhatta = input.df_Input[bhattaVar]
        else:
            bhattaVar = ['depVar' if i == input.set_depVarName[0] else i for i in bhattaVar]
            dt_bhatta = input.df_Input[bhattaVar]

        def f_bhattaCoef(mu1, mu2, Sigma1, Sigma2):
            Sig = (Sigma1 + Sigma2) / 2

            '''
            ldet_s = unlist(determinant(Sig, logarithm=TRUE))[1]
            ldet_s1 < - unlist(determinant(Sigma1, logarithm=TRUE))[1]
            ldet_s2 < - unlist(determinant(Sigma2, logarithm=TRUE))[1]
            d1 < - mahalanobis(mu1, mu2, Sig, tol=1e-20) / 8
            d2 < - 0.5 * ldet.s - 0.25 * ldet.s1 - 0.25 * ldet.s2
            d < - d1 + d2
            bhatta.coef < - 1 / exp(d)
            return (bhatta.coef)
            '''