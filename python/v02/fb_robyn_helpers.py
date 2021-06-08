# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

########################################################################################################################
# FUNCTIONS


def rsq(true, predicted):
    """
    :param true:
    :param predicted:
    :return:
    """

    sse = sum((predicted - true) ** 2)
    sst = sum((true - sum(true)/len(true)) ** 2)
    return 1 - sse / sst
