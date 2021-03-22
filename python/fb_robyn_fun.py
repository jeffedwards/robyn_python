import pandas as pd
import math
import numpy as np


def rsq(true,predicted):
    """
    Define r-squared function
    """
    sse = sum((predicted - true) ** 2)
    sst = sum((true - sum(true)/len(true)) ** 2)
    return 1 - sse / sst

