########################################################################################################################
# IMPORTS


########################################################################################################################
# FUNCTIONS

def unit_format(x_in):
    """
    :param x_in: a number in decimal or float format
    :return: the number rounded and in certain cases abbreviated in the thousands, millions, or billions
    """

    # suffixes = ["", "Thousand", "Million", "Billion", "Trillion", "Quadrillion"]
    number = str("{:,}".format(x_in))
    n_commas = number.count(',')
    # print(number.split(',')[0], suffixes[n_commas])

    if n_commas >= 3:
        x_out = f'{round(x_in/1000000000, 1)} bln'
    elif n_commas == 2:
        x_out = f'{round(x_in/1000000, 1)} mio'
    elif n_commas == 1:
        x_out = f'{round(x_in/1000, 1)} tsd'
    else:
        x_out = str(int(round(x_in, 0)))

    return x_out
