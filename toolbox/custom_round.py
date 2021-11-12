"""Custom round function.

Functions
---------
custom_round

"""

__author__ = "Elizabeth Barnes and Randal J Barnes"
__version__ = "30 October 2021"


def custom_round(x, base=5):
    """Custom round function to round to nearest base, including integer

    Arguments
    ---------
    x : float
        List of values to be rounded
    base : float default=5
        base to be rounded to

    Returns
    -------
    vector of length x rounded to nearest base

    """
    return int(base * round(float(x)/base))
