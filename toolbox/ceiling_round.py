"""Custom round function for values larger than a particular value.

Functions
---------
ceiling_round

"""

__author__ = "Elizabeth Barnes and Randal J Barnes"
__version__ = "8 November 2021"


def ceiling_round(x,ceiling):
    """Custom round function to round to a ceiling value for values larger than that.

    Arguments
    ---------
    x : float
        List of values to be rounded
    ceiling : float 
        cutoff all larger values will be set to

    Returns
    -------
    vector of length x with a ceiling at the specified ceiling 

    """
    if x > ceiling:
        return ceiling
    else:
        return x




