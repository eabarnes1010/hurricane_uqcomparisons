"""Summary statistics table.

Functions
---------
print_summary_statistics(data_dictionary, sigfigs=2, plusfigs=1, file=sys.stdout)

"""
import sys

import numpy as np
import tabulate

__author__ = "Randal J Barnes and Elizabeth Barnes"
__version__ = "30 October 2021"


def print_summary_statistics(data_dictionary, sigfigs=2, plusfigs=1, file=sys.stdout):
    """Print a summary statistics table for the data in the dictionary.

    The summary statistics include:
        "cnt" : count
        "min" : minimum value
        "25%" : 25 percentile
        "50%" : median
        "75%" : 75 percentile
        "max" : maximum value
        "avg" : arithmetic average
        "std" : standard deviation (sample)
        "irq" : inter-quartile range

    Arguments
    ---------
    data_dictionary : dict
        The key:values pairs in the dictionary are name and data vector.
        The name is a str, the data vector is a ndarray of shape (n,).

    sigfigs : int, default=2
        The number of places to the right of the decimal point used to
        print the data order statistics.

    plusfigs : int, default=1
        The number of additional places to the right of the decimal point
        used to print the mean and standard deviation.

    file : file object, default=sys.stdout (i.e. the screen)
        Where to print the table.

    Returns
    -------
    None

    Usage
    -----
    print_summary_statistics({"y_train" : y_train, "y_val" : y_val})

    Notes
    -----
    * The computations do not filter out nan values.  So, if there are
        nan or inf values in the data vector, the resulting statistics
        will be impacted.

    * If the computation of any statistic fails and raises an exception,
        then the entire row in the table is set to nan.

    """
    # Make the table.
    table = []
    for key in data_dictionary:
        data_vector = data_dictionary[key]
        try:
            row = [
                key,
                np.shape(data_vector)[0],
                np.min(data_vector),
                np.percentile(data_vector, 25),
                np.median(data_vector),
                np.percentile(data_vector, 75),
                np.max(data_vector),
                np.mean(data_vector),
                np.std(data_vector),
                np.percentile(data_vector, 75) - np.percentile(data_vector, 25),
            ]
        except:
            row = [
                key,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        table.append(row)

    # Print a table.
    print("\n", file=file)
    print(
        tabulate.tabulate(
            table,
            [" ", "cnt", "min", "25%", "50%", "75%", "max", "avg", "std", "irq"],
            tablefmt="presto",
            floatfmt=(
                "s",
                ".0f",
                f".{sigfigs}f",
                f".{sigfigs}f",
                f".{sigfigs}f",
                f".{sigfigs}f",
                f".{sigfigs}f",
                f".{sigfigs+plusfigs}f",
                f".{sigfigs+plusfigs}f",
                f".{sigfigs}f",
            ),
        ),
        file=file,
    )
    print("\n", file=file)
