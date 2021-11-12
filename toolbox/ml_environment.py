"""Gather the version numbers for a set of machine learning packages.

Functions
---------
get_ml_environment()

"""
__author__ = "Randal J Barnes"
__version__ = "20 October 2021"

import sys

try:
    import matplotlib

    matplotlib_version = matplotlib.__version__
except ImportError:
    matplotlib_version = None

try:
    import numpy

    numpy_version = numpy.__version__
except ImportError:
    numpy_version = None

try:
    import pandas

    pandas_version = pandas.__version__
except ImportError:
    pandas_version = None

try:
    import scipy

    scipy_version = scipy.__version__
except ImportError:
    scipy_version = None

try:
    import sklearn

    sklearn_version = sklearn.__version__
except ImportError:
    sklearn_version = None

try:
    import tensorflow

    tensorflow_version = tensorflow.__version__
except ImportError:
    tensorflow_version = None

try:
    import tensorflow_probability

    tensorflow_probability_version = tensorflow_probability.__version__
except ImportError:
    tensorflow_probability_version = None


def get_ml_environment():
    return {
        "python": sys.version,
        "matplotlib": matplotlib_version,
        "numpy": numpy_version,
        "pandas": pandas_version,
        "scipy": scipy_version,
        "sklearn": sklearn_version,
        "tensorflow": tensorflow_version,
        "tensorflow_probability": tensorflow_probability_version,
    }
