"""Prediction functions.

Functions
---------
params
percentile_value

"""
import numpy as np
import shash_tfp

__author__ = "Elizabeth Barnes and Randal J Barnes"
__version__ = "30 May 2022"


def params(x_inputs, model):
    """Funtion to make shash parameter predictions for shash2, shash3, shash4

    Arguments
    ---------
    x_inputs : floats
        matrix of inputs to the network

    model : tensorflow model
        neural network for predictions

    Returns
    -------
    vector of predicted parameters

    """
    y_pred = model.predict(x_inputs)
    mu_pred = y_pred[:, 0]
    sigma_pred = y_pred[:, 1]

    gamma_pred = np.zeros(np.shape(y_pred[:, 0]),dtype='float32')
    if np.shape(y_pred)[1] >= 3:
        gamma_pred = y_pred[:, 2]

    tau_pred = np.ones(np.shape(y_pred[:, 0]),dtype='float32')
    if np.shape(y_pred)[1] >=4 :
        tau_pred = y_pred[:, 3]

    return mu_pred, sigma_pred, gamma_pred, tau_pred


def percentile_value(mu_pred, sigma_pred, gamma_pred, tau_pred, percentile_frac=0.5):
    """Function to obtain percentile value of the shash distribution."""
    
    dist = shash_tfp.Shash(mu_pred, sigma_pred, gamma_pred, tau_pred)
    
    return dist.quantile(percentile_frac).numpy()
