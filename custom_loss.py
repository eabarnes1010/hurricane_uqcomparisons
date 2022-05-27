"""Custom loss functions for nonlinear, heterogeneous, parametric regression
    using the sinh-arcsinh normal distribution.

Functions
---------
compute_bnnshash_NLL(y,distr)
compute_NLL(y,distr)
compute_shash_NLL(y_true, pred)

"""
import tensorflow as tf
import shash

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "27 May 2022"


def compute_bnnshash_NLL(y, distr): 
    return -2.*distr.log_prob(y)  #900: 5, 901:2


def compute_NLL(y, distr): 
    return -distr.log_prob(y) 


def compute_shash_NLL(y_true, pred):
    """Negative log-likelihood loss using the sinh-arcsinh normal distribution.

    Arguments
    ---------
    y_true : tensor
        The ground truth values.
        shape = [batch_size, n_parameter]

    pred :
        The predicted local conditionsal distribution parameter values.
        shape = [batch_size, n_parameters]

    Returns
    -------
    loss : tensor, shape = [1, 1]
        The average negative log-likelihood of the batch using the predicted
        conditional distribution parameters.

    Notes
    -----
    * The value of n_parameters depends on the chosen form of the conditional
        sinh-arcsinh normal distribution.
            shash2 -> n_parameter = 2, i.e. mu, sigma
            shash3 -> n_parameter = 3, i.e. mu, sigma, gamma
            shash4 -> n_parameter = 4, i.e. mu, sigma, gamma, tau

    * Since sigma and tau must be strictly positive, the network learns the
        log of these two parameters.

    * If gamma is not learned (i.e. shash2), they are set to 0.

    * If tau is not learned (i.e. shash2 or shash3), they are set to 1.

    """
    mu = pred[:, 0]
    sigma = pred[:, 1]

    if pred.shape[1] >= 3:
        gamma = pred[:, 2]
    else:
        gamma = tf.zeros_like(mu)

    if pred.shape[1] >= 4:
        tau = pred[:, 3]
    else:
        tau = tf.ones_like(mu)

    loss = -shash.log_prob(y_true[:, 0], mu, sigma, gamma, tau)
    return tf.reduce_mean(loss, axis=-1)
