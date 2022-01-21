"""sinh-arcsinh normal distribution w/o using tensorflow_probability.

Functions
---------
cdf(x, mu, sigma, gamma, tau=None)
    cumulative distribution function (cdf).

log_prob(x, mu, sigma, gamma, tau=None)
    log of the probability density function.

mean(mu, sigma, gamma, tau=None)
    distribution mean.

median(mu, sigma, gamma, tau=None)
    distribution median.

prob(x, mu, sigma, gamma, tau=None)
    probability density function (pdf).

quantile(pr, mu, sigma, gamma, tau=None)
    inverse cumulative distribution function.

rvs(mu, sigma, gamma, tau=None, size=1)
    generate random variates.

stddev(mu, sigma, gamma, tau=None)
    distribution standard deviation.

variance(mu, sigma, gamma, tau=None)
    distribution variance.

Notes
-----
* This module uses only tensorflow.  This module does not use the
tensorflow_probability library.

* The sinh-arcsinh normal distribution was defined in [1]. A more accessible
presentation is given in [2]. 

* The notation and formulation used in this code was taken from [3], page 143.
In the gamlss.dist/CRAN package the distribution is called SHASHo. 

* There is a typographical error in the presentation of the probability 
density function on page 143 of [3]. There is an extra "2" in the denomenator
preceeding the "sqrt{1 + z^2}" term.

References
----------
[1] Jones, M. C. & Pewsey, A., Sinh-arcsinh distributions,
Biometrika, Oxford University Press, 2009, 96, 761-780.
DOI: 10.1093/biomet/asp053.

[2] Jones, C. & Pewsey, A., The sinh-arcsinh normal distribution,
Significance, Wiley, 2019, 16, 6-7.
DOI: 10.1111/j.1740-9713.2019.01245.x.
https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1740-9713.2019.01245.x

[3] Stasinopoulos, Mikis, et al. (2021), Distributions for Generalized 
Additive Models for Location Scale and Shape, CRAN Package.
https://cran.r-project.org/web/packages/gamlss.dist/gamlss.dist.pdf

"""
import numpy as np
import scipy
import scipy.stats
import tensorflow as tf

__author__ = "Randal J. Barnes and Elizabeth A. Barnes"
__date__ = "14 January 2022"


SQRT_TWO = 1.4142135623730950488016887
ONE_OVER_SQRT_TWO = 0.7071067811865475244008444
TWO_PI = 6.2831853071795864769252868
SQRT_TWO_PI = 2.5066282746310005024157653
ONE_OVER_SQRT_TWO_PI = 0.3989422804014326779399461


def _jones_pewsey_P(q):
    """P_q function from page 764 of [1].

    Arguments
    ---------
    q : float, array like

    Returns
    -------
    P_q : array like of same shape as q.

    Notes
    -----
    * The formal equation is

            jp = 0.25612601391340369863537463 * (
                scipy.special.kv((q + 1) / 2, 0.25) +
                scipy.special.kv((q - 1) / 2, 0.25)
            )

        The strange constant 0.25612... is "sqrt( sqrt(e) / (8*pi) )" computed
        with a high-precision calculator.  The special function

            scipy.special.kv

        is the Modified Bessel function of the second kind: K(nu, x).

    * But, we cannot use the scipy.special.kv function during tensorflow
        training.  This code uses a 6th order polynomial approximation in
        place of the formal function.

    * This approximation is well behaved for 0 <= q <= 10. Since q = 1/tau
        or q = 2/tau in our applications, the approximation is well behaved
        for 1/10 <= tau < infty.

    """
    # A 6th order polynomial approximation of log(_jones_pewsey_P) for the
    # range 0 <= q <= 10.  Over this range, the max |error|/true < 0.0025.
    # These coefficients were computed by minimizing the maximum relative
    # error, and not by a simple least squares regression.
    coeffs = [
        9.37541380598926e-06,
        -0.000377732651131894,
        0.00642826706073389,
        -0.061281078712518,
        0.390956214318641,
        -0.0337884356755193,
        0.00248824801827172
    ]
    return tf.math.exp(tf.math.polyval(coeffs, q))


def cdf(x, mu, sigma, gamma, tau=None):
    """Cumulative distribution function (cdf).

    Parameters
    ----------
    x : float (batch size x 1) Tensor
        The values at which to compute the probability density function.

    mu : float (batch size x 1) Tensor
        The location parameter. Must be the same shape as x.

    sigma : float (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same
        shape as x.

    gamma : float (batch size x 1) Tensor
        The skewness parameter. Must be the same shape as x.

    tau : float (batch size x 1) Tensor or None
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape as x. If tau is None then the default value of tau=1 is used.

    Returns
    -------
    F : float (batch size x 1) Tensor.
        The computed cumulative probability distribution function (cdf)
        evaluated at the values of x.  F has the same shape as x.

    Notes
    -----
    * This function uses the tensorflow.math.erf function rather than the
    tensorflow_probability normal distribution functions.

    """
    y = (x - mu) / sigma    
    
    if tau is None:
        z = tf.math.sinh(tf.math.asinh(y) - gamma)
    else:
        z = tf.math.sinh(tau * tf.math.asinh(y) - gamma)

    return 0.5 * (1.0 + tf.math.erf(ONE_OVER_SQRT_TWO * z))


def log_prob(x, mu, sigma, gamma, tau=None):
    """Log-probability density function.

    Parameters
    ----------
    x : float (batch size x 1) Tensor
        The values at which to compute the probability density function.

    mu : float (batch size x 1) Tensor
        The location parameter. Must be the same shape as x.

    sigma : float (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same
        shape as x.

    gamma : float (batch size x 1) Tensor
        The skewness parameter. Must be the same shape as x.

    tau : float (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape as x. If tau is None then the default value of tau=1 is used.

    Returns
    -------
    f : float (batch size x 1) Tensor.
        The natural logarithm of the computed probability density function
        evaluated at the values of x.  f has the same shape as x.

    Notes
    -----
    * This function is included merely to emulate the tensorflow_probability
    distributions.

    """
    return tf.math.log(prob(x, mu, sigma, gamma, tau))


def mean(mu, sigma, gamma, tau=None):
    """The distribution mean.

    Arguments
    ---------
    mu : float (batch size x 1) Tensor
        The location parameter.

    sigma : float (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same
        shape as mu.

    gamma : float (batch size x 1) Tensor
        The skewness parameter. Must be the same shape as mu.

    tau : float (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape as mu. If tau is None then the default value of tau=1 is used.

    Returns
    -------
    x : float (batch size x 1) Tensor.
        The computed distribution mean values.

    Notes
    -----
    * This equation for evX can be found on page 764 of [1].

    """
    if tau is None:
        evX = tf.math.sinh(gamma) * 1.35453080648132  
    else:
        evX = tf.math.sinh(gamma / tau) * _jones_pewsey_P(1.0 / tau)

    return mu + sigma * evX


def median(mu, sigma, gamma, tau=None):
    """The distribution median.

    Arguments
    ---------
    mu : float (batch size x 1) Tensor
        The location parameter.

    sigma : float (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same
        shape as mu.

    gamma : float (batch size x 1) Tensor
        The skewness parameter. Must be the same shape as mu.

    tau : float (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape as mu. If tau is None then the default value of tau=1 is used.

    Returns
    -------
    x : float (batch size x 1) Tensor.
        The computed distribution mean values.

    Notes
    -----
    * This code uses the basic formula:

        E(a*X + b) = a*E(X) + b

    * The E(X) is computed using the moment equation given on page 764 of [1].

    """
    if tau is None:
        return mu + sigma * tf.math.sinh(gamma)
    else:
        return mu + sigma * tf.math.sinh(gamma / tau)


def prob(x, mu, sigma, gamma, tau=None):
    """Probability density function (pdf).

    Parameters
    ----------
    x : float (batch size x 1) Tensor
        The values at which to compute the probability density function.

    mu : float (batch size x 1) Tensor
        The location parameter. Must be the same shape as x.

    sigma : float (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same
        shape as x.

    gamma : float (batch size x 1) Tensor
        The skewness parameter. Must be the same shape as x.

    tau : float (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape as x. If tau is None then the default value of tau=1 is used.

    Returns
    -------
    f : float (batch size x 1) Tensor.
        The computed probability density function evaluated at the values of x.
        f has the same shape as x.

    Notes
    -----
    * This code uses the equations on page 143 of [3], and the associated
    notation.

    """
    y = (x - mu) / sigma
    
    if tau is None:
        rsqr = tf.math.square(tf.math.sinh(tf.math.asinh(y) - gamma))
        return (
            ONE_OVER_SQRT_TWO_PI
            / sigma
            * tf.math.sqrt((1 + rsqr) / (1 + tf.math.square(y)))
            * tf.math.exp(-rsqr / 2)
        )
        
    else:
        rsqr = tf.math.square(tf.math.sinh(tau * tf.math.asinh(y) - gamma))
        return (
            ONE_OVER_SQRT_TWO_PI
            * (tau / sigma)
            * tf.math.sqrt((1 + rsqr) / (1 + tf.math.square(y)))
            * tf.math.exp(-rsqr / 2)
        )


def quantile(pr, mu, sigma, gamma, tau=None):
    """Inverse cumulative distribution function.

    Arguments
    ---------
    pr : float (batch size x 1) Tensor.
        The probabilities at which to compute the values.

    mu : float (batch size x 1) Tensor
        The location parameter. Must be the same shape as pr.

    sigma : float (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same
        shape as pr.

    gamma : float (batch size x 1) Tensor
        The skewness parameter. Must be the same shape as pr.

    tau : float (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape as pr. If tau is None then the default value of tau=1 is used.

    Returns
    -------
    x : float (batch size x 1) Tensor.
        The computed values at the specified probabilities. f has the same
        shape as pr.

    """
    z = tf.math.ndtri(pr)
            
    if tau is None:
        return mu + sigma * tf.math.sinh(tf.math.asinh(z) + gamma)    
    else:
        return mu + sigma * tf.math.sinh((tf.math.asinh(z) + gamma) / tau)


def rvs(mu, sigma, gamma, tau=None, size=1):
    """Generate an array of random variates.

    Arguments
    ---------
    mu : float or double scalar
        The location parameter.

    sigma : float or double scalar
        The scale parameter. Must be strictly positive.

    gamma : float or double scalar
        The skewness parameter.

    tau : float or double scalar, or None
        The tail-weight parameter. Must be strictly positive. 
        If tau is None then the default value of tau=1 is used.

    size : int or tuple of ints, default=1.
        The number of random variates.

    Returns
    -------
    x : double ndarray of size=size
        The generated random variates.

    """
    z = scipy.stats.norm.rvs(size=size)
    
    if tau is None:
        return mu + sigma * np.sinh(np.arcsinh(z) + gamma)
    else:
        return mu + sigma * np.sinh((np.arcsinh(z) + gamma) / tau)


def stddev(mu, sigma, gamma, tau=None):
    """The distribution standard deviation.

    Arguments
    ---------
    mu : float (batch size x 1) Tensor
        The location parameter.

    sigma : float (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same
        shape as mu.

    gamma : float (batch size x 1) Tensor
        The skewness parameter. Must be the same shape as mu.

    tau : float (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape as mu. If tau is None then the default value of tau=1 is used.

    Returns
    -------
    x : float (batch size x 1) Tensor.
        The computed distribution standard deviation values.

    """
    return tf.math.sqrt(variance(mu, sigma, gamma, tau))


def variance(mu, sigma, gamma, tau=None):
    """The distribution variance.

    Arguments
    ---------
    mu : float (batch size x 1) Tensor
        The location parameter.

    sigma : float (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same
        shape as mu.

    gamma : float (batch size x 1) Tensor
        The skewness parameter. Must be the same shape as mu.

    tau : float (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape as mu. If tau is None then the default value of tau=1 is used.

    Returns
    -------
    x : float (batch size x 1) Tensor.
        The computed distribution variance values.

    Notes
    -----
    * This code uses two basic formulas:

        var(X) = E(X^2) - (E(X))^2
        var(a*X + b) = a^2 * var(X)

    * The E(X) and E(X^2) are computed using the moment equations given on
    page 764 of [1].

    """
    if tau is None:
        evX = tf.math.sinh(gamma) * 1.35453080648132
        evX2 = (tf.math.cosh(2 * gamma) * 3.0 - 1.0) / 2
    else:
        evX = tf.math.sinh(gamma / tau) * _jones_pewsey_P(1.0 / tau)
        evX2 = (tf.math.cosh(2 * gamma / tau) * _jones_pewsey_P(2.0 / tau) - 1.0) / 2
        
    return tf.math.square(sigma) * (evX2 - tf.math.square(evX))
