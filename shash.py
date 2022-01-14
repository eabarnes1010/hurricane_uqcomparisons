"""sinh-arcsinh normal distribution w/o using tensorflow_probability.

Functions
---------
cdf(x, mu, sigma, gamma, tau)
    cumulative distribution function (cdf).

log_prob(x, mu, sigma, gamma, tau)
    log of the probability density function.

mean(mu, sigma, gamma, tau)
    distribution mean.

median(mu, sigma, gamma, tau)
    distribution median.

prob(x, mu, sigma, gamma, tau)
    probability density function (pdf).

quantile(pr, mu, sigma, gamma, tau)
    inverse cumulative distribution function.

rvs(mu, sigma, gamma, tau, size=1)
    generate random variates.

stddev(mu, sigma, gamma, tau)
    distribution standard deviation.

variance(mu, sigma, gamma, tau)
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
    q : float or double, array like

    Returns
    -------
    P_q : array like of same shape as q.

    Notes
    -----
    * The formal equation is

            Pq = sqrt( sqrt(e) / (8*pi) ) * (
                besselk(nu = (q+1)/2, z = 0.25) +
                besselk(nu = (q-1)/2, z = 0.25)
            )

        where "besselk" is the modified Bessel function of the second kind.

        The besselk function is available as scipy.special.kv, but we cannot
        use this function during tensorflow training as it is not tensorflow-
        aware.

        This code uses a 6th order polynomial approximation for the log(Pq)
        in place of the formal function. This approximation is well behaved
        for 0 <= q <= 10. Over this range, the max |error|/true < 6.4e-06.

        Since q = 1/tau, or q = 2/tau, in our applications, the approximation
        is well behaved for 1/10 <= tau < infty.

    """
    # These coefficients were computed by minimizing the maximum relative
    # error over the range 0 <= q <= 10.
    coeffs = [
        8.20354501338284e-06,
        -0.000359600349085236,
        0.00646940923214159,
        -0.0628533171453501,
        0.395372203069515,
        -0.0283864669666028,
        0.0257772853487392,
    ]
    return tf.math.exp(tf.math.polyval(coeffs, tf.cast(q, dtype=tf.float32)))


def cdf(x, mu, sigma, gamma, tau):
    """Cumulative distribution function (cdf).

    Parameters
    ----------
    x : float or double (batch size x 1) Tensor
        The values at which to compute the probability density function.

    mu : float or double (batch size x 1) Tensor
        The location parameter. Must be the same shape and dtype as x.

    sigma : float or double (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same shape
        and dtype as x.

    gamma : float or double (batch size x 1) Tensor
        The skewness parameter. Must be the same shape and dtype as x.

    tau : float or double (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape and dtype as x.

    Returns
    -------
    F : float or double (batch size x 1) Tensor.
        The computed cumulative probability distribution function (cdf)
        evaluated at the values of x.  F has the same shape and dtype as x.

    Notes
    -----
    * This function uses the tensorflow.math.erf function rather than the
    tensorflow_probability normal distribution functions.

    """
    y = tf.cast((x - mu) / sigma, tf.float32)
    z = tf.math.sinh(tau * tf.math.asinh(y) - gamma)
    return 0.5 * (1.0 + tf.math.erf(ONE_OVER_SQRT_TWO * z))


def log_prob(x, mu, sigma, gamma, tau):
    """Log-probability density function.

    Parameters
    ----------
    x : float or double (batch size x 1) Tensor
        The values at which to compute the probability density function.

    mu : float or double (batch size x 1) Tensor
        The location parameter. Must be the same shape and dtype as x.

    sigma : float or double (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same shape
        and dtype as x.

    gamma : float or double (batch size x 1) Tensor
        The skewness parameter. Must be the same shape and dtype as x.

    tau : float or double (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape and dtype as x.

    Returns
    -------
    f : float or double (batch size x 1) Tensor.
        The natural logarithm of the computed probability density function
        evaluated at the values of x.  f has the same shape and dtype as x.

    Notes
    -----
    * This function is included merely to emulate the tensorflow_probability
    distributions.

    """
    return tf.math.log(prob(x, mu, sigma, gamma, tau))


def mean(mu, sigma, gamma, tau):
    """The distribution mean.

    Arguments
    ---------
    mu : float or double (batch size x 1) Tensor
        The location parameter.

    sigma : float or double (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same shape
        and dtype as mu.

    gamma : float or double (batch size x 1) Tensor
        The skewness parameter. Must be the same shape and dtype as mu.

    tau : float or double (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape and dtype as mu.

    Returns
    -------
    x : float or double (batch size x 1) Tensor.
        The computed distribution mean values.

    Notes
    -----
    * This equation for evX can be found on page 764 of [1].

    """
    evX = tf.math.sinh(gamma / tau) * _jones_pewsey_P(1.0 / tau)
    return mu + sigma * evX


def median(mu, sigma, gamma, tau):
    """The distribution median.

    Arguments
    ---------
    mu : float or double (batch size x 1) Tensor
        The location parameter.

    sigma : float or double (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same shape
        and dtype as mu.

    gamma : float or double (batch size x 1) Tensor
        The skewness parameter. Must be the same shape and dtype as mu.

    tau : float or double (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape and dtype as mu.

    Returns
    -------
    x : float or double (batch size x 1) Tensor.
        The computed distribution mean values.

    Notes
    -----
    * This code uses the basic formula:

        E(a*X + b) = a*E(X) + b

    * The E(X) is computed using the moment equation given on page 764 of [1].

    """
    return mu + sigma * tf.math.sinh(gamma / tau)


def prob(x, mu, sigma, gamma, tau):
    """Probability density function (pdf).

    Parameters
    ----------
    x : float or double (batch size x 1) Tensor
        The values at which to compute the probability density function.

    mu : float or double (batch size x 1) Tensor
        The location parameter. Must be the same shape and dtype as x.

    sigma : float or double (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same shape
        and dtype as x.

    gamma : float or double (batch size x 1) Tensor
        The skewness parameter. Must be the same shape and dtype as x.

    tau : float or double (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape and dtype as x.

    Returns
    -------
    f : float or double (batch size x 1) Tensor.
        The computed probability density function evaluated at the values of x.
        f has the same shape and dtype as x.

    Notes
    -----
    * This code uses the equations on page 143 of [3], and the associated
    notation.

    """
    y = (x - mu) / sigma
    rsqr = tf.math.square(tf.math.sinh(tau * tf.math.asinh(y) - gamma))
    return (
        ONE_OVER_SQRT_TWO_PI
        * (tau / sigma)
        * tf.math.sqrt((1 + rsqr) / (1 + tf.math.square(y)))
        * tf.math.exp(-rsqr / 2)
    )


def quantile(pr, mu, sigma, gamma, tau):
    """Inverse cumulative distribution function.

    Arguments
    ---------
    pr : float or double (batch size x 1) Tensor.
        The probabilities at which to compute the values.

    mu : float or double (batch size x 1) Tensor
        The location parameter. Must be the same shape and dtype as pr.

    sigma : float or double (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same shape
        and dtype as pr.

    gamma : float or double (batch size x 1) Tensor
        The skewness parameter. Must be the same shape and dtype as pr.

    tau : float or double (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape and dtype as pr.

    Returns
    -------
    x : float or double (batch size x 1) Tensor.
        The computed values at the specified probabilities. f has the same
        shape and dtype as pr.

    """
    z = tf.math.ndtri(pr)
    return mu + sigma * tf.math.sinh((tf.math.asinh(z) + gamma) / tau)


def rvs(mu, sigma, gamma, tau, size=1):
    """Generate an array of random variates.

    Arguments
    ---------
    mu : float or double scalar
        The location parameter.

    sigma : float or double scalar
        The scale parameter. Must be strictly positive.

    gamma : float or double scalar
        The skewness parameter.

    tau : float or double scalar
        The tail-weight parameter. Must be strictly positive.

    size : int or tuple of ints, default=1.
        The number of random variates.

    Returns
    -------
    x : double ndarray of size=size
        The generated random variates.

    Notes
    -----
    * Unlike the other functions in this module, this function only accepts
    scalar arguments, not tensors.

    """
    z = scipy.stats.norm.rvs(size=size, random_state=np.random.RandomState(seed=888))
    return mu + sigma * np.sinh((np.arcsinh(z) + gamma) / tau)


def stddev(mu, sigma, gamma, tau):
    """The distribution standard deviation.

    Arguments
    ---------
    mu : float or double (batch size x 1) Tensor
        The location parameter.

    sigma : float or double (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same shape
        and dtype as mu.

    gamma : float or double (batch size x 1) Tensor
        The skewness parameter. Must be the same shape and dtype as mu.

    tau : float or double (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape and dtype as mu.

    Returns
    -------
    x : float or double (batch size x 1) Tensor.
        The computed distribution mean values.

    """
    return tf.math.sqrt(variance(mu, sigma, gamma, tau))


def variance(mu, sigma, gamma, tau):
    """The distribution variance.

    Arguments
    ---------
    mu : float or double (batch size x 1) Tensor
        The location parameter.

    sigma : float or double (batch size x 1) Tensor
        The scale parameter. Must be strictly positive. Must be the same shape
        and dtype as mu.

    gamma : float or double (batch size x 1) Tensor
        The skewness parameter. Must be the same shape and dtype as mu.

    tau : float or double (batch size x 1) Tensor
        The tail-weight parameter. Must be strictly positive. Must be the same
        shape and dtype as mu.

    Returns
    -------
    x : float or double (batch size x 1) Tensor.
        The computed distribution mean values.

    Notes
    -----
    * This code uses two basic formulas:

        var(X) = E(X^2) - (E(X))^2
        var(a*X + b) = a^2 * var(X)

    * The E(X) and E(X^2) are computed using the moment equations given on
    page 764 of [1].

    """
    evX = tf.math.sinh(gamma / tau) * _jones_pewsey_P(1.0 / tau)
    evX2 = (tf.math.cosh(2 * gamma / tau) * _jones_pewsey_P(2.0 / tau) - 1.0) / 2

    return tf.math.square(sigma) * (evX2 - tf.math.square(evX))
