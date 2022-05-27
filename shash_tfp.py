import tensorflow as tf
import tensorflow_probability as tfp

import logging
logging.getLogger().setLevel(logging.ERROR)

__author__ = "Randal J. Barnes and Elizabeth A. Barnes"
__date__ = "27 May 2022"


class Shash(tfp.distributions.SinhArcsinh):
    """The sinh-arcsinh-normal distribution using TensorFlow Probability.

    Shash is a wrapper subclass of the tfp.distributions.SinhArcsinh class.
    This wrapper fixes the engendering distribution to be Normal, and adds
    methods for various distributional statistics that are not currently
    implemented in TensorFlow.

    Attributes
    ----------
        None (beyond those of the baseclass)

    Public Methods
    --------------
    mean()
        distribution mean.

    median()
        distribution median.

    skew()
        distribution skewness.

    stddev()
        distribution standard deviation.

    variance()
        distribution variance.

    Notes
    -----
    * The sinh-arcsinh normal distribution was defined in [1]. A more
    accessible presentation is given in [2].

    * The formulation, and notation, used in this code differs from that
    presented in [1] and [2].  The notation and formulation are taken from
    TensorFlow [3].

    * The relationships between the notation and formulation for
    Jones and Pewsey [1, 2] versus Tensorflow [3] are:

        Jones and Pewsey    TensorFlow Probability
        ----------------    ----------------------
        xi                  loc
        eta                 scale * 2/sinh(asinh(2) * tailweight)
        epsilon             skewness
        delta               1/tailweight

    * The mean, skewness, stddev, and variance defined by Shash MAY NOT
    be used in the loss function because the computations require the
    the Bessel Function of the sencod kind.  The TensorFlow implementation,
    tfp.math.bessel_kve, gives the following warning.

        "Warning: Gradients with respect to the first parameter v
        are currently not defined."

    References
    ----------
    [1] Jones, M. C. & Pewsey, A., Sinh-arcsinh distributions, Biometrika,
    Oxford University Press, 2009, 96, 761-780. DOI: 10.1093/biomet/asp053.

    [2] Jones, C. & Pewsey, A., The sinh-arcsinh normal distribution,
    Significance, Wiley, 2019, 16, 6-7. DOI: 10.1111/j.1740-9713.2019.01245.x.

    [3] tfp.distributions.SinhArcsinh. url: https://www.tensorflow.org/
    probability/api_docs/python/tfp/distributions/SinhArcsinh.

    """

    def __init__(self, loc, scale, skewness=None, tailweight=None):
        """

        Arguments
        ---------
        loc	        floating point Tensor.
        scale	    Tensor of same dtype and shape as loc.
        skewness	Tensor of same dtype and shape as loc, or None.
        tailweight	Tensor of same dtype and shape as loc, or None.

        Notes
        -----
        * If skewness is None, it defaults to 0.0 (i.e. shash2).
        * If tailweight is None, it defaults to 1.0 (i.e. shash3).

        """
        super(__class__, self).__init__(loc, scale, skewness, tailweight, name="Shash")

    @staticmethod
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
        * The formal equation is converted from kv to kve as follows:

            P_q
            = exp(1/4) * (kv((q + 1) / 2, 0.25) + kv((q - 1) / 2, 0.25)) / sqrt(8*pi)
            = (exp(1/4) * kv((q + 1) / 2, 0.25) + exp(1/4) * kv((q - 1) / 2, 0.25)) / sqrt(8*pi)
            = (kve((q + 1) / 2, 0.25) + kve((q - 1) / 2, 0.25)) / sqrt(8*pi)

        where
        -- kv is the Modified Bessel function of the second kind: K(nu, x)
        -- kve is the exponentially scaled modified Bessel function of the
           second kind: K(nu, x) * exp(abs(x)).

        * The constant "0.199471140200716339" is one over the square root
        of 8pi, computed using a high-precision calculator.

        """
        return 0.199471140200716339 * (
            tfp.math.bessel_kve((q + 1) / 2, 0.25)
            + tfp.math.bessel_kve((q - 1) / 2, 0.25)
        )

    def _convert_tfp_to_jones_and_pewsey(self):
        """Convert paramters from TensorFlow to Jones and Pewsey.

        Convert the internal parameters using the Tensorflow formulation [3]
        to the parameters using the Jones and Pewsey formulation [1, 2].

        This conversion accounts for the shash3 and shash2 cases.  If the
        tailweight parameter is None (shash3), then delta = 1.0. If the
        skewness parameter is None (shash2), then epsilon = 0.0.

        Returns
        -------
        xi Tensor of same dtype and shape as loc.
            Jones and Pewsey location parameter.

        eta Tensor of same dtype and shape as loc.
            Jones and Pewsey scale parameter.

        epsilon	Tensor of same dtype and shape as loc.
            Jones and Pewsey skewness parameter.

        delta Tensor of same dtype and shape as loc.
            Jones and Pewsey tailweight parameter.

        Notes
        -----
        * The relationships between the notation and formulation for
        Jones and Pewsey [1, 2] versus Tensorflow [3] are:

            Jones and Pewsey    TensorFlow Probability
            ----------------    ----------------------
            xi                  loc
            eta                 scale * 2/sinh(asinh(2) * tailweight)
            epsilon             skewness
            delta               1/tailweight

        * This conversion of the parameters is neccesary to use the detailed
        formulas for the moments given in Jones and Pewsey [1].

        We could have converted the formulas to the tfp notation, but such a
        reformulation does not save significant computational time. For
        example, the computation of the scaling factor in eta is always
        required given the tfp tailweight. Using the Jones and Pewsey [1]
        parameters allows for a direct implementation of published formulas.

        Furthermore, the creation of this conversion function allows for the
        seamless, and somewhat invisible, implementation of shash2 ans shash3.

        """
        xi = self.parameters["loc"]

        if self.parameters["skewness"] is None:
            epsilon = tf.zeros_like(self.parameters["loc"])
        else:
            epsilon = self.parameters["skewness"]

        if self.parameters["tailweight"] is None:
            eta = self.parameters["scale"]
            delta = tf.ones_like(self.parameters["loc"])
        else:
            eta = (
                self.parameters["scale"]
                * 2.0
                / tf.math.sinh(tf.math.asinh(2.0) * self.parameters["tailweight"])
            )
            delta = 1.0 / self.parameters["tailweight"]

        return xi, eta, epsilon, delta

    def mean(self):
        """The distribution mean.

        Returns
        -------
        x : Tensor of same dtype and shape as loc specified at initialization.
            The computed distribution mean values.

        Notes
        -----
        * The equation for E(X) can be found on page 764 of [1].

        """
        xi, eta, epsilon, delta = self._convert_tfp_to_jones_and_pewsey()

        evX = tf.math.sinh(epsilon / delta) * self._jones_pewsey_P(1.0 / delta)

        return xi + eta * evX

    def median(self):
        """The distribution median.

        Returns
        -------
        x : Tensor of same dtype and shape as loc specified at initialization.
            The computed distribution mean values.

        Notes
        -----
        * This direct calculation of the median is more than 16 times
        faster than using quantile(0.5).

        """
        xi, eta, epsilon, delta = self._convert_tfp_to_jones_and_pewsey()

        return xi + eta * tf.math.sinh(epsilon / delta)

    def skew(self):
        """The distribution skewness. Named as such to not overwrite the "skewness" parameter.

        Returns
        -------
        x : Tensor of same dtype and shape as loc specified at initialization.
            The computed distribution skewness values.

        Notes
        -----
        * The E(X), E(X^2), and E(X^3) are computed using the moment equations
        given on page 764 of [1].

        """
        xi, eta, epsilon, delta = self._convert_tfp_to_jones_and_pewsey()

        evX = tf.math.sinh(epsilon / delta) * self._jones_pewsey_P(1.0 / delta)
        evX2 = (
            tf.math.cosh(2.0 * epsilon / delta) * self._jones_pewsey_P(2.0 / delta)
            - 1.0
        ) / 2.0
        evX3 = (
            tf.math.sinh(3.0 * epsilon / delta) * self._jones_pewsey_P(3.0 / delta)
            - 3.0 * tf.math.sinh(epsilon / delta) * self._jones_pewsey_P(1.0 / delta)
        ) / 4.0

        evY3 = (
            tf.pow(xi, 3)
            + 3.0 * tf.math.square(xi) * eta * evX
            + 3.0 * xi * tf.math.square(eta) * evX2
            + tf.math.pow(eta, 3) * evX3
        )
        mu = xi + eta * evX
        sigma = eta * tf.math.sqrt(evX2 - evX * evX)

        return (
            evY3 - 3.0 * mu * tf.math.square(sigma) - tf.math.pow(mu, 3)
        ) / tf.math.pow(sigma, 3)

    def stddev(self):
        """The distribution standard deviation.

        Returns
        -------
        x : Tensor of same dtype and shape as loc specified at initialization.
            The computed distribution standard deviation values.

        """
        return tf.math.sqrt(self.variance())

    def variance(self):
        """The distribution variance.

        Returns
        -------
        x : Tensor of same dtype and shape as loc specified at initialization.
            The computed distribution variance values.

        Notes
        -----
        * This code uses two basic formulas:

            var(X) = E(X^2) - (E(X))^2
            var(a*X + b) = a^2 * var(X)

        * The E(X) and E(X^2) are computed using the moment equations given on
        page 764 of [1].

        """
        xi, eta, epsilon, delta = self._convert_tfp_to_jones_and_pewsey()

        evX = tf.math.sinh(epsilon / delta) * self._jones_pewsey_P(1.0 / delta)
        evX2 = (
            tf.math.cosh(2 * epsilon / delta) * self._jones_pewsey_P(2.0 / delta) - 1.0
        ) / 2

        return tf.math.square(eta) * (evX2 - tf.math.square(evX))
