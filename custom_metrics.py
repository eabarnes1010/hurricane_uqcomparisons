"""Define custom metrics.

Classes
-------
CustomMAE(tf.keras.metrics.Metric)
    Compute the mean absolute error.

InterquartileCapture(tf.keras.metrics.Metric)
    Compute the fraction of true values between the 25 and 75 percentiles.

Usage
-----
* Include CustomMAE() as a metric in model.compile; e.g.

    model.compile(
        ...
        metrics=[CustomMAE()],
    )

"""
import tensorflow as tf
import shash_tfp

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "30 October 2021"


class CustomMAE(tf.keras.metrics.Metric):
    """Compute the prediction mean absolute error.

    The "predicted value" is the median of the conditional distribution.

    Notes
    -----
    * The computation is done by maintaining running sums of total predictions
        and correct predictions made across all batches in an epoch. The
        running sums are reset at the end of each epoch.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.error = self.add_weight("error", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, pred, sample_weight=None):
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

        dist = shash_tfp.Shash(mu, sigma, gamma, tau)
        predictions = dist.median()
        # predictions = dist.mean()

        error = tf.math.abs(y_true[:, 0] - predictions)
        batch_error = tf.reduce_sum(error)
        batch_total = tf.math.count_nonzero(error)

        self.error.assign_add(tf.cast(batch_error, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.error / self.total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


class InterquartileCapture(tf.keras.metrics.Metric):
    """Compute the fraction of true values between the 25 and 75 percentiles.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight("count", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, pred, sample_weight=None):
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

        dist = shash_tfp.Shash(mu, sigma, gamma, tau)
        lower = dist.quantile(0.25)
        upper = dist.quantile(0.75)

        batch_count = tf.reduce_sum(
            tf.cast(
                tf.math.logical_and(
                    tf.math.greater(y_true[:, 0], lower),
                    tf.math.less(y_true[:, 0], upper)
                ),
                tf.float32
            )

        )
        batch_total = len(y_true[:, 0])

        self.count.assign_add(tf.cast(batch_count, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.count / self.total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


class SignTest(tf.keras.metrics.Metric):
    """Compute the fraction of true values above the median.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight("count", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, pred, sample_weight=None):
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

        dist = shash_tfp.Shash(mu, sigma, gamma, tau)
        median = dist.median()

        batch_count = tf.reduce_sum(
            tf.cast(tf.math.greater(y_true[:, 0], median), tf.float32)
        )
        batch_total = len(y_true[:, 0])

        self.count.assign_add(tf.cast(batch_count, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.count / self.total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

