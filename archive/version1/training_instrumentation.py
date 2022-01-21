"""Custom real-time training instrumentation."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
import shash

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "30 October 2021"


class TrainingInstrumentation(tf.keras.callbacks.Callback):
    """Plot real-time training instrumentation panel.

    If x_data and onehot_data are not given, the instrumentation panel
    includes only the real-time plot of the training and validation loss.

    If the x_data and onehot_data are given, the instrumentation panel also
    includes the PIT histogram plot, and histogram plots for each of the
    local conditional distribution parameters, updated in real time.

    Parameters
    ----------
    x_data : tensor, default=None
        The x_train (or x_valid) tensor.  If either x_data or onehot_data
        is specifed, then both must be specified and they must have the
        same number of rows.

    onehot_data : tensor, default=None
        The onehot_train (or onehot_valid) tensor. If either x_data or
        onehot_data is specifed, then both must be specified and they must
        have the same number of rows.

    figsize: (float, float), default=(13, 7)
        Size of the instrumentation panel.

    interval: int, default=1
        Number of epochs (steps) between refreshing the instruments.  By
        default, interval=1, and the intruments are updated ever epoch.

    Usage
    -----
    * Include TrainingInstrumentation() as a callback in model.fit; e.g.

        training_callback = TrainingInstrumentation(
            x_train_std, onehot_train, interval=10
        )
        ...
        history = model.fit(
            ...
            callbacks=[training_callback],
        )

    Notes
    -----
    * This Class is explcitly designed for the SHASH distribution, with
        parameter names 'mu', 'sigma', 'gamma', and 'tau'.

    """

    def __init__(
        self,
        x_data=None,
        onehot_data=None,
        figsize=(13, 7),
        interval=1,
    ):
        super().__init__()
        self.x_data = x_data
        self.onehot_data = onehot_data
        self.figsize = figsize
        self.interval = interval

    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))

        if epoch % self.interval == 0:
            clear_output(wait=True)
            plt.figure(figsize=self.figsize)

            best_epoch = np.argmin(self.val_loss)
            plt.subplot(3, 2, 1)
            plt.plot(self.loss, "o", color="#7570b3", label="train", markersize=2)
            plt.plot(self.val_loss, "o", color="#e7298a", label="valid", markersize=2)
            plt.axvline(x=best_epoch, linestyle="--", color="gray")
            plt.title(f"Loss After {epoch} Epochs")
            plt.grid(True)
            plt.legend(
                [
                    f"train = {logs.get('loss'):.3f}",
                    f"valid = {logs.get('val_loss'):.3f}",
                ]
            )

            if (self.x_data is not None) and (self.onehot_data is not None):
                preds = self.model.predict(self.x_data)

                if preds.shape[1] >= 1:
                    mu = preds[:, 0]
                    plt.subplot(3, 2, 3)
                    plt.hist(mu, bins=30, color="#7fc97f", edgecolor="k")
                    plt.legend(["mu"])

                if preds.shape[1] >= 2:
                    # sigma = tf.math.exp(preds[:, 1])
                    sigma = preds[:, 1]
                    plt.subplot(3, 2, 4)
                    plt.hist(sigma, bins=30, color="#beaed4", edgecolor="k")
                    plt.legend(["sigma"])
                else:
                    sigma = tf.zeros_like(mu)

                if preds.shape[1] >= 3:
                    gamma = preds[:, 2]
                    plt.subplot(3, 2, 5)
                    plt.hist(gamma, bins=30, color="#fdc086", edgecolor="k")
                    plt.legend(["gamma"])
                else:
                    gamma = tf.zeros_like(mu)

                if preds.shape[1] >= 4:
                    # tau = tf.math.exp(preds[:, 3])
                    tau = preds[:, 3]
                    plt.subplot(3, 2, 6)
                    plt.hist(tau, bins=30, color="#ffff99", edgecolor="k")
                    plt.legend(["tau"])
                else:
                    tau = tf.ones_like(mu)

                F = shash.cdf(self.onehot_data[:, 0], mu, sigma, gamma, tau)
                plt.subplot(3, 2, 2)
                plt.hist(
                    F.numpy(),
                    bins=np.linspace(0, 1, 21),
                    color="#386cb0",
                    edgecolor="k",
                )
                plt.legend(["PIT"])
                plt.axhline(y=F.shape[0] / 20, color="b", linestyle="--")

            plt.show()
