"""Plot the model.fit training history."""

import numpy as np
import matplotlib.pyplot as plt

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "30 October 2021"


def plot_history(history, model_name):
    """Plot the model.fit training history and save the resulting figure.

    Creates a 2-by-2 block of subplots.  The four plots are:
        1 -- training and validations loss history.
        2 -- training and validations customMAE history.
        3 -- training and validations InterquartileCapture history.
        4 -- training and validations SignTest history.

    Arguments
    ---------
    history : tf.keras.callbacks.History
        The history must have at least the following eight items
        in the history.history.keys()
            "loss",
            "val_loss",
            "custom_mae",
            "val_custom_mae",
            "interquartile_capture",
            "val_interquartile_capture",
            "sign_test",
            "val_sign_test"

    model_name : str
        The resulting figure is saved to:
            "figures/model_diagnostics/" + model_name + ".png"

    Returns
    -------
    None

    """
    TRAIN_COLOR = "#7570b3"
    VALID_COLOR = "#e7298a"
    FIGSIZE = (14, 10)
    FONTSIZE = 12
    DPIFIG = 300.0

    best_epoch = np.argmin(history.history["val_loss"])

    plt.figure(figsize=FIGSIZE)

    # Plot the training and validations loss history.
    plt.subplot(2, 2, 1)
    plt.plot(
        history.history["loss"],
        "o",
        color=TRAIN_COLOR,
        markersize=3,
        label="train",
    )

    plt.plot(
        history.history["val_loss"],
        "o",
        color=VALID_COLOR,
        markersize=3,
        label="valid",
    )

    plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
    plt.title("Log-likelihood Loss Function")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.legend(frameon=True, fontsize=FONTSIZE)

    # Plot the training and validations customMAE history.
    try:
        plt.subplot(2, 2, 2)
        plt.plot(
            history.history["custom_mae"],
            "o",
            color=TRAIN_COLOR,
            markersize=3,
            label="train",
        )
        plt.plot(
            history.history["val_custom_mae"],
            "o",
            color=VALID_COLOR,
            markersize=3,
            label="valid",
        )

        plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
        plt.title("Mean |true - median|")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend(frameon=True, fontsize=FONTSIZE)
    except:
        print('no mae metric, skipping plot')
        
    # Plot the training and validations InterquartileCapture history.
    try:
        plt.subplot(2, 2, 3)
        plt.plot(
            history.history["interquartile_capture"],
            "o",
            color=TRAIN_COLOR,
            markersize=3,
            label="train",
        )
        plt.plot(
            history.history["val_interquartile_capture"],
            "o",
            color=VALID_COLOR,
            markersize=3,
            label="valid",
        )

        plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
        plt.title("Fraction Between 25 and 75 Percentile")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend(frameon=True, fontsize=FONTSIZE)
    except:
        print('no interquartile_capture, skipping plot')

    # Plot the training and validations SignTest history.
    try:
        plt.subplot(2, 2, 4)
        plt.plot(
            history.history["sign_test"],
            "o",
            color=TRAIN_COLOR,
            markersize=3,
            label="train",
        )
        plt.plot(
            history.history["val_sign_test"],
            "o",
            color=VALID_COLOR,
            markersize=3,
            label="valid",
        )

        plt.axvline(x=best_epoch, linestyle="--", color="tab:gray")
        plt.title("Fraction Above the Median")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend(frameon=True, fontsize=FONTSIZE)
    except:
        print('no sign-test, skipping plot')

    # Draw and save the plot.
    plt.tight_layout()
    plt.savefig("figures/model_diagnostics/" + model_name + ".png", dpi=DPIFIG)
    plt.show()
