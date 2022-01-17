"""Plot the model.fit training history and other analysis metrics."""

import numpy as np
import matplotlib.pyplot as plt
import shash

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "17 December 2021"


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
    
def compute_iqr(uncertainty_type, onehot_val, bnn_cpd=None, x_val=None, model_shash = None):
    if(uncertainty_type=='bnn'):
        lower = np.percentile(bnn_cpd,25,axis=1)
        upper = np.percentile(bnn_cpd,75,axis=1)        
    else:
        shash_pred = model_shash.predict(x_val)
        mu = shash_pred[:,0]
        sigma = shash_pred[:,1]
        gamma = shash_pred[:,2]
        tau = np.ones(np.shape(mu))

        lower = shash.quantile(0.25, mu, sigma, gamma, tau)
        upper = shash.quantile(0.75, mu, sigma, gamma, tau)

    return lower, upper
    
def compute_interquartile_capture(uncertainty_type, onehot_val, bnn_cpd=None, x_val=None, model_shash = None):
    
    bins = np.linspace(0, 1, 11)
    bins_inc = bins[1]-bins[0]

    if(uncertainty_type=='bnn'):
        lower, upper = compute_iqr(uncertainty_type, onehot_val, bnn_cpd=bnn_cpd)
        # lower = np.percentile(bnn_cpd,25)
        # upper = np.percentile(bnn_cpd,75)        
        # iqr_capture = np.logical_and(onehot_val[:,0]>lower,onehot_val[:,0]<upper)
    else:
#         shash_pred = model_shash.predict(x_val)
#         mu = shash_pred[:,0]
#         sigma = shash_pred[:,1]
#         gamma = shash_pred[:,2]
#         tau = np.ones(np.shape(mu))

#         lower = shash.quantile(0.25, mu, sigma, gamma, tau)
#         upper = shash.quantile(0.75, mu, sigma, gamma, tau)
        lower, upper = compute_iqr(uncertainty_type, onehot_val, x_val=x_val, model_shash=model_shash)
    iqr_capture = np.logical_and(onehot_val[:,0]>lower,onehot_val[:,0]<upper)

    return np.sum(iqr_capture.astype(int))/np.shape(iqr_capture)[0]
    
    
    
def compute_pit(uncertainty_type, onehot_val, bnn_cpd=None, x_val=None, model_shash = None):
    
    bins = np.linspace(0, 1, 11)
    bins_inc = bins[1]-bins[0]

    if(uncertainty_type=='bnn'):
        bnn_cdf = np.zeros((np.shape(bnn_cpd)[0],)) 
        for sample in np.arange(0,np.shape(bnn_cpd)[0]):
            i = np.where(onehot_val[sample,0]<bnn_cpd[sample,:])[0]
            bnn_cdf[sample] = len(i)/np.shape(bnn_cpd)[1]

        pit_hist = np.histogram(bnn_cdf,
                                bins,
                                weights=np.ones_like(bnn_cdf)/float(len(bnn_cdf)),
                               )
    else:
        shash_pred = model_shash.predict(x_val)
        mu = shash_pred[:,0]
        sigma = shash_pred[:,1]
        gamma = shash_pred[:,2]
        tau = np.ones(np.shape(mu))
        F = shash.cdf(onehot_val[:,0], mu, sigma, gamma, tau)
        pit_hist = np.histogram(F,
                                  bins,
                                  weights=np.ones_like(F)/float(len(F)),
                                 )
    # pit metric from Bourdin et al. (2014) and Nipen and Stull (2011)
    # compute expected deviation of PIT for a perfect forecast
    B   = len(pit_hist[0])
    D   = np.sqrt(1/B * np.sum( (pit_hist[0] - 1/B)**2 ))
    EDp = np.sqrt( (1.-1/B) / (onehot_val.shape[0]*B) )

    return bins, pit_hist, D, EDp


def compute_nll(uncertainty_type, onehot_val, bnn_cpd=None, model_shash=None, x_val=None):
    
    if(uncertainty_type=='bnn'):
        # bnn NLL
        bins_inc = 2.5
        bins = np.arange(-100,110,bins_inc)
        nloglike = np.zeros((np.shape(bnn_cpd)[0],))
        for sample in np.arange(0,np.shape(bnn_cpd)[0]):
            hist_bnn = np.histogram(bnn_cpd[sample,:],
                                    bins
                                   )
            i = np.argmin(np.abs( (bins[:-1]+bins_inc/2) - onehot_val[sample,0]))
            if(hist_bnn[0][i]==0):
                print('---sample ' + str(sample) + ' negative-log-likelihood being set to 10.0 due to log(0) issues')        
                nloglike[sample] = 10. #np.nan
            else:
                nloglike[sample] = -np.log(hist_bnn[0][i]/(bins_inc*np.sum(hist_bnn[0])))
    else:
        # shash NLL 
        shash_pred = model_shash.predict(x_val)
        mu = shash_pred[:,0]
        sigma = shash_pred[:,1]
        gamma = shash_pred[:,2]
        tau = np.ones(np.shape(mu))
        nloglike = -shash.log_prob(onehot_val[:,0], mu, sigma, gamma, tau)        
        
    return nloglike

def compute_errors(onehot_val, pred_mean, pred_median, pred_mode):

    mean_error = np.mean(np.abs(pred_mean - onehot_val[:,0]))
    median_error = np.mean(np.abs(pred_median - onehot_val[:,0]))
    mode_error = np.mean(np.abs(pred_mode - onehot_val[:,0]))
    
    return mean_error, median_error, mode_error

def compute_iqr_error_corr(uncertainty_type, onehot_val, bnn_cpd=None, pred_median=None, x_val=None, model_shash = None):

    from scipy import stats
    
    # compute IQR
    bins = np.linspace(0, 1, 11)
    bins_inc = bins[1]-bins[0]

    if(uncertainty_type=='bnn'):
        lower, upper = compute_iqr(uncertainty_type, onehot_val, bnn_cpd=bnn_cpd)
    else:
        lower, upper = compute_iqr(uncertainty_type, onehot_val, x_val=x_val, model_shash=model_shash)
    iqr   = upper - lower
    
    # compute median_errors
    median_errors = np.abs(pred_median - onehot_val[:,0])
    
    # compute correlation between median error and IQR
    iqr_error_spearman = stats.spearmanr(iqr,median_errors)
    iqr_error_pearson  = stats.pearsonr(iqr,median_errors)

    return iqr_error_spearman, iqr_error_pearson