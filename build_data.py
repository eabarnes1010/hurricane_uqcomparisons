"""Build the split and scaled training and validation hurricane data arrays.

Functions
---------
build_hurricane_data(data_path, settings, verbose=0)

"""
import pprint

import numpy as np
import pandas as pd

import toolbox


__author__ = "Elizabeth A. Barnes and Randal J Barnes"
__version__ = "17 December 2021"


def build_hurricane_data(data_path, settings, verbose=0):
    """Build the training and validation tensors.

    The settings['target'] specifies which data set to build.There are five
    different possible targets: intensity, logitude, latitude, radial, and
    angle.

    Arguments
    ---------
    data_path : str
        The input filepath, not including the file name.

    settings : dict
        The parameters defining the current experiment.

    verbose : int
        0 -> silent
        1 -> description only
        2 -> description and y statistics

    Returns
    -------
    x_train : numpy.ndarray
        The training split of the x data.
        shape = [n_train, n_features].

    onehot_train : numpy.ndarray
        The training split of the scaled y data is in the first column.
        The remaining columns are filled with zeros. The number of columns
        equal the number of distribution parameters.
        shape = [n_train, n_parameters].

    x_val : numpy.ndarray
        The validation split of the x data.
        shape = [n_val, n_features].

    onehot_val : numpy.ndarray
        The validation split of the scaled y data is in the first column.
        The remaining columns are filled with zeros. The number of columns
        equal the number of distribution parameters.
        shape = [n_val, n_parameters].

    data_summary : dict
        A descriptive dictionary of the data.

    df_val : pandas dataframe
        A pandas dataframe containing validation records.  The dataframe
        contains all columns from the original file.
        However, the dataframe contains only rows from the validation data
        set that satisfy the specified basin and leadtime requirements, and
        were not eliminated due to missing values.

        The dataframe has the shuffled order of the rows.  In particular,
        the rows of df_val align with the rows of x_val and onehot_val.

    Notes
    -----
    * No scaling or normalization is applied during data preparation.

    """
    # Setup for the selected target.
    if settings["target"] == "intensity":
        x_names = [
            "NCI",
            "DSDV", "LGDV", "HWDV", "AVDV",
            "VMXC", "DV12", "SLAT", "SSTN", "SHDC", "DTL",
        ]
        y_name = ["OBDV"]
        missing = None

    elif settings["target"] == "longitude":
        # x_names = ["AVDX", "EMDX", "EGDX", "HWDX", "LONC"]
        x_names = [
            "NCT",
            "AVDX", "EMDX", "EGDX", "HWDX",
            "LONC", "LATC",
            "VMXC", "DV12", "SHDC", "SSTN", "DTL",
        ]
        y_name = ["OBDX"]
        missing = -9999

    elif settings["target"] == "latitude":
        # x_names = ["AVDY", "EMDY", "EGDY", "HWDY", "LATC"]
        x_names = [
            "NCT", "AVDY", "EMDY", "EGDY", "HWDY",
            "LONC", "LATC",
            "VMXC", "DV12", "SHDC", "SSTN", "DTL",
        ]
        y_name = ["OBDY"]
        missing = -9999

    elif settings["target"] == "radial":
        x_names = [
            "NCT",
            "AVDX", "EMDX", "EGDX", "HWDX",
            "AVDY", "EMDY", "EGDY", "HWDY",
            "LONC", "LATC",
            "VMXC", "DV12", "SHDC", "SSTN", "DTL",
            "DSDV", "LGDV", "HWDV", "AVDV",
        ]
        y_name = ["OBDR"]
        missing = -9999

    else:
        raise NotImplementedError

    # Get the data from the specified file and filter out the unwanted rows.
    datafile_path = data_path + settings["filename"]
    df_raw = pd.read_table(datafile_path, sep="\s+")

    df = df_raw[
        (df_raw["ATCF"].str.contains(settings["basin"])) &
        (df_raw["ftime(hr)"] == settings["leadtime"])
    ]

    if missing is not None:
        df = df.drop(df.index[df[y_name[0]] == missing])

    # Shuffle the rows in the df Dataframe, using the numpy rng.
    # rng = np.random.default_rng(settings['rng_seed'])
    df = df.sample(frac=1,random_state=settings['rng_seed'])

    # Extract the x columns and the y column.
    x_data = df[x_names].to_numpy()
    y_data = np.squeeze(df[y_name].to_numpy())

    # Get the training and validation splits.
    n_val   = settings["n_val"]
    n_train = settings["n_train"]
    if(settings["n_train"] == "max"):
        n_train = np.shape(x_data)[0] - n_val
    else:
        n_train = settings["n_train"]
        
    if(n_val + n_train > np.shape(x_data)[0]):
        raise ValueError('n_val + n_train > np.shape(x_data)')
    
    x_train = x_data[:n_train]
    y_train = y_data[:n_train]

    # grab subset of training for evaluating out-of-sample predictions
    try:
        if(settings["train_condition"]=='DV12<=15'):
            i_var = x_names.index('DV12')
            i_index = np.where(x_train[:,i_var]<=15)[0]
            x_train = x_train[i_index,:]
            y_train = y_train[i_index]  
    except:
        print('settings["train_condition"] is undefined')
    
    if n_val == 0:
        x_val  = x_train
        y_val  = y_train
        df_val = df
    else:
        x_val  = x_data[n_train:n_train+n_val]
        y_val  = y_data[n_train:+n_train+n_val]
        df_val = df.iloc[n_train:+n_train+n_val]

    
    # Create 'onehot' y arrays. The y values go in the first column, and the
    # remaining columns are zero -- i.e. dummy columns.  These dummy columns
    # are required by tensorflow; the number of columns must equal the number
    # of distribution parameters.
    if settings["uncertainty_type"] == "bnn":
        n_parameters = 1
    elif settings["uncertainty_type"] == "shash2":
        n_parameters = 2
    elif settings["uncertainty_type"] == "shash3":
        n_parameters = 3
    elif settings["uncertainty_type"] == "shash4":
        n_parameters = 4
    else:
        raise NotImplementedError

    onehot_train = np.zeros((len(y_train), n_parameters))
    onehot_val = np.zeros((len(y_val), n_parameters))

    onehot_train[:, 0] = y_train
    onehot_val[:, 0] = y_val

    # Make a descriptive dictionary.
    data_summary = {
        "datafile_path": datafile_path,
        "x_train_shape": tuple(x_train.shape),
        "x_val_shape": tuple(x_val.shape),
        "onehot_train_shape": tuple(onehot_train.shape),
        "onehot_val_shape": tuple(onehot_val.shape),
        "x_names": x_names,
        "y_name": y_name,
    }

    # Report the results.
    if verbose >= 1:
        pprint.pprint(data_summary, width=80)

    if verbose >= 2:
        toolbox.print_summary_statistics({"y_train": y_train, "y_val": y_val}, sigfigs=1)

    return (
        x_train,
        onehot_train,
        x_val,
        onehot_val,
        data_summary,
        df_val,
    )
