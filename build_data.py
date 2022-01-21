"""Build the split and scaled training and validation hurricane data arrays.

Functions
---------
build_hurricane_data(data_path, settings, verbose=0)

"""
import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

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
    df_raw = df_raw.rename(columns={'Date': 'year'})    

    df = df_raw[
        (df_raw["ATCF"].str.contains(settings["basin"])) &
        (df_raw["ftime(hr)"] == settings["leadtime"])
    ]

    if missing is not None:
        df = df.drop(df.index[df[y_name[0]] == missing])

    # Shuffle the rows in the df Dataframe, using the numpy rng.
    # rng = np.random.default_rng(settings['rng_seed'])
    df = df.sample(frac=1,random_state=settings['rng_seed'])
    df = df.reset_index(drop=True)

    #======================================================================
    # Train/Validation/Test Split
    
    # Get the testing data
    if settings["test_condition"] is None:
        pass
    elif settings["test_condition"] == "cluster":
        
        from scipy.cluster.vq import kmeans,vq
        numclust = 6
        
        data = np.copy(df[x_names].to_numpy())
        data_mean = np.mean(data,axis=0)
        data_std  = np.std(data,axis=0)
        data = (data - data_mean)/data_std

        clusters, dist = kmeans(data, numclust, iter=500, seed=settings["rng_seed"])
        cluster_label, _ = vq(data,clusters)
        class_freq = np.bincount(cluster_label)
        cluster_out = np.argmin(class_freq)

        index = np.where(cluster_label == cluster_out)[0]
        df_test = df.iloc[index]
        x_test = df_test[x_names].to_numpy()
        y_test = np.squeeze(df_test[y_name].to_numpy())
        df_test = df_test.reset_index(drop=True)
        
        df = df.drop(index)
        df = df.reset_index(drop=True)
        
        if verbose != 0:
            fig, axs = plt.subplots(1,2, figsize=(15,5))
            plt.sca(axs[0])
            plt.hist(cluster_label,np.arange(-.5,numclust+.5,1.), width=.98)
            plt.title('Sample Count by Cluster')
            plt.ylabel('number of samples')
            plt.xlabel('cluster')
            plt.xticks((0,1,2,3))
            plt.sca(axs[1])
            for ic in np.arange(0,numclust):
                plt.plot(x_names,clusters[ic,:], label='cluster ' + str(ic),linewidth=2)
            plt.legend()
            plt.title('Cluster Centroid')
            plt.ylabel('standardized units')
            plt.xlabel('predictor')
            plt.show() 
    else:
        years = settings["test_condition"]
        if verbose != 0:
            print('years' + str(years) + ' withheld for testing')
        index = df.index[df['year'].isin(years)]   
        df_test = df.iloc[index]
        x_test = df_test[x_names].to_numpy()
        y_test = np.squeeze(df_test[y_name].to_numpy())
        df_test = df_test.reset_index(drop=True)
        
        df = df.drop(index)
        df = df.reset_index(drop=True)
        
    # get the validation data
    if settings["val_condition"] == "random":
        index = np.arange(0,settings["n_val"])
        if(len(index)<100):
            raise Warning("Are you sure you want n_val > 100?")
            
    elif settings["val_condition"] == "years":
        unique_years = df['year'].unique()
        years = unique_years[:settings["n_val"]]
        index = df.index[df['year'].isin(years)] 
        
    df_val = df.iloc[index]
    x_val = df_val[x_names].to_numpy()
    y_val = np.squeeze(df_val[y_name].to_numpy())
    df_val = df_val.reset_index(drop=True)
    
    df = df.drop(index)
    df = df.reset_index(drop=True)
    
    if settings["test_condition"] is None:
        df_test = df_val.copy()
        x_test  = copy.deepcopy(x_val)
        y_test  = copy.deepcopy(y_val)
    
    # Subsample training if desired
    if settings["n_train"] == "max":
        df_train = df.copy()
    else:
        df_train = df.iloc[:settings["n_train"]]
    x_train = df_train[x_names].to_numpy()
    y_train = np.squeeze(df_train[y_name].to_numpy())
    df_train = df_train.reset_index(drop=True)
    
    #======================================================================    
    # Create 'onehot' y arrays. The y values go in the first column, and the
    # remaining columns are zero -- i.e. dummy columns.  These dummy columns
    # are required by tensorflow; the number of columns must equal the number
    # of distribution parameters.
    if settings["uncertainty_type"] in ("bnn","mcdrop","reg"):
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
    onehot_test = np.zeros((len(y_test), n_parameters))    

    onehot_train[:, 0] = y_train
    onehot_val[:, 0] = y_val
    onehot_test[:, 0] = y_test

    # Make a descriptive dictionary.
    data_summary = {
        "datafile_path": datafile_path,
        "x_train_shape": tuple(x_train.shape),
        "x_val_shape": tuple(x_val.shape),
        "x_test_shape": tuple(x_test.shape),        
        "onehot_train_shape": tuple(onehot_train.shape),
        "onehot_val_shape": tuple(onehot_val.shape),
        "onehot_test_shape": tuple(onehot_test.shape),        
        "x_names": x_names,
        "y_name": y_name,
    }

    # Report the results.
    if verbose >= 1:
        pprint.pprint(data_summary, width=80)

    if verbose >= 2:
        toolbox.print_summary_statistics({"y_train" : onehot_train[:,0], 
                                          "y_val"   : onehot_val[:,0], 
                                          "y_test"  : onehot_test[:,0]}, 
                                         sigfigs=1)
        
    # change dtype of onehot
    onehot_train = onehot_train.astype('float32')
    onehot_val = onehot_val.astype('float32')    
    onehot_test = onehot_test.astype('float32')        

    return (
        data_summary,        
        x_train,
        onehot_train,
        x_val,
        onehot_val,
        x_test,
        onehot_test,        
        df_train,
        df_val,
        df_test,
    )
