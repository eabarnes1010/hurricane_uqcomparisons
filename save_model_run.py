"""Save the model, weights, history, and metadata.

Functions
---------
save_model_run(data_summary, fit_summary, model, model_path,
    model_name, settings, version)

"""
import json
import pickle

import tensorflow as tf

import toolbox

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "30 October 2021"


def save_model_run(
    data_summary,
    fit_summary,
    model,
    model_path,
    model_name,
    settings,
    version,
):
    """Save the model, weights, history, and metadata.

    Arguments
    ---------
    data_summary : dict

    fit_summary : dict

    model : tensorflow.keras.models.Model

    model_path : str
        Path to the folder for saved models, which is used to store the
            *_model,
            *_weights.h5,
            *_history.pickle, and
            *_metadata.json
        files for a run.

    model_name : str
        The unique model name to distinguish one run form another. This name
        is the initial component of each saved file and model folder.

    settings : dict
        Dictionary of experiment settings for the run.

    version : str
        Version of the train_intensity notebook.

    Returns
    -------
    None

    """
    # Save the model, weights, and history.
    tf.keras.models.save_model(
        model, model_path + model_name + "_model", overwrite=True
    )

    model.save_weights(model_path + model_name + "_weights.h5")

    with open(model_path + model_name + "_history.pickle", "wb") as handle:
        pickle.dump(model.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the metadata.
    metadata = {
        "VERSION": version,
        "MACHINE_LEARNING_ENVIRONMENT": toolbox.get_ml_environment(),
        "MODEL_NAME": model_name,
        "SETTINGS": settings,
        "DATA_SUMMARY": data_summary,
        "FIT_SUMMARY": fit_summary,
    }
    with open(model_path + model_name + "_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent="   ", cls=toolbox.NumpyEncoder)
