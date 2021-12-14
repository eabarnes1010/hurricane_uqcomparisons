"""Hurricane prediction experimental settings"""

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "30 October 2021"


def get_settings(experiment_name):
    experiments = {

        "intensity0_AL72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3', #'shash3'
            "leadtime": 72,
            "basin": "AL",
            "target": "intensity",
            "loss": "likelihood",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 888,  
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 300,
            "ridge_param": 0.0,
            "n_val": 300,
            "n_test": 0,
        }, 
        
        "intensity1_AL72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn', #'shash3'
            "leadtime": 72,
            "basin": "AL",
            "target": "intensity",
            "loss": "likelihood",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 888,  
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 300,
            "ridge_param": 0.0,
            "n_val": 300,
            "n_test": 0,
        },        
    }

    return experiments[experiment_name]
