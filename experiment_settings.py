"""Hurricane prediction experimental settings

uncertainty_type: "shash3", "bnn", "mcdrop", "reg"
val_condition   : "random", "years"  

"""

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "21 January 2022"


def get_settings(experiment_name):
    experiments = {     
        "intensity0_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'reg',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "dropout_rate": [0.,0.,0.],
            "ridge_param": [0.0,0.0],            
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": [605, 122, 786, 311, 888, 999, 578, 331, 908, 444],
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "test_condition": (2018,),
            "val_condition": "random",
            "n_val": 256,
            "n_train": "max",               
        },                
        "intensity1_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "dropout_rate": [0.,0.,0.],
            "ridge_param": [0.0,0.0],            
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": [605, 122, 786, 311, 888, 999, 578, 331, 908, 444],
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "test_condition": (2018,),
            "val_condition": "random",#"random","years"
            "n_val": 256,
            "n_train": "max",            
        },        
        "intensity2_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "dropout_rate": [0.,0.,0.],
            "ridge_param": [0.0,0.0],            
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": [605, 122, 786, 311, 888, 999, 578, 331, 908, 444],
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "test_condition": (2018,),
            "val_condition": "random",#"random","years"
            "n_val": 256,
            "n_train": "max",            
        },     
        "intensity3_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [60, 40],
            "dropout_rate": [0.,0.75,0.75],
            "ridge_param": [0.0,0.0],            
            "learning_rate": 0.00005,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": [605, 122, 786, 311, 888, 999, 578, 331, 908, 444],
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "test_condition": (2018,),
            "val_condition": "random",#"random","years"
            "n_val": 256,
            "n_train": "max",                
        },             
        
    }

    return experiments[experiment_name]
