"""Hurricane prediction experimental settings

uncertainty_type: "shash3", "bnn", "mcdrop", "reg"
val_condition   : "random", "years"  

"""

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__   = "17 March 2022"


def get_settings(experiment_name):
    experiments = {     
        "intensity201_AL24": {
            "filename": "nnfit_vlist_16-Mar-2022_eab.dat",
            "uncertainty_type": 'shash3',  #'reg', 'shash3', 'bnn', 'mcdrop'  
            "leadtime": 24,
            "basin": "AL",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "dropout_rate": [0.,0.,0.],
            "ridge_param": [0.0,0.0],            
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed_list": [416, 739,],#[416, 222, 598, 731, 414, 187, 650, 891, 739, 241],
            "rng_seed": None,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 250,
            "test_condition": "leave-one-out",
            "years_test": None,
            "val_condition": "random",
            "n_val": 200,
            "n_train": "max",
        },      
        
        "intensity4_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',  #'reg', 'shash3', 'bnn', 'mcdrop'  
            "leadtime": 48,
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
            "rng_seed_list": [416, 739,],#[416, 222, 598, 731, 414, 187, 650, 891, 739, 241],
            "rng_seed": None,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 250,
            "test_condition": "leave-one-out",
            "years_test": None,
            "val_condition": "random",
            "n_val": 200,
            "n_train": "max",
        },
        
        "intensity5_EPCP24": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',  #'reg', 'shash3', 'bnn', 'mcdrop'  
            "leadtime": 24,
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
            "rng_seed_list": [416, 739,],#[416, 222, 598, 731, 414, 187, 650, 891, 739, 241],
            "rng_seed": None,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 250,
            "test_condition": "leave-one-out",
            "years_test": None,
            "val_condition": "random",
            "n_val": 200,
            "n_train": "max",
        }        
        
    
    }

    return experiments[experiment_name]

