"""Hurricane prediction experimental settings

uncertainty_type: "shash3", "bnn", "mcdrop", "reg"
val_condition   : "random", "years"  

"""

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__   = "03 June 2022"


def get_settings(experiment_name):
    experiments = {   
        
     
        
        
        #=======================Old SHASH Code===============================================
        
        "intensity201_AL24": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'shash3', 
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        "intensity202_AL48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'shash3',  
            "leadtime": 48,
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        "intensity203_AL72": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'shash3',  
            "leadtime": 72,
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        "intensity204_AL96": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'shash3',  
            "leadtime": 96,
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        "intensity205_AL120": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'shash3',  
            "leadtime": 120,
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        
        "intensity301_EPCP24": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'shash3', 
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        "intensity302_EPCP48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'shash3',  
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        "intensity303_EPCP72": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        "intensity304_EPCP96": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'shash3', 
            "leadtime": 96,
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        "intensity305_EPCP120": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'shash3',  
            "leadtime": 120,
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        
        #------------------------------------------------------------
        # Additional UQ methods
        #------------------------------------------------------------
        
        "intensity521_AL48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'mcdrop',  
            "leadtime": 48,
            "basin": "AL",
            "target": "intensity",
            "undersample": False,
            "hiddens": [60, 40],
            "dropout_rate": [0.,0.75,0.75],
            "ridge_param": [0.0,0.0],            
            "learning_rate": 0.00005,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        
        "intensity522_AL48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'bnn',  
            "leadtime": 48,
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        
        "intensity523_AL48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'bnnshash',  
            "n_shash_params" : 3,                        
            "leadtime": 48,
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        
       "intensity531_EPCP48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'mcdrop',  
            "leadtime": 48,
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
        
        "intensity532_EPCP48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'bnn',  
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
       
        
        "intensity533_EPCP48": {
            "filename": "nnfit_vlist_02-Jun-2022.dat",
            "uncertainty_type": 'bnnshash',  
            "n_shash_params" : 3,                        
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
            "rng_seed_list": [222, 333, 416, 599, 739],
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
       
    
    }

    return experiments[experiment_name]

