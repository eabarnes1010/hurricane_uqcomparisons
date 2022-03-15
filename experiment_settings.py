"""Hurricane prediction experimental settings

uncertainty_type: "shash3", "bnn", "mcdrop", "reg"
val_condition   : "random", "years"  

"""

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "15 March 2022"


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
            "uncertainty_type": 'mcdrop',  
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
        "intensity10_EPCP72": {
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
            "test_condition": (2017,),
            "val_condition": "random",
            "n_val": 256,
            "n_train": "max",               
        },                
        "intensity11_EPCP72": {
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
            "test_condition": (2017,),
            "val_condition": "random",
            "n_val": 256,
            "n_train": "max",            
        },        
        "intensity12_EPCP72": {
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
            "test_condition": (2017,),
            "val_condition": "random",
            "n_val": 256,
            "n_train": "max",            
        },     
        "intensity13_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'mcdrop',  
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
            "test_condition": (2017,),
            "val_condition": "random",
            "n_val": 256,
            "n_train": "max",                
        },             
        "intensity20_EPCP72": {
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
            "test_condition": (2019,),
            "val_condition": "random",
            "n_val": 256,
            "n_train": "max",               
        },                
        "intensity21_EPCP72": {
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
            "test_condition": (2019,),
            "val_condition": "random",
            "n_val": 256,
            "n_train": "max",            
        },        
        "intensity22_EPCP72": {
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
            "test_condition": (2019,),
            "val_condition": "random",
            "n_val": 256,
            "n_train": "max",            
        },     
        "intensity23_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'mcdrop',  
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
            "test_condition": (2019,),
            "val_condition": "random",
            "n_val": 256,
            "n_train": "max",                
        },             

        "intensity30_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'reg',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [5, 5],
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
        "intensity31_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [5, 5],
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
        "intensity32_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [5, 5],
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
        "intensity33_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'mcdrop',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [20, 20],
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
        "intensity40_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'reg',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [10, 10],
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
        "intensity41_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [10, 10],
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
        "intensity42_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [10, 10],
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
        "intensity43_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'mcdrop',  
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [40, 40],
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
