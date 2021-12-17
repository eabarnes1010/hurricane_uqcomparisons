"""Hurricane prediction experimental settings"""

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "30 October 2021"


def get_settings(experiment_name):
    experiments = {     
       
        
        "intensity0_AL72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3', 
            "leadtime": 72,
            "basin": "AL",
            "target": "intensity",
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
            "n_train": "max",
        }, 
        
        "intensity1_AL72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn', 
            "leadtime": 72,
            "basin": "AL",
            "target": "intensity",
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
            "n_train": "max",
        },      
        
        "intensity2_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',            
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
        },  
        "intensity3_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
        },   
        
        "intensity4_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',            
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
        },        
        "intensity5_EPCP72": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',            
            "leadtime": 72,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
        }, 
        
        "intensity100_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',            
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": 804,
        },  
        "intensity101_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": 804,
        },         
        
        "intensity102_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',            
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": 402,
        },  
        "intensity103_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": 402,
        },   
        "intensity104_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',            
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
            "train_condition": "DV12<=15",              
        },  
        "intensity105_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
            "train_condition": "DV12<=15",             
        },      
        "intensity106_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',            
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
            "train_condition": "VMXC<=90",              
        },  
        "intensity107_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
            "train_condition": "VMXC<=90",             
        },      
        
        "intensity200_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'shash3',            
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
            "train_condition": "cluster",                         
        }, 
        
        "intensity201_EPCP48": {
            "filename": "nnfit_vlist_intensity_and_track_extended.dat",
            "uncertainty_type": 'bnn',            
            "leadtime": 48,
            "basin": "EP|CP",
            "target": "intensity",
            "undersample": False,
            "hiddens": [15, 10],
            "learning_rate": 0.0001,
            "momentum": 0.9,
            "nesterov": True,
            "batch_size": 64,
            "rng_seed": 999,
            "act_fun": "relu",
            "n_epochs": 25_000,
            "patience": 100,
            "ridge_param": 0.0,
            "n_val": 512,
            "n_train": "max",
            "train_condition": "cluster",                         
        },        
             
               
    }

    return experiments[experiment_name]
