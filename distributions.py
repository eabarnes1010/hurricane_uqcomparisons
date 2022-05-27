
import tensorflow_probability as tfp
import tensorflow as tf
import shash_tfp

def normal_dist(params): 
    return tfp.distributions.Normal(loc=params[:,0:1], scale=params[:,1:2])

def shash2_dist(params):
    return shash_tfp.Shash(params[:,0:1],params[:,1:2])

def shash3_dist(params):
    return shash_tfp.Shash(params[:,0:1],params[:,1:2],params[:,2:3])

def shash4_dist(params):
    return shash_tfp.Shash(params[:,0:1],params[:,1:2],params[:,2:3],params[:,3:4])


    