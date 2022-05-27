
import tensorflow_probability as tfp
import tensorflow as tf
import shash

def normal_dist(params): 
    return tfp.distributions.Normal(loc=params[:,0:1], scale=params[:,1:2])


def normal_softplus(params): 
    return tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))
    

def shash_dist(params):
    return tfp.distributions.SinhArcsinh(params[:,0:1],params[:,1:2],params[:,2:3],params[:,3:4])

    