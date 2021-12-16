
import tensorflow_probability as tfp
import tensorflow as tf

def normal_softplus(params): 
    return tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))


