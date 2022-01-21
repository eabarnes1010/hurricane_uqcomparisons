"""Build the fully-connected network architecture.

Functions
---------
build_shash_model(hiddens, input_shape, output_shape, ridge_penalty, act_fun, rng_seed)
build_bnn_model(hiddens, input_shape, output_shape, ridge_penalty, act_fun, rng_seed)

"""
import numpy as np

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow import keras
import tensorflow_probability as tfp
from distributions import normal_softplus

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__version__ = "14 December 2021"

class Exponentiate(keras.layers.Layer):
    """Custom layer to exp the sigma and tau estimates inline."""

    def __init__(self, **kwargs):
        super(Exponentiate, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.exp(inputs)


def build_shash_model(
    x_train, onehot_train, hiddens, output_shape, ridge_penalty=0.0, act_fun="relu", rng_seed=999, dropout_rate=0.0,
):
    """Build the fully-connected shash network architecture with
    internal scaling.

    Arguments
    ---------
    x_train : numpy.ndarray
        The training split of the x data.
        shape = [n_train, n_features].

    onehot_train : numpy.ndarray
        The training split of the scaled y data is in the first column.
        The remaining columns are filled with zeros. The number of columns
        equal the number of distribution parameters.
        shape = [n_train, n_parameters].

    hiddens : list (integers)
        Numeric list containing the number of neurons for each layer.

    output_shape : integer {2, 3, 4}
        The number of distribution output parameters to be learned.

    ridge_penalty : float, default=0.0
        The L2 regularization penalty for the first layer.

    act_fun : function, default="relu"
        The activation function to use on the deep hidden layers.

    Returns
    -------
    model : tensorflow.keras.models.Model

    Notes
    -----
    * The number of output units is determined by the output_shape argument.
        If output_shape is:
        2 -> output_layer = [mu_unit, sigma_unit]
        3 -> output_layer = [mu_unit, sigma_unit, gamma_unit]
        4 -> output_layer = [mu_unit, sigma_unit, gamma_unit, tau_unit]

    * Unlike most of EAB's models, the features are normalized within the
        network.  That is, the x_train, y_train, ... y_test should not be
        externally normalized or scaled.

    * In essence, the network is learning the shash parameters for the
        normalized y values. Say mu_z and sigma_z where

            z = (y - y_avg)/y_std

        The mu_unit and sigma_unit layers rescale the learned mu_z
        and sigma_z parameters back to the dimensions of the y values.
        Specifically,

            mu_y    = y_std * mu_z + y_avg
            sigma_y = y_std * sigma_z

        However, since the model works with log(sigma) we must use

            log(sigma_y) = log(y_std * sigma_z) = log(y_std) + log(sigma_z)

    * Note the gamma and tau parameters of the shash distribution are
        dimensionless by definition. So we do not need to rescale gamma
        and tau.

    """
    # The avg and std for feature normalization are computed from x_train.
    # Using the .adapt method, these are set once and do not change, but
    # the constants travel with the model.
    inputs = tf.keras.Input(shape=x_train.shape[1:])

    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(x_train)
    x = normalizer(inputs)

    # linear network only
    if hiddens[0] == 0:
        x = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridge_penalty),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
        )(x)
    else:
        # Initialize the first hidden layer.
        x = tf.keras.layers.Dense(
            units=hiddens[0],
            activation=act_fun,
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridge_penalty),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
        )(x)
        # x = tf.keras.layers.Dropout(
        #     rate=dropout_rate,
        #     seed=rng_seed,            
        # )(x)        

        # Initialize the subsequent hidden layers.
        for layer_size in hiddens[1:]:
            x = tf.keras.layers.Dense(
                units=layer_size,
                activation=act_fun,
                use_bias=True,
                kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
                bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
            )(x)
            # x = tf.keras.layers.Dropout(
            #     rate=dropout_rate,
            #     seed=rng_seed,            
            # )(x)            

    # Compute the mean and standard deviation of the y_train data to rescale
    # the mu and sigma parameters.
    y_avg = np.mean(onehot_train[:, 0])
    y_std = np.std(onehot_train[:, 0])

    # mu_unit.  The network predicts the scaled mu_z, then the resclaing
    # layer scales it up to mu_y.
    mu_z_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
        name="mu_z_unit",
    )(x)

    mu_unit = tf.keras.layers.Rescaling(
        scale=y_std,
        offset=y_avg,
        name="mu_unit",
    )(mu_z_unit)

    # sigma_unit. The network predicts the log of the scaled sigma_z, then
    # the resclaing layer scales it up to log of sigma y, and the custom
    # Exponentiate layer converts it to sigma_y.
    log_sigma_z_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="log_sigma_z_unit",
    )(x)

    log_sigma_unit = tf.keras.layers.Rescaling(
        scale=1.0,
        offset=np.log(y_std),
        name="log_sigma_unit",
    )(log_sigma_z_unit)

    sigma_unit = Exponentiate(
        name="sigma_unit",
    )(log_sigma_unit)

    # Add gamma and tau units if requested.
    if output_shape == 2:
        output_layer = tf.keras.layers.concatenate([mu_unit, sigma_unit], axis=1)

    else:
        # gamma_unit. The network predicts the gamma directly.
        gamma_unit = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            use_bias=True,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.keras.initializers.Zeros(),
            name="gamma_unit",
        )(x)

        if output_shape == 3:
            output_layer = tf.keras.layers.concatenate(
                [mu_unit, sigma_unit, gamma_unit], axis=1
            )

        else:
            # tau_unit. The network predicts the log of the tau, then
            # the custom Exponentiate layer converts it to tau.
            log_tau_unit = tf.keras.layers.Dense(
                units=1,
                activation="linear",
                use_bias=True,
                bias_initializer=tf.keras.initializers.Zeros(),
                kernel_initializer=tf.keras.initializers.Zeros(),
                name="log_tau_unit",
            )(x)

            tau_unit = Exponentiate(
                name="tau_unit",
            )(log_tau_unit)

            if output_shape == 4:
                output_layer = tf.keras.layers.concatenate(
                    [mu_unit, sigma_unit, gamma_unit, tau_unit], axis=1
                )
            else:
                raise NotImplementedError

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model

def build_bnn_model(
    x_train, onehot_train, hiddens, output_shape, ridge_penalty=0.0, act_fun="relu", rng_seed=999,
):
    """Build the fully-connected BNN architecture with
    internal scaling.

    Arguments
    ---------
    x_train : numpy.ndarray
        The training split of the x data.
        shape = [n_train, n_features].

    onehot_train : numpy.ndarray
        The training split of the scaled y data is in the first column.
        shape = [n_train,].

    hiddens : list (integers)
        Numeric list containing the number of neurons for each layer.

    output_shape : integer 
        The prediction.

    ridge_penalty : float, default=0.0
        The L2 regularization penalty for the first layer.
        ***NOT USED FOR THE BNN***

    act_fun : function, default="relu"
        The activation function to use on the deep hidden layers.
    
    rng_seed : integer, default=999
        Random seed for layer initialization
    

    Returns
    -------
    model : tensorflow.keras.models.Model

    Notes
    -----

    """
    # The avg and std for feature normalization are computed from x_train.
    # Using the .adapt method, these are set once and do not change, but
    # the constants travel with the model.
    inputs = tf.keras.Input(shape=x_train.shape[1:])

    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(x_train)
    x = normalizer(inputs)

    # define kl divergence functions for BNN
    # the two lines below rescale the kl divergence ("kind of a bug fix for TFP" - Duerr 2020)
    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (x_train.shape[0] * 1.0)
    bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (x_train.shape[0] * 1.0)
    
    # Initialize the first hidden layer.
    x = tfp.layers.DenseFlipout(
        units=hiddens[0],
        activation=act_fun,
        bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
        bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_divergence_fn=bias_divergence_fn,
        seed=rng_seed,        
    )(x)

    # Initialize the subsequent hidden layers.
    for layer_size in hiddens[1:]:
        x = tfp.layers.DenseFlipout(
            units=layer_size,
            activation=act_fun,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_divergence_fn=bias_divergence_fn,
            seed=rng_seed,
        )(x)

    # penultimate layer (sigma, mu)
    params = tfp.layers.DenseFlipout(
             units=2,
             bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
             bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
             kernel_divergence_fn=kernel_divergence_fn,
             bias_divergence_fn=bias_divergence_fn,
             seed=rng_seed,
    )(x)
    
    dist = tfp.layers.DistributionLambda(normal_softplus)(params)
    model = tf.keras.models.Model(inputs=inputs, outputs=dist)    
    
    # model_params = tf.keras.models.Model(inputs=inputs, outputs=params) # to be used later to study the params if you want to
    
    return model


def build_mcdrop_model(
    x_train, onehot_train, hiddens, output_shape, ridge_penalty=0.0, act_fun="relu", rng_seed=999, dropout_rate=0.0,
):
    """Build the fully-connected MC-Dropout architecture with
    internal scaling.

    Arguments
    ---------
    x_train : numpy.ndarray
        The training split of the x data.
        shape = [n_train, n_features].

    onehot_train : numpy.ndarray
        The training split of the scaled y data is in the first column.
        shape = [n_train,].

    hiddens : list (integers)
        Numeric list containing the number of neurons for each layer.

    output_shape : integer 
        The prediction.

    ridge_penalty : float, default=0.0
        The L2 regularization penalty for the first layer.
        ***NOT USED FOR THE BNN***

    act_fun : function, default="relu"
        The activation function to use on the deep hidden layers.
    
    rng_seed : integer, default=999
        Random seed for layer initialization
    

    Returns
    -------
    model : tensorflow.keras.models.Model

    Notes
    -----

    """
    # The avg and std for feature normalization are computed from x_train.
    # Using the .adapt method, these are set once and do not change, but
    # the constants travel with the model.
    inputs = tf.keras.Input(shape=x_train.shape[1:])

    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(x_train)
    x = normalizer(inputs)

    # Initialize the first hidden layer.   
    x = tf.keras.layers.Dense(
        units=hiddens[0],
        activation=act_fun,
        use_bias=True,
        kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridge_penalty),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),     
    )(x)
    
    x = tf.keras.layers.Dropout(
        rate=dropout_rate,
        seed=rng_seed,            
    )(x)    

    # Initialize the subsequent hidden layers.
    for layer_size in hiddens[1:]:        
        x = tf.keras.layers.Dense(
            units=hiddens[0],
            activation=act_fun,
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridge_penalty),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
        )(x)
        
        x = tf.keras.layers.Dropout(
            rate=dropout_rate,
            seed=rng_seed,            
        )(x)
    
    # final layer
    x = tf.keras.layers.Dense(output_shape,
                              activation='linear',
                              use_bias=True,
                              kernel_initializer=tf.keras.initializers.HeNormal(seed=rng_seed),
                              bias_initializer=tf.keras.initializers.HeNormal(seed=rng_seed),
                             )(x)        
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)    
    
    return model
