# import the necessary packages
import glob
import os
import time
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Define encoder - we only need to create latent variables to pass to the sampling network
# This is due to the assumption that the DeepFace representation of a face is already a latent vector
def enc(latent_var_shape = 2):
    inputs = keras.Input(shape=(None,2622,), name='input_layer')
    mean = layers.Dense(latent_var_shape, name='mean')(inputs)
    log_var = layers.Dense(latent_var_shape, name='log_var')(inputs)
    model = tf.keras.Model(inputs, (mean, log_var), name="Encoder")
    return model

# Define VAE sampling network
def sampling_reparam(distribution_params):
    mean,log_var = distribution_params
    epsilon = tf.random.normal(shape=tf.shape(mean),mean=0,stddev=1)
    z = mean + tf.exp(log_var / 2) * epsilon
    return z

def sampling(latent_var_shape=2):
    mean = keras.Input(shape=(None,latent_var_shape,),name="input_layer1")
    log_var = keras.Input(shape=(None,latent_var_shape,),name="input_layer2")
    out = layers.Lambda(sampling_reparam, name="samp_encoder_output")([mean,log_var])
    enc_2 = tf.keras.Model([mean,log_var],out,name="samp_Encoder_2")
    return enc_2

# Decoder
def dec(latent_var_shape=2):
    inputs = keras.Input(shape=(None,latent_var_shape,),name="input_layer")
    x = layers.Dense(32,name="dense_1")(inputs)
    
    x = layers.Dense(64,name="dense_2")(x)
    x = layers.LeakyReLU(name="lrelu_1")(x)
    
    x = layers.Dense(128,name="dense_3")(x)
    x = layers.LeakyReLU(name="lrelu_2")(x)
    
    outputs = layers.Dense(197,activation="tanh",name="dense_4_output")(x)
    model = tf.keras.Model(inputs,outputs,name="Decoder")
    return model

# Optimizer
optimizer = tf.keras.optimizers.Adam(lr=5e-4)

# Define Losses
def mse_loss(y_true, y_pred):
    r_loss = tf.math.reduce_mean(tf.square(y_true-y_pred),axis= [1,2])
    return 1000*r_loss

def kl_loss(mean,log_var):
    kl_loss = -0.5*tf.math.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis = 1)
    return kl_loss

def cat_cross_entropy(y_true,y_pred):
    # they need to be the same size (rectangular matricies)
    slices = [(4,13),(13,22),(49,59),(37,49),(163,179),(180,196)]
    idxDiff = slices[0][1]-slices[0][0]
    loss = []
    sameDiffList = []
    for idx1,idx2 in slices:
        if idx2 - idx1 == idxDiff:
            sameDiffList.append((idx1,idx2))
            continue
        elif idx2-idx1 != idxDiff:
            YTRUE = tf.concat([y_true[...,i[0]:i[1]] for i in sameDiffList],axis=2)
            YPRED = tf.concat([y_pred[...,i[0]:i[1]] for i in sameDiffList],axis=2)
            loss.append(tf.reduce_sum(tf.losses.categorical_crossentropy(YTRUE,YPRED)))
            
            sameDiffList = []
            sameDiffList.append((idx1,idx2))
            continue
    return tf.reduce_mean(loss)

def vae_loss(y_true, y_pred,mean,log_var):
    mse = mse_loss(y_true,y_pred)
    categorical = cat_cross_entropy(y_true,y_pred)
    kl = kl_loss(mean,log_var)
    return (mse,kl,mse + kl + categorical)

#Training
@tf.function
def train_step(inp,gt,enc,dec,samp):
    with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
        mean,log_var = enc(inp)
        latent = samp([mean,log_var])
        generated = dec(latent)
        
        generated = tf.cast(generated,tf.float64)
        mean = tf.cast(mean,tf.float64)
        log_var = tf.cast(log_var,tf.float64)
        
        mse,kl,loss = vae_loss(gt,generated,mean,log_var)
        # val_loss = mse_loss(val_gt,val_inp)
        
    encGrad = encoder.gradient(loss,enc.trainable_variables)
    decGrad = decoder.gradient(loss,dec.trainable_variables)
    
    optimizer.apply_gradients(zip(encGrad,enc.trainable_variables))
    optimizer.apply_gradients(zip(decGrad,dec.trainable_variables))
    return mse, kl, loss

def train(dataset,epochs):
    #init
    latent_var_shape = 16
    encode = enc(latent_var_shape)
    decode = dec(latent_var_shape)
    sample = sampling(latent_var_shape)
    for epoch in range(epochs):
        start = time.time()
        for inp,gt in dataset:
            mse,kl,loss= train_step(inp,gt,encode,decode,sample)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        # print(f"MSE: {mse}")
        # print(f"KLDIV: {kl}")
        print (f'TRAINING LOSS: {np.mean(loss)}')
        # print (f'VALIDATION LOSS (MSE): {val_loss}')
    return encode,decode,sample