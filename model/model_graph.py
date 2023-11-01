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
def dec():
    inputs = keras.Input(shape=(None,2622,), name='input_layer')
    x = layers.Dense(2622//2,name="dense_1")(inputs)
    
    x = layers.Dense(2622//4,name="dense_2")(x)
    x = layers.LeakyReLU(name="lrelu_1")(x)
    
    x = layers.Dense(2622//6,name="dense_3")(x)
    x = layers.LeakyReLU(name="lrelu_2")(x)
    
    out = layers.Dense(197,name="out",activation="tanh")(x)
    
    # # Return each separately as different activation functions could be utilised
    # #Body
    # r_cos_sin = layers.Dense(3, name='r_cos_sin',activation="tanh")(x)
    # gender = layers.Dense(1, name='gender',activation="sigmoid")(x)
    # skintone = layers.Dense(9,name='skintone',activation="softmax")(x)
    
    # #Headshape
    # headshape_preset = layers.Dense(9,name="headshape_preset",activation="softmax")(x)
    # headshape_refine_dir =  layers.Dense(2,name="headshape_refine",activation="softmax")(x)
    
    # #Eyes
    # eyes = layers.Dense(13,name="eyes_sliders",activation="tanh")
    
    # #eyebrows
    
    model = tf.keras.Model(inputs, out, name="Decoder")
    return model

# Optimizer
optimizer = tf.keras.optimizers.Adam(lr=5e-4)

# Define Losses
def mse_loss(y_true, y_pred):
    r_loss = tf.math.reduce_mean(tf.square(y_true-y_pred),axis= [1,2])
    return 1000*r_loss

def binary_cross_entropy(y_true,y_pred):
    # TODO add all the bce viable vectors [0,0,0,0,1,0,0]
    bce = tf.keras.losses.binary_crossentropy()
#Training
@tf.function
def train_step(inp,gt,dec):
    with tf.GradientTape() as decoder:
        generated = dec(inp)
        generated = tf.cast(generated,tf.float64)
        loss = mse_loss(gt,generated)
        
    decGrad = decoder.gradient(loss,dec.trainable_variables)
    
    optimizer.apply_gradients(zip(decGrad,dec.trainable_variables))
    return loss

def train(dataset,epochs):
    #init
    decode = dec()
    for epoch in range(epochs):
        start = time.time()
        for inp,gt in dataset:
            loss= train_step(inp,gt,decode)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print (f'TRAINING LOSS: {np.mean(loss)}')
        # print (f'VALIDATION LOSS (MSE): {val_loss}')
    return decode