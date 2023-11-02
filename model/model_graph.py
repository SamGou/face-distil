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
    
    x = layers.Dense(2622//8,name="dense_3")(x)
    x = layers.LeakyReLU(name="lrelu_2")(x)
    
    x = layers.Dense(2622//16,name="dense_3")(x)
    x = layers.LeakyReLU(name="lrelu_2")(x)
    
    # out = layers.Dense(197,name="out",activation="tanh")(x)
    # TODO ALL TANH'S CAN BE COMBINED, SOFTMAX STAY SEPARATE    
    # # Return each separately as different activation functions could be utilised
    # #Body
    r_cos_sin = layers.Dense(3, name='r_cos_sin',activation="tanh")(x)
    gender = layers.Dense(1, name='gender',activation="sigmoid")(x)
    skintone = layers.Dense(9,name='skintone',activation="softmax")(x)
    
    # #Headshape
    headshape_preset = layers.Dense(9,name="headshape_preset",activation="softmax")(x)
    headshape_refine_dir =  layers.Dense(2,name="headshape_refine",activation="softmax")(x)
    
    # Eyes
    eyes_sliders = layers.Dense(4,name="eyes_sliders",activation="tanh")(x)
    eyes_mix = layers.Dense(9,name="eyes_mix",activation="softmax")(x)
    
    # eyecolour
    eye_colour = layers.Dense(12,name="eyecolour",activation="softmax")(x)
    
    #Eyebrows
    eyebrows_sliders = layers.Dense(4,name="eyebrows",activation="tanh")(x)
    eyebrows_mix = layers.Dense(10,name="eyebrows_mix",activation="softmax")(x)
    
    #Forehead
    forehead_sliders = layers.Dense(2,name="forehead_sliders",activation="tanh")(x)
    forehead_mix = layers.Dense(9,name="forehead_mix",acitvation="softmax")(x)
    
    #Nose
    nose_sliders = layers.Dense(9,name="nose_sliders",activation ="tanh")(x)
    nose_mix = layers.Dense(9,name="nose_mix", activation="softmax")(x)
    
    #Ears
    ears_sliders = layers.Dense(3,name="ears_sliders",activation="tanh")(x)
    ears_mix = layers.Dense(9,name="ears_mix",activation="softmax")(x)
    
    #Cheeks 
    cheeks_sliders = layers.Dense(3,name="cheeks_sliders",activation="tanh")(x)
    cheeks_mix = layers.Dense(9,name="cheeks_mix",activation="softmax")(x)
    
    #Mouth
    mouth_sliders = layers.Dense(4,name="mouth_sliders",activation="tanh")(x)
    mouth_mix = layers.Dense(9,name="mouth_mix",activation="softmax")(x)
    
    #Jaw
    jaw_sliders = layers.Dense(3, name="jaw_sliders",activation="tanh")(x)
    jaw_mix = layers.Dense(9,name = "jaw_mix",activation="tanh")(x)
    
    #Chin
    chin_sliders = layers.Dense(3, name="chin_sliders",activation="tanh")(x)
    chin_mix = layers.Dense(9,name="chin_mix",activation="softmax")(x)
    
    #neck
    neck_sliders = layers.Dense(2,name="neck_sliders",activation="tanh")(x)
    neck_mix = layers.Dense(9,name="neck_mix",activation="softmax")(x)
    
    #facialforms
    ff1_preset = layers.Dense(16, name="ff1_preset",activation="softmax")(x)
    ff2_preset = layers.Dense(16, name="ff2_preset",activation="softmax")(x)
    ff1_intensity = layers.Dense(1,name="ff1_intensity",activation="sigmoid")(x)
    ff2_intensity = layers.Dense(1,name="ff2_intensity",activation="sigmoid")(x)
    model = tf.keras.Model(inputs, (r_cos_sin,gender,skintone,headshape_preset,headshape_refine_dir,
                                    eyes_sliders,eyes_mix,eye_colour,eyebrows_sliders,eyebrows_mix,
                                    forehead_sliders,forehead_mix,nose_sliders,nose_mix,
                                    ears_sliders,ears_mix,cheeks_sliders,cheeks_mix,mouth_sliders,mouth_mix,
                                    jaw_sliders,jaw_mix,chin_sliders,chin_mix,neck_sliders,neck_mix,
                                    ff1_preset,ff1_intensity,ff2_preset,ff2_intensity), name="Decoder")
    return model

# Optimizer
optimizer = tf.keras.optimizers.Adam(lr=5e-4)

# Define Losses
def mse_loss(y_true, y_pred):
    r_loss = tf.math.reduce_mean(tf.square(y_true-y_pred),axis= [1,2])
    return 1000*r_loss

def binary_cross_entropy(y_true,y_pred):
    # TODO add all the bce viable vectors [0,0,0,0,1,0,0]
    # they need to be the same size (rectangular matricies)
    
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