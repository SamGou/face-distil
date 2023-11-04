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
    x = layers.Dense(2622,name="dense_1")(inputs)
    
    x = layers.Dense(2622//4,name="dense_2")(x)
    x = layers.LeakyReLU(name="lrelu_1")(x)
    
    x = layers.Dense(2622//8,name="dense_3")(x)
    x = layers.LeakyReLU(name="lrelu_2")(x)
    
    x = layers.Dense(2622//12,name="dense_4")(x)
    x = layers.LeakyReLU(name="lrelu_3")(x)
    
    x = layers.Dense(2622//16,name="dense_5")(x)
    x = layers.LeakyReLU(name="lrelu_4")(x)
    
    out = layers.Dense(197,name="out",activation="linear")(x)
    
    model = tf.keras.Model(inputs, out, name="Decoder")
    return model

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

# Define Losses
def mse_loss(y_true, y_pred):
    r_loss = tf.math.reduce_mean(tf.square(y_true-y_pred),axis= [1,2])
    return 1000*r_loss

def cat_cross_entropy(y_true,y_pred):
    # Categorical CE requries the tensors to be same shape (rectangular matricies)
    # Define the start and end idxs of all onehot encodings, group by size and finally,
    # sum up and average the categorical loss
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



def mix_loss(y_true,y_pred,verb=False):
    
    def _my_tf_round(x, decimals = 0):
        multiplier = tf.constant(10**decimals, dtype=x.dtype)
        return tf.round(x * multiplier) / multiplier
    
    # For the mix sliders use cosine_similarity as they are 1 dimensional vectors of values < 1
    # This is in addition to the value improving forces of the MSE
    #Take slices of mix sliders
    slices = [(28,37),(64,73),(82,91),(94,103),(106,115),(119,128),(131,140),(143,152),(154,163)]
    YTRUE = tf.concat([y_true[...,i[0]:i[1]] for i in slices],axis=2)
    YPRED = tf.concat([y_pred[...,i[0]:i[1]] for i in slices],axis=2)
    if verb:
        tf.print(_my_tf_round(YTRUE,2),summarize=-1)
        tf.print(_my_tf_round(YPRED,2),summarize=-1)

    cos_sim = tf.losses.cosine_similarity(YTRUE,YPRED)
    
    def normalise(x):
        norm = tf.divide(tf.add(x,1), 2)
        corrected_norm = tf.where(norm < 1e-10, tf.constant(0.0, dtype=x.dtype), norm)
        return corrected_norm
    
    return tf.reduce_mean(normalise(cos_sim))

def custom_loss(y_true,y_pred,verb=False):
    # Add all the losses together
    sliderLoss = mse_loss(y_true,y_pred)
    onehotLoss = cat_cross_entropy(y_true,y_pred)
    mixLoss = mix_loss(y_true,y_pred,verb)
    return sliderLoss + onehotLoss + mixLoss
    
#Training
@tf.function
def train_step(inp,gt,dec,**kwargs):
    with tf.GradientTape() as decoder:
        generated= dec(inp)
        generated = tf.cast(generated,tf.float64)
        loss = custom_loss(gt,generated,kwargs["verb"])

    decGrad = decoder.gradient(loss,dec.trainable_variables)
    
    optimizer.apply_gradients(zip(decGrad,dec.trainable_variables))
    return loss

def train(dataset,epochs):
    #init 
    decode = dec()
    for epoch in range(epochs):
        start = time.time()
        verb = True if epoch == epochs - 1 else False
        for inp,gt in dataset:
            loss= train_step(inp,gt,decode,verb=verb)
        print (f'TRAINING LOSS: {np.mean(loss)} Time for epoch {epoch + 1} is {time.time()-start} sec', end='\r')
        # print (f'VALIDATION LOSS (MSE): {val_loss}')
    return decode

if __name__ == "__main__":
    decode = dec()
    # tf.saved_model.save(decode,"graph_saved_model")
    