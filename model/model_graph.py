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
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    r_loss = tf.math.reduce_mean(tf.square(y_true-y_pred),axis= [1,2])
    return r_loss

def soft_f1(y_true, y_pred):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        tp = tf.reduce_sum(y_pred * y_true, axis=2)
        fp = tf.reduce_sum(y_pred * (1 - y_true), axis=2)
        fn = tf.reduce_sum((1 - y_pred) * y_true, axis=2)

        soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = tf.reduce_mean(cost,axis=1) # average on all labels
        return macro_cost

def cat_cross_entropy(y_true,y_pred):
    # Categorical CE requries the tensors to be same shape (rectangular matricies)
    # Define the start and end idxs of all onehot encodings, group by size and finally,
    # sum up and average the categorical loss
    def _percentage(x):
        x = tf.abs(x)
        sum_axis = tf.reshape(tf.reduce_sum(x,axis = 2),shape=(x.shape[0],x.shape[1],1))
        return tf.divide(x,sum_axis)
        
    slices = [(4,13),(13,22),(49,59),(37,49),(163,179),(180,196)]
    idxDiff = slices[0][1]-slices[0][0]
    loss = tf.zeros([1,1,y_true.shape[0]],dtype=tf.float32)
    sameDiffList = []
    for idx1,idx2 in slices:
        if idx2 - idx1 == idxDiff:
            sameDiffList.append((idx1,idx2))
            
        elif idx2-idx1 != idxDiff:
            YTRUE = tf.concat([y_true[...,i[0]:i[1]] for i in sameDiffList],axis=1)
            YPRED = tf.concat([y_pred[...,i[0]:i[1]] for i in sameDiffList],axis=1)
            YPRED = _percentage(YPRED)
            f1loss = soft_f1(YTRUE,YPRED)
            loss = tf.concat([loss,tf.reshape(f1loss,(1,1,YTRUE.shape[0]))],axis=1)
            
            sameDiffList = []
            sameDiffList.append((idx1,idx2))
            idxDiff = idx2-idx1

    YTRUE = tf.concat([y_true[...,i[0]:i[1]] for i in sameDiffList],axis=1)
    YPRED = tf.concat([y_pred[...,i[0]:i[1]] for i in sameDiffList],axis=1)
    YPRED = _percentage(YPRED)
    f1loss = soft_f1(YTRUE,YPRED)
    loss = tf.concat([loss,tf.reshape(f1loss,(1,1,YTRUE.shape[0]))],axis=1)[:,1:,:]
    return tf.reduce_sum(loss,axis=1)[0]

def mix_loss(y_true,y_pred):
    
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    # For the mix sliders use cosine_similarity as they are 1 dimensional vectors of values < 1
    # This is in addition to the value improving forces of the MSE
    #Take slices of mix sliders
    slices = [(28,37),(64,73),(82,91),(94,103),(106,115),(119,128),(131,140),(143,152),(154,163)]
    YTRUE = tf.concat([y_true[...,i[0]:i[1]] for i in slices],axis=2)
    YPRED = tf.concat([y_pred[...,i[0]:i[1]] for i in slices],axis=2)
    cos_sim = tf.losses.cosine_similarity(YTRUE,YPRED,axis=[1,2])
    
    def normalise(x):
        norm = tf.divide(tf.add(x,1), 2)
        corrected_norm = tf.where(norm < 1e-10, tf.constant(0.0, dtype=x.dtype), norm)
        return corrected_norm
    
    return normalise(cos_sim)

def custom_loss(y_true,y_pred):
    # Add all the losses together
    sliderLoss = mse_loss(y_true,y_pred)
    onehotLoss = cat_cross_entropy(y_true,y_pred)
    mixLoss = mix_loss(y_true,y_pred)
    return 100*(sliderLoss + onehotLoss + mixLoss)
    
#Training
@tf.function
def train_step(inp,gt,dec):
    with tf.GradientTape() as decoder:
        generated= dec(inp)
        loss = custom_loss(gt,generated)
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
        print (f'TRAINING LOSS: {np.mean(loss)} Time for epoch {epoch + 1} is {time.time()-start} sec', end='\r')
    return decode

if __name__ == "__main__":
    decode = dec()
    # tf.saved_model.save(decode,"graph_saved_model")
    