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

INIT_LR = 5e-4
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
    
    out = layers.Dense(197,name="out",activation="tanh")(x)
    
    model = tf.keras.Model(inputs, out, name="Decoder")
    return model

# Define Utils
def normalise(x):
    # normalise from (-1,1) to (0,1)
    norm = tf.divide(tf.add(x,1), 2)
    corrected_norm = tf.where(norm < 1e-10, tf.constant(0.0, dtype=x.dtype), norm)
    return corrected_norm

# Define Losses
def slider_loss(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    slices = [(0,4),(22,24),(24,28),(59,62),(62,64),
              (73,82),(91,94),(103,106),(115,119),(128,131),
              (140,143),(152,154),(179,180),(196,197)]
    YTRUE = tf.concat([y_true[...,i[0]:i[1]] for i in slices],axis=2)
    YPRED = tf.concat([y_pred[...,i[0]:i[1]] for i in slices],axis=2)
    r_loss = tf.math.reduce_mean(tf.square(YTRUE-YPRED),axis= [1,2])
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

def gender_loss(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    YTRUE = y_true[...,3:4]
    YPRED = y_pred[...,3:4]
    YPRED = normalise(YPRED)
    f1loss = soft_f1(YTRUE,YPRED)
    return f1loss
      
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
    
    def _mix_post_process(x,threshold= 0.04):
        ex = tf.where(x>=0.,x,0.0)
        sums = tf.reshape(tf.reduce_sum(ex,axis=2),(ex.shape[0],ex.shape[1],1))
        sums = tf.where(sums>1., sums, 1.)
        ex  = tf.divide(ex, sums)
        ex = tf.where(ex>=threshold,ex,0.0)
        return ex
    
    def _get_index_loss(y_true,y_pred):
        idx_ytrue = tf.where(y_true>0., 1.,0.)
        idx_ypred = tf.where(y_pred>0.,1.,0.)
        return soft_f1(idx_ytrue,idx_ypred)
        
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    slices = [(28,37),(64,73),(82,91),(94,103),(106,115),(119,128),(131,140),(143,152),(154,163)]
    YTRUE = tf.concat([y_true[...,i[0]:i[1]] for i in slices],axis=1)
    YPRED = tf.concat([y_pred[...,i[0]:i[1]] for i in slices],axis=1)
    YPRED = _mix_post_process(YPRED)
    idx_loss = _get_index_loss(YTRUE,YPRED)
    mse = tf.math.reduce_mean(tf.square(YTRUE-YPRED),axis= [1,2])
    
    return mse+idx_loss 

def custom_loss(y_true,y_pred):
    # Add all the losses together
    sliderLoss = slider_loss(y_true,y_pred)
    onehotLoss = cat_cross_entropy(y_true,y_pred)
    mixLoss = mix_loss(y_true,y_pred)
    gl = gender_loss(y_true,y_pred)
    return 100*(sliderLoss + onehotLoss + mixLoss + gl)
    
optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR)

class LRSched:
    def __init__(self,initial_learningrate):
        self.last_change_epoch = 0
        self.init_lr = initial_learningrate
        self.warmup_steps = 666
        self.warmup_target = 5e-3
        self.warmup_threshold = 1000
        
    def K(self,init,target,steps):
        return np.log(target/init)/(self.last_change_epoch+steps)
    
    def set_learning_rate(self,opt,epoch,epochs):
        if epoch >= int(epochs*0.6):
            steps_to_target = 2000
            if epoch > int(epochs*0.6)+ steps_to_target:
                opt.learning_rate = 1e-6
            elif epoch == int(epochs*0.6):
                self.last_change_epoch = int(epochs*0.6)
                opt.learning_rate = 5e-5*np.exp(epoch*self.K(5e-5,1e-6,steps_to_target))
            else:
                opt.learning_rate = 5e-5*np.exp(epoch*self.K(5e-5,1e-6,steps_to_target))
            
        elif epoch >= int(epochs*0.3):
            steps_to_target = 1500
            if epoch > int(epochs*0.3) + steps_to_target:
                opt.learning_rate = 1e-6
            elif epoch == int(epochs*0.3):
                self.last_change_epoch = int(epochs*0.3)
                opt.learning_rate = 1e-4*np.exp(epoch*self.K(1e-4,5e-5,steps_to_target))
            else:
                opt.learning_rate = 1e-4*np.exp(epoch*self.K(1e-4,5e-5,steps_to_target))
            
        elif epoch >= self.warmup_threshold:
            steps_to_target = 1500
            if epoch > self.warmup_threshold:
                opt.learning_rate = 1e-4
            elif epoch == self.warmup_threshold:
                self.last_change_epoch = self.warmup_threshold
                opt.learning_rate = self.warmup_target*np.exp(epoch*self.K(self.warmup_target,1e-4,steps_to_target))
            else:
                opt.learning_rate = self.warmup_target*np.exp(epoch*self.K(self.warmup_target,1e-4,steps_to_target))
            
        elif epoch <= self.warmup_steps:
            opt.learning_rate = self.init_lr*np.exp(epoch*self.K(self.init_lr,self.warmup_target,self.warmup_steps))    
        
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
    LR = LRSched(INIT_LR)
    for epoch in range(epochs):
        LR.set_learning_rate(optimizer,epoch,epochs)
        start = time.time()
        for inp,gt in dataset:
            loss= train_step(inp,gt,decode)
        print (f'TRAINING LOSS: {np.mean(loss)} EPOCH {epoch + 1} LR {optimizer.learning_rate.numpy()} TIMESTEP {time.time()-start}', end='\r')
    return decode

if __name__ == "__main__":
    decode = dec()
    # tf.saved_model.save(decode,"graph_saved_model")
    