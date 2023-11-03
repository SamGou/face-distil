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
    
    # -----------------------------EXPERIMENTAL--------------------------------------------
    # ERROR - ValueError: Dimension 2 in both shapes must be equal, but are 9 and 16. Shapes are [14,1,9] and [14,1,16].
    # From merging shape 15 with other shapes. for '{{node Cast/x}} = Pack[N=18, T=DT_FLOAT, axis=0](Decoder/sliders/Tanh, Decoder/gender/Sigmoid, Decoder/skintone/Softmax, Decoder/headshape_preset/Softmax, Decoder/headshape_refine/Softmax, Decoder/eyes_mix/Softmax, Decoder/eyecolour/Softmax, Decoder/eyebrows_preset/Softmax, Decoder/forehead_mix/Softmax, Decoder/nose_mix/Softmax, Decoder/ears_mix/Softmax, Decoder/cheeks_mix/Softmax, Decoder/mouth_mix/Softmax, Decoder/jaw_mix/Softmax, Decoder/chin_mix/Softmax, Decoder/neck_mix/Softmax, Decoder/ff1_preset/Softmax, Decoder/ff2_preset/Softmax)' with input shapes: [14,1,41], [14,1,1], [14,1,9], [14,1,9], [14,1,2], [14,1,9], [14,1,12], [14,1,10], [14,1,9], [14,1,9], [14,1,9], [14,1,9], [14,1,9], [14,1,9], [14,1,9], [14,1,9], [14,1,16], [14,1,16].
    # Return each separately as different activation functions could be utilised    
    # Body
    # The sum in `sliders` is the output lengths of the commented dense layers
    sliders = layers.Dense(3+4+3+2+9+3+3+4+3+3+2, name='sliders',activation="tanh")(x)
    gender = layers.Dense(1, name='gender',activation="sigmoid")(x)
    skintone = layers.Dense(9,name='skintone',activation="softmax")(x)
    
    # #Headshape
    headshape_preset = layers.Dense(9,name="headshape_preset",activation="softmax")(x)
    headshape_refine_dir =  layers.Dense(2,name="headshape_refine",activation="softmax")(x)
    
    # Eyes
    #eyes_sliders = layers.Dense(4,name="eyes_sliders",activation="tanh")(x)
    eyes_mix = layers.Dense(9,name="eyes_mix",activation="softmax")(x)
    
    # eyecolour
    eye_colour = layers.Dense(12,name="eyecolour",activation="softmax")(x)
    
    #Eyebrows
    #eyebrows_sliders = layers.Dense(3,name="eyebrows",activation="tanh")(x)
    eyebrows_preset = layers.Dense(10,name="eyebrows_preset",activation="softmax")(x)
    
    #Forehead
    #forehead_sliders = layers.Dense(2,name="forehead_sliders",activation="tanh")(x)
    forehead_mix = layers.Dense(9,name="forehead_mix",activation="softmax")(x)
    
    #Nose
    #nose_sliders = layers.Dense(9,name="nose_sliders",activation ="tanh")(x)
    nose_mix = layers.Dense(9,name="nose_mix", activation="softmax")(x)
    
    #Ears
    #ears_sliders = layers.Dense(3,name="ears_sliders",activation="tanh")(x)
    ears_mix = layers.Dense(9,name="ears_mix",activation="softmax")(x)
    
    #Cheeks 
    #cheeks_sliders = layers.Dense(3,name="cheeks_sliders",activation="tanh")(x)
    cheeks_mix = layers.Dense(9,name="cheeks_mix",activation="softmax")(x)
    
    #Mouth
    #mouth_sliders = layers.Dense(4,name="mouth_sliders",activation="tanh")(x)
    mouth_mix = layers.Dense(9,name="mouth_mix",activation="softmax")(x)
    
    #Jaw
    #jaw_sliders = layers.Dense(3, name="jaw_sliders",activation="tanh")(x)
    jaw_mix = layers.Dense(9,name = "jaw_mix",activation="softmax")(x)
    
    #Chin
    #chin_sliders = layers.Dense(3, name="chin_sliders",activation="tanh")(x)
    chin_mix = layers.Dense(9,name="chin_mix",activation="softmax")(x)
    
    #neck
    #neck_sliders = layers.Dense(2,name="neck_sliders",activation="tanh")(x)
    neck_mix = layers.Dense(9,name="neck_mix",activation="softmax")(x)
    
    #facialforms
    ff1_preset = layers.Dense(16, name="ff1_preset",activation="softmax")(x)
    ff2_preset = layers.Dense(16, name="ff2_preset",activation="softmax")(x)
    ff1_intensity = layers.Dense(1,name="ff1_intensity",activation="sigmoid")(x)
    ff2_intensity = layers.Dense(1,name="ff2_intensity",activation="sigmoid")(x)
    
    continuous = tf.concat([sliders,gender,ff1_intensity,ff2_intensity],axis=2)
    softmax = tf.concat([skintone,headshape_preset,headshape_refine_dir,
                                    eyes_mix,eye_colour,eyebrows_preset,
                                    forehead_mix,nose_mix,
                                    ears_mix,cheeks_mix,mouth_mix,
                                    jaw_mix,chin_mix,neck_mix,
                                    ff1_preset,ff2_preset],axis=2)
    
    model = tf.keras.Model(inputs,(continuous,softmax),name="Decoder")
    
    #-----------------------------EXPERIMENTAL--------------------------------------------
    return model

# Optimizer
optimizer = tf.keras.optimizers.Adam(lr=5e-4)

# Define Losses
def mse_loss(y_true, y_pred):
    # # Create the slices for all the slider values and add them to a new tensor
    # slices = [(0,3),(24,28),(59,62),
    #           (62,64),(73,82),(91,94),
    #           (103,106),(115,119),(128,131),
    #           (140,143),(152,154),(179,180),(196,197)]
    # YTRUE = tf.concat([y_true[...,i[0]:i[1]] for i in slices],axis=2)
    # YPRED = tf.concat([y_pred[...,i[0]:i[1]] for i in slices],axis=2)
    # tf.print(Y,summarize=-1)
    r_loss = tf.math.reduce_mean(tf.square(y_true-y_pred),axis= [1,2])
    return 1000*r_loss

def cat_cross_entropy(y_true,y_pred):
    
    # TODO add all the bce viable vectors [0,0,0,0,1,0,0]
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

def custom_loss(y_true,y_pred):
    slider_loss = mse_loss(y_true,y_pred)
    onehot_loss = cat_cross_entropy(y_true,y_pred)
    return slider_loss + onehot_loss
    
#Training
@tf.function
def train_step(inp,gt,dec):
    with tf.GradientTape() as decoder:
        generated= dec(inp)
        generated = tf.cast(generated,tf.float64)
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
        # print (f'VALIDATION LOSS (MSE): {val_loss}')
    return decode

if __name__ == "__main__":
    decode = dec()
    # tf.saved_model.save(decode,"graph_saved_model")
    