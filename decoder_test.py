from model.model import Decoder
from data_generation import DataUtils
from deepface import DeepFace
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np
import pprint
import matplotlib.pyplot as plt
data = DataUtils()
models = [
  "VGG-Face", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]
embedding_objs = DeepFace.represent(img_path = "data/ingame_images/0.png",model_name=models[0])
x_train = tf.convert_to_tensor(embedding_objs[0]["embedding"])
x_test = tf.convert_to_tensor(data.generateEncodedDataExample())
latent_shape = x_train.shape[0]
output_shape = x_test.shape[0]
x_train = tf.expand_dims(x_train,axis=0)
x_test = tf.expand_dims(x_test,axis=0)
autoencoder = Decoder(latent_shape=latent_shape,output_shape=output_shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
history = autoencoder.fit(x_train, x_test,
            epochs=100,
            shuffle=False,
            validation_data=(x_train, x_test))

decoded_vec = autoencoder.decoder(x_train).numpy()
