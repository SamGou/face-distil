import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model

class Decoder(Model):
    def __init__(self,latent_shape,output_shape):
        super().__init__()

        self.decoder = tf.keras.models.Sequential([
            tf.keras.Input(shape=(None,latent_shape,)),
            layers.Dense(latent_shape,activation="relu"),
            layers.Dense(latent_shape//2,activation="relu"),
            layers.Dense(latent_shape//4,activation="relu"),
            layers.Dense(latent_shape//6,activation="relu"),
            layers.Dense(latent_shape//8,activation="relu"),
            layers.Dense(output_shape,activation="tanh")
        ])

        
    def call(self,inputs):
            x = self.decoder(inputs)
            return x

