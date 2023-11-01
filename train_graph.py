from model.model_graph import train
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from model.model import Decoder
from tensorflow.keras import losses
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#set seed
tf.random.set_seed(2041999)

# set constants
GROUND_TRUTH_PATH = "data/ground_truth/"
INPUT_PATH = "data/embeddings"
DATA_JSON_PATH = "data/data.json"
BS = 10
NUM_STEPS= 10

#Loading the data
#Ground Truth load
ground_truth = np.load(os.path.join(GROUND_TRUTH_PATH,"GT.npy"))
ground_truth = np.expand_dims(ground_truth,1)
DATA_LENGTH = len(ground_truth) # get dataset length
gtDataset = tf.data.Dataset.from_tensor_slices(ground_truth)

# Input load 
input = np.load(os.path.join(INPUT_PATH,"embeddings.npy"))
input = np.expand_dims(input,1)
inputDataset = tf.data.Dataset.from_tensor_slices(input)

#Split
dataset = tf.data.Dataset.from_tensor_slices((input,ground_truth))
dataset = dataset.shuffle(1024)
valDS = dataset.take(int(DATA_LENGTH*0.3)) # set apart 30% for test and validation
valDS = valDS.skip(2) # Val set
testDS = valDS.take(2) # Test set
trainDS = dataset.skip(int(DATA_LENGTH*0.3)) # 70 % training
print("\n\nTRAIN DATASET LENGTH: ",trainDS.cardinality().numpy())

# Set Input pipeline
trainDS = (trainDS
           .shuffle(1024)
           .cache()
           .repeat(1)
           .batch(64)
           .prefetch(AUTOTUNE))

decode = train(trainDS,1000)

inp,gt = list(trainDS.take(1).as_numpy_iterator())[0]
prediction  = decode.predict(inp)

print(np.round(prediction,2))
print(gt)