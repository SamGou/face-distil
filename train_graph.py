from model.model_graph import train
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras import losses
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# set seed
tf.random.set_seed(2041999)

# set constants
GROUND_TRUTH_PATH = "data/ground_truth/"
INPUT_PATH = "data/embeddings"
DATA_JSON_PATH = "data/data.json"


# Loading the data
# Ground Truth load
ground_truth = np.load(os.path.join(GROUND_TRUTH_PATH, "GT.npy"))
ground_truth = np.expand_dims(ground_truth, 1)
DATA_LENGTH = len(ground_truth)  # get dataset length
gtDataset = tf.data.Dataset.from_tensor_slices(ground_truth)

# Input load
input = np.load(os.path.join(INPUT_PATH, "embeddings.npy"))
input = np.expand_dims(input, 1)
inputDataset = tf.data.Dataset.from_tensor_slices(input)

# Split
dataset = tf.data.Dataset.from_tensor_slices((input, ground_truth))
dataset = dataset.shuffle(1024)
# set apart 30% for test and validation
testDS = dataset.take(int(DATA_LENGTH*0.3))
# valDS = valDS.skip(2) # Val set
# testDS = valDS.take(2) # Test set
trainDS = dataset.skip(int(DATA_LENGTH*0.3))  # 70 % training
# trainDS = trainDS.take(2)
print("\n\nTRAIN DATASET LENGTH: ", trainDS.cardinality().numpy())

# Set Input pipeline
trainDS = (trainDS
           .shuffle(1024)
           .cache()
           .repeat(1)
           .batch(64)
           .prefetch(AUTOTUNE))

decode = train(trainDS, 10000)


def predict(dataset):
    for inp, gt in dataset:
        prediction = decode.predict(inp)
        prediction = np.round(prediction, 2)
        
        tf.print(prediction, summarize=-1)
        tf.print(np.round(gt, 2), summarize=-1)

def _percentage(x):
        x = tf.abs(x)
        sum_axis = tf.reshape(tf.reduce_sum(x,axis = 2),shape=(x.shape[0],x.shape[1],1))
        return tf.divide(x,sum_axis)
    
prediction = predict(testDS.take(1))

slices = [(4,13),(13,22),(49,59),(37,49),(163,179),(180,196)]
for idx1,idx2 in slices:
    prediction[...,idx1:idx2] = _percentage(prediction[...,idx1:idx2])
print(prediction)