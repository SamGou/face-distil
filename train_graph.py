from model.model_graph import train, normalise
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
testDS = dataset.take(int(DATA_LENGTH*0.2))
# valDS = valDS.skip(2) # Val set
# testDS = valDS.take(2) # Test set
trainDS = dataset.skip(int(DATA_LENGTH*0.2))  # 70 % training
# trainDS = trainDS.take(2)
print("\n\nTRAIN DATASET LENGTH: ", trainDS.cardinality().numpy())

# Set Input pipeline
trainDS_batched = (trainDS
           .shuffle(1024)
           .cache()
           .repeat(1)
           .batch(64)
           .prefetch(AUTOTUNE))

decode = train(trainDS_batched, 10000)

def predict(dataset):
    diff_mu = []
    diff_sig = []
    
    def _percentage_predict(x):
        x = tf.abs(x)
        sum_axis = tf.reshape(tf.reduce_sum(x,axis = 1),shape=(x.shape[0],1))
        return tf.divide(x,sum_axis)

    def _round(x,base=0.1):
        return base * tf.round(x/base)
    
    def _mix_post_process(x, base=.05,threshold= 0.04):
        ex = tf.where(x>=0.,x,0.0)
        sums = tf.reduce_sum(ex,axis=1)
        sums = tf.where(sums>1., sums, 1.)
        ex  = tf.divide(ex, sums)
        ex = tf.where(ex>=threshold,ex,0.0)
        ex = _round(ex,base)
        return ex
    
    def _slider_post_process(x):
        slices_round10 = [(22,24),(24,28),(59,62),(62,64),
              (73,82),(91,94),(103,106),(115,119),(128,131),
              (140,143),(152,154)]
        slices_round5 = [(179,180),(196,197)]
        for idx1,idx2 in slices_round10:
            x[...,idx1:idx2] = _round(x[...,idx1:idx2],base=0.1)
        for idx1,idx2 in slices_round5:
            x[...,idx1:idx2] = _round(x[...,idx1:idx2],base=0.05)
        return x
            
    for inp, gt in dataset:
        prediction = decode.predict(inp)
        slices_onehot = [(4,13),(13,22),(49,59),(37,49),(163,179),(180,196)]
        
        slices_mix = [(28,37),(64,73),(82,91),(94,103),(106,115),
                      (119,128),(131,140),(143,152),(154,163)]
        
        # One hot & mix & sliders post-process
        prediction = _slider_post_process(prediction)
        for idx1,idx2 in slices_onehot:
            prediction[...,idx1:idx2] = _percentage_predict(prediction[...,idx1:idx2])
        for idx1,idx2 in slices_mix:
            prediction[...,idx1:idx2] = _mix_post_process(prediction[...,idx1:idx2])
        # gender post-process
        prediction[...,3:4] = normalise(prediction[...,3:4])
        
        diff = prediction-gt
        prediction = np.round(prediction, 2)
        tf.print("PRED",prediction, summarize=-1)
        tf.print("GT",np.round(gt, 2), summarize=-1) 
        diff_mu.append(np.mean(diff))
        diff_sig.append(np.std(diff))
    
    print("PREDICTIONS MEAN SQ ERROR %2f +/- %2f" % (np.mean(np.square(diff_mu)), np.mean(diff_sig)))
    
print("\n\nTEST DATASET LENGTH: ", testDS.cardinality().numpy())
prediction = predict(testDS.take(2))
# tf.saved_model.save(decode, "./trained-model")
