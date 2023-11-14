import tensorflow as tf
import numpy as np
from deepface import DeepFace
from model.model_graph import normalise
from data_generation import DataGen
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# set seed
tf.random.set_seed(2041999)

# set constants
IMAGE_PATH = "data/test/jolie.jpg"
MODEL_PATH = "./trained-model"

# make embedding
embedding = DeepFace.represent("data/test/jolie.jpg",model_name="VGG-Face")
# embedding = np.expand_dims(embedding[0]["embedding"],axis=0)
embedding = np.reshape(embedding[0]["embedding"],(1,1,-1))
embedding = tf.convert_to_tensor(embedding,dtype=tf.float32)
dg = DataGen()
#Load model
model = tf.saved_model.load("./trained-model")
def predict_image(model,image_emb):
    def _percentage_predict(x):
        x = tf.abs(x)
        sum_axis = tf.reshape(tf.reduce_sum(x,axis = 1),shape=(x.shape[0],1))
        return tf.divide(x,sum_axis)

    def _mix_post_process(x, base=.05,threshold= 0.04):
        ex = tf.where(x>=0.,x,0.0)
        sums = tf.reduce_sum(ex,axis=1)
        sums = tf.where(sums>1., sums, 1.)
        ex  = tf.divide(ex, sums)
        ex = tf.where(ex>=threshold,ex,0.0)
        ex = base * tf.round(ex/base)
        return ex
    
    def _round(x,base=0.1):
        return base * tf.round(x/base)
    
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
    
    prediction = model(image_emb)[0].numpy()
    slices = [(4,13),(13,22),(49,59),(37,49),(163,179),(180,196)]
    slices_mix = [(28,37),(64,73),(82,91),(94,103),(106,115),
                  (119,128),(131,140),(143,152),(154,163)]
    
    prediction = _slider_post_process(prediction)
    for idx1,idx2 in slices:
        prediction[...,idx1:idx2] = _percentage_predict(prediction[...,idx1:idx2])
    for idx1,idx2 in slices_mix:
        prediction[...,idx1:idx2] = _mix_post_process(prediction[...,idx1:idx2])
    prediction[...,3:4] = normalise(prediction[...,3:4])
    
    prediction = np.round(prediction, 2)
    return prediction

    
prediction = predict_image(model,embedding)
print("PRED",prediction)
print("DECODED",dg.decodeOutput(prediction))