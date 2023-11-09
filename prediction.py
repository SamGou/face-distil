import tensorflow as tf
import numpy as np
from deepface import DeepFace
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

#Load model
model = tf.saved_model.load("./trained-model")
def predict_image(model,image_emb):
    def _percentage_predict(x):
        x = tf.abs(x)
        sum_axis = tf.reshape(tf.reduce_sum(x,axis = 1),shape=(x.shape[0],1))
        return tf.divide(x,sum_axis)

    prediction = model(image_emb)[0].numpy()
    slices = [(4,13),(13,22),(49,59),(37,49),(163,179),(180,196)]
    for idx1,idx2 in slices:
        prediction[...,idx1:idx2] = _percentage_predict(prediction[...,idx1:idx2])

    prediction = np.round(prediction, 2)
    print("PRED",prediction,prediction.shape)
    
predict_image(model,embedding)