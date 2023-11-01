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
ground_truth = np.expand_dims(ground_truth[:,:4],1)
DATA_LENGTH = len(ground_truth) # get dataset length
gtSize = ground_truth[0].shape[1]
gtDataset = tf.data.Dataset.from_tensor_slices(ground_truth)

# Input load 
input = np.load(os.path.join(INPUT_PATH,"embeddings.npy"))
input = np.expand_dims(input,1)
inputSize = input[0].shape[1]
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
           .repeat()
           .batch(64)
           .prefetch(AUTOTUNE))

# Save best performing model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./checkpoints/checkpoint1",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model = Decoder(latent_shape=inputSize,output_shape=gtSize)
model.compile(optimizer='adam', loss=losses.MeanSquaredError())
history = model.fit(trainDS,
             epochs=100,
             steps_per_epoch=NUM_STEPS,
             validation_data=valDS)
            #  callbacks=[model_checkpoint_callback])
model.summary()
graph = pd.DataFrame(history.history).plot(figsize=(8,5))
plt.savefig("./results/Test1.png")

for inp, out in testDS:
    numpy_input = inp.numpy()
    numpy_output = out.numpy()
    predict = model(numpy_input).numpy()
    print("\nPREDICT\n\n", np.round(predict,2))
    print("\nGROUND TRUTH\n\n", numpy_output)

#tf.debugging.set_log_device_placement(True)
# if memory issues, attempt set_memory growth
