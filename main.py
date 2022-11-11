import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('./models/keras_model.h5', compile=False)
print(model.summary())

# Load the labels
class_names = open('./models/labels.txt', 'r').readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# image = Image.open('./data/test/angular_leaf_spot/angular_leaf_spot_test.0.jpg').convert('RGB')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./data/train",
  seed=123,
  image_size=size,
  batch_size=64)

corrections = 0
num_test_samples = 0

for image_batch, labels_batch in train_ds:
    #turn the image into a numpy array
    image_array = np.asarray(image_batch)
    labels_batch = np.asarray(labels_batch)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # run the inference
    prediction = model.predict(normalized_image_array)
    predictions = np.argmax(prediction, axis=1)
    corrections += (predictions == labels_batch).sum()
    num_test_samples += len(image_batch)

train_acc = (corrections / num_test_samples) * 100.0
print("total test acc:", train_acc, "%")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./data/test",
  seed=123,
  image_size=size,
  batch_size=64)

corrections = 0
num_test_samples = 0

for image_batch, labels_batch in test_ds:
    #turn the image into a numpy array
    image_array = np.asarray(image_batch)
    labels_batch = np.asarray(labels_batch)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # run the inference
    prediction = model.predict(normalized_image_array)
    predictions = np.argmax(prediction, axis=1)
    corrections += (predictions == labels_batch).sum()
    num_test_samples += len(image_batch)

test_acc = (corrections / num_test_samples) * 100.0
print("total test acc:", test_acc, "%")