import tensorflow as tf
import tensorflow_datasets as tfds 
tfds.disable_progress_bar()
tf.config.list_physical_devices('GPU')

import math
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    #change the type of the image to tf.float32
    images /= 255
    return images, labels

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
# each dataset contains a set of images and their corersponding labels 

class_names = metadata.features['label'].names
print("Class names: {}".format(class_names))

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)
# althought normalize returns 2 values, train_dataset contains to objects which are then returned by normalize

train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()
#caching the data in memeory makes training faster supposedly

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    #first convolution layer, utilzies padding so image remains as 28x28
    #creates 32 output images from 1 input image
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    #maxpooled down to a 14x14 image
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),   #the new input image is now a 14x14 image, creating 64 
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    #maxpooled down to a 7x7 image
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))