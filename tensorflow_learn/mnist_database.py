import tensorflow as tf
import tensorflow_datasets as tfds 
tfds.disable_progress_bar()

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
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)
# althought normalize returns 2 values, train_dataset contains to objects which are then returned by normalize

train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()
#caching the data in memeory makes training faster supposedly 

#plt.figure(figsize=(10,10))
#for i, (image, label) in enumerate(train_dataset.take(25)):
#    image = image.numpy().reshape((28,28))
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid()
#    plt.imshow(image, cmap=plt.cm.binary)
#    plt.xlabel(class_names[label])
#plt.show()

'''
Begin to construct model 
'''

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #unpacks each image (28x28) into a 1d array of 784 pixels
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    #first hidden layer contains 128 neurons
    #each neuron recieves information from all 784 pixels and adjusted the weights and biases as required

    #tf.keras.layers.Dense(256, activation=tf.nn.relu),
    #added own keras layer

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    #final output layer contains 10 neurons reciving input from the 128 neurons, with each
    # of the 10 neurons representing each clothing type 
    # SOFTMAX: a probability function converting vectors of numbers into vector of probabilities
    # sum of all 10 neurons is equal to 1 (per a probability function)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
#metrics is used to calulcate how accurate teh model classifies the information 

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
#contiuosly shuffle dataset
# ensure that the order is randomized so model doesn't learn from the order of the dataset
#.batch(32) tells model.fit to use 32 images or a batach of 32 images at a time while adjusting weights and biases 
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=6, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps = math.ceil(num_test_examples/32))

for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

print(predictions[0])
print(np.argmax(predictions[0]))
#prints the weight with the highest value (highest confidence value)
