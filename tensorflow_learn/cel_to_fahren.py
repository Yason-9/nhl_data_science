import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float) #features (inputs to the model)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float) #labels (outpus of the model) 
#example, a pair of corresponding features and labels

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
# units = 1: # of neurons in the layer
# input_shape: sepcifies that the input to this layer is a single value 
#              a one-dimensional array with one member 
model = tf.keras.Sequential([l0])
# .Sequential takes a list of layers as an argument, specifying the calculations required for feature to label 

#alternate declaration 
#model = tf.keras.Sequential([
#  tf.keras.layers.Dense(units=1, input_shape=[1])
#])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
#compile the model with a loss and optimization function
#loss uses mse, ootpimizer uses "Adam", essentially a version of stochastic gradient descent 
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Training Complete")

plt.xlabel("Training Epoch")
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

print(model.predict([100.0]))
print("Function variable values; {}".format(l0.get_weights()))
#Call the Dense layer, and retrieve the variables trainined by the model 