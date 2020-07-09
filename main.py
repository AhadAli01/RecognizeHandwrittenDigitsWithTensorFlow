#import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

#load the dataset into var called mnist
mnist = tf.keras.datasets.mnist
#10 or 20% split for testing while rest for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#"normalizing" the data or simply scaling the rgb grey value of 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#create the neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #from 2D to 1D
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu)) #two dense hidden layers
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)) #one for the digits

#to compile and optimize the model for accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#to train model
model.fit(x_train, y_train, epochs=5)

#finally evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: ", accuracy*100, "%")
print("Loss: ", loss)

#save model for future
model.save('digits.model')
