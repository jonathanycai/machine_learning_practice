import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# exploratory data analysis
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale pixel values to a range of 0 to 1 before feeding them to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

# display images of our training set and their labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# set up layers of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # transforms the format of the images from a 2d-array (of 28 by 28 pixels)
    keras.layers.Dense(128, activation='relu'), # densely connected, or fully connected, neural layers
    keras.layers.Dense(10) # 10-node softmax layer, returns an array of 10 probability scores that sum to 1
])

# compile the model
model.compile(optimizer='adam'
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures how well the model is doing during training
              metrics=['accuracy'])