import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[22])
plt.show()
print(x_train[22])
print(y_train[22])