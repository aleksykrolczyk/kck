import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

"""
Will create new nn model and save it in digits.model
"""

IN_SHAPE = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Getting rid of zeroes from both train and test sets
for i in range(len(x_train)):
    if y_train[i] == 0:
        xsize, xshape = x_train[i].size, x_train[i].shape
        x_train[i] = np.zeros(xsize).reshape(xshape)
        
for i in range(len(x_test)):
    if y_test[i] == 0:
        xsize, xshape = x_test[i].size, x_test[i].shape
        x_test[i] = np.zeros(xsize).reshape(xshape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train, x_test = x_train/255, x_test/255

model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=IN_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

print('DONE')
model.evaluate(x_test, y_test)

# In case you want to see metrics on test set
# loss, acc = model.evaluate(x_test, y_test)
# print(loss)
# print(acc)

model.summary()
model.save('digits.model')