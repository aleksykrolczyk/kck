import tensorflow as tf
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import cv2

SIZE = (28, 28)


def erode_and_resize(img, steps=2):
    for i in range(steps):
        img = morphology.erosion(img)
    img = cv2.resize(img, SIZE)
    img = morphology.erosion(img)
    return img


model = tf.keras.models.load_model('digits.model')


imgs = []
for i in range(10):
    img = cv2.imread(f'digits_examples/{i}.jpg', cv2.IMREAD_GRAYSCALE)
    img = erode_and_resize(img)
    img = 255 - img
    plt.imshow(img)
    plt.show()
    img = img.reshape(28, 28, 1)
    imgs.append(img)

imgs = np.asarray(imgs)
imgs = imgs / 255

predictions = model.predict(imgs)

for i in range(10):
    print(np.argmax(predictions[i]))

