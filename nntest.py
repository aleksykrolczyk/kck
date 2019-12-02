# import tensorflow as tf
# import numpy as np
# from skimage import morphology
# import cv2
#
# SIZE = (28, 28)
#
#
# def erode_and_resize(img, steps=3):
#     for i in range(steps):
#         img = morphology.erosion(img)
#     img = cv2.resize(img, SIZE)
#     return img
#
#
# model = tf.keras.models.load_model('digits.model')
#
#
# imgs = []
# for i in range(10):
#     img = cv2.imread(f'digits_examples/{i}.jpg', cv2.IMREAD_GRAYSCALE)
#     img = erode_and_resize(img)
#     img = 255 - img
#     img = img.reshape(28, 28, 1)
#     imgs.append(img)
#
# imgs = np.asarray(imgs)
# imgs = imgs / 255
#
# predictions = model.predict(imgs)
#
# for i in range(10):
#     print(np.argmax(predictions[i]))
#

# model = tf.keras.models.load_model('digits.model')
