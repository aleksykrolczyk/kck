import cv2
import numpy as np
from skimage import morphology
import tensorflow as tf

SIZE = (28, 28)

def show(img):
    cv2.imshow("", img)
    cv2.waitKey(0)

def sort_contours(cnts, method="left-to-right"):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i]))
    return (cnts, boundingBoxes)


def extract(img_for_box_extraction_path, cropped_dir_path):
    print('a')
    base_img = cv2.imread(img_for_box_extraction_path, 0)
    base_img = morphology.erosion(base_img)
    threshold, img = cv2.threshold(base_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = 255 - img
    print('a')
    kernel_length = np.array(img).shape[1] // 30
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    print('a')
    img_vertical = cv2.erode(img, vertical_kernel, iterations=3)
    img_vertical = cv2.dilate(img_vertical, vertical_kernel, iterations=3)
    print('a')
    img_horizontal = cv2.erode(img, horizontal_kernel, iterations=3)
    img_horizontal = cv2.dilate(img_horizontal, horizontal_kernel, iterations=3)
    print('a')
    img = cv2.addWeighted(img_vertical, 0.5, img_horizontal, 0.5, 0.0)
    img = cv2.erode(~img, kernel, iterations=2)
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print('a')
    img = cv2.erode(img, kernel=np.ones([6,6]), iterations=1)
    print('a')
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    print('a')
    idx = 0
    model = tf.keras.models.load_model('digits.model')
    print('a')
    def dilate_and_resize(img, steps=2):
        for i in range(steps):
            img = morphology.dilation(img)
        img = cv2.resize(img, SIZE)
        img = 255 - img
        img = img/255
        img = img.reshape(28, 28, 1)
        return img


    sudoku = []
    row = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)

        idx += 1
        new_img = base_img[y:y + h, x:x + w]

        new_img = dilate_and_resize(new_img)
        data = np.array([new_img])
        data = data.astype('float32')
        prediction = model.predict(data)
        row.insert(0, np.argmax(prediction))
        if (i+1) % 9 == 0:
            sudoku.append(row)
            row = []
    return np.array(sudoku).T.tolist()


sudoku = extract("result.jpg", "")
for row in sudoku:
    print(row)
