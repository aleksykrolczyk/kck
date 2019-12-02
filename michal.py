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
    base_img = cv2.imread(img_for_box_extraction_path, 0)
    base_img = morphology.erosion(base_img)
    threshold, img = cv2.threshold(base_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = 255 - img

    show(img)

    kernel_length = np.array(img).shape[1] // 30
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_vertical = cv2.erode(img, vertical_kernel, iterations=3)
    img_vertical = cv2.dilate(img_vertical, vertical_kernel, iterations=3)
    show(img_vertical)

    img_horizontal = cv2.erode(img, horizontal_kernel, iterations=3)
    img_horizontal = cv2.dilate(img_horizontal, horizontal_kernel, iterations=3)
    show(img_horizontal)

    img = cv2.addWeighted(img_vertical, 0.5, img_horizontal, 0.5, 0.0)
    img = cv2.erode(~img, kernel, iterations=2)
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img = cv2.erode(img, kernel=np.ones([6,6]), iterations=1)

    show(img)
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0

    model = tf.keras.models.load_model('digits.model')

    def dilate_and_resize(img, steps=2):
        for i in range(steps):
            img = morphology.dilation(img)
        img = cv2.resize(img, SIZE)
        img = 255 - img
        img = img/255
        img = img.reshape(28, 28, 1)
        return img

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        idx += 1
        new_img = base_img[y:y + h, x:x + w]

        new_img = dilate_and_resize(new_img)
        data = np.array([new_img])
        data = data.astype('float32')
        prediction = model.predict(data)
        print(np.argmax(prediction[0]))
        show(new_img)

#extract("sudoku.jpeg", "./Cropped/")
#extract("sudoku1.jpg", "./Cropped/")
#extract("sudoku2.jpg", "./Cropped/")
#extract("s1.png", "")
#extract("s2.png", "")
#extract("s3.png", "")
extract("result.jpg", "")
# extract("s3R.png", "")
