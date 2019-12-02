import cv2
import numpy as np


def show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)


def extract(path):
    base_img = cv2.imread(path, 0)
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

    rects = []

    for contour in contours:
        rects.append(cv2.boundingRect(contour))

    if len(rects) != 81:
        print("UWAGA, wykryto " + len(rects) + " pól, nie 81")

    rects.sort(key=lambda x: x[1] * 9 + x[0])

    for r in rects:
        x, y, w, h = r

        new_img = base_img[y:y + h, x:x + w]
        show(new_img)


extract("s1.png")
