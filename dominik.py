from skimage import morphology
import numpy as np
import cv2

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def calculate_points(im):
    x1, x2, x3, x4, y1, y2, y3, y4 = -1, -1, -1, -1, -1, -1, -1, -1
    height, width = im.shape

    for x in range(height):
        if (x1 != -1 and y1 != -1):
            break
        for y in range(width):
            # print(im[x, y])
            if (im[x, y] == 255):
                x1, y1 = x, y
                break
    for x in range(height - 1, 0, -1):
        if (x2 != -1 and y2 != -1):
            break
        for y in range(width - 1, 0, -1):
            if (im[x, y] == 255):
                x2, y2 = x, y
                break

    for y in range(width):
        if (x3 != -1 and y3 != -1):
            break
        for x in range(height):
            if (im[x, y] == 255):
                x3, y3 = x, y
                break

    for y in range(width - 1, 0, -1):
        if (x4 != -1 and y4 != -1):
            break
        for x in range(height - 1, 0, -1):
            if (im[x, y] == 255):
                x4, y4 = x, y
                break

    pts = np.array([(y1, x1), (y2, x2), (y3, x3), (y4, x4)], dtype="float32")

    return pts

def pers_transform(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    im_bw = cv2.bitwise_not(image)
    thresh, im_bw = cv2.threshold(im_bw, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    im_bw = morphology.erosion(im_bw)

    pts = calculate_points(im_bw)

    warped = four_point_transform(image, pts)

    cv2.imshow("Original", image)
    cv2.waitKey(0)
    cv2.imwrite('temp.jpg', warped)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)


pers_transform('data/sud3.jpg')