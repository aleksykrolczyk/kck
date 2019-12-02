from skimage import morphology
import numpy as np
import cv2


def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")


    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]


    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]


    return rect


def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect


    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))


    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))


    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")


    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def check_straight(corners,max_x,max_y,min_x,min_y):
    for i in corners:
        x, y = i.ravel()
        if x == max_x:
            return 1
        if x == min_x:
            return 1
        if y == min_y:
            return 1
        if y == max_y:
            return 1
    return 0


def get_corners(im):

    im = morphology.dilation(im)

    corners = cv2.goodFeaturesToTrack(im,100,0.08,10)
    max_x,max_y,min_x,min_y=0,0,10000,10000
    for i in corners:
        x, y = i.ravel()
        cv2.circle(im, (x, y), 3, 255, -1)
        if x >= max_x:
            max_x = x
            pt1 = (x, y)
        if x <= min_x:
            min_x = x
            pt2 = (x, y)
        if y <= min_y:
            min_y = y
            pt3 = (x, y)
        if y >= max_y:
            max_y = y
            pt4 = (x, y)

    if check_straight(corners,max_x,max_y,min_x,min_y)==1:
        return []

    cv2.imshow("",im)
    pts = np.array([pt1,pt2,pt3,pt4], dtype="float32")
    print(pts)
    return pts



    cv2.imshow('',im)

def preprocessing():
    pts = get_corners(image)

    if pts==[]:
        return image
    else:
        return four_point_transform(image,pts)


image = cv2.imread('test12.png', cv2.IMREAD_GRAYSCALE)
