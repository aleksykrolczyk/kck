import cv2
import numpy as np
from skimage import morphology
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

SIZE = (28, 28)

def output(a):
    sys.stdout.write(str(a))

def print_field(field):
    if not field:
        output("No solution")
        return
    for i in range(9):
        for j in range(9):
            cell = field[i][j]
            if cell == 0 or isinstance(cell, set):
                output('.')
            else:
                output(cell)
            if (j + 1) % 3 == 0 and j < 8:
                output(' |')

            if j != 8:
                output(' ')
        output('\n')
        if (i + 1) % 3 == 0 and i < 8:
            output("- - - + - - - + - - -\n")


def findNextCellToFill(grid, i, j):
    for x in range(i, 9):
        for y in range(j, 9):
            if grid[x][y] == 0:
                return x, y
    for x in range(0, 9):
        for y in range(0, 9):
            if grid[x][y] == 0:
                return x, y
    return -1, -1


def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            secTopX, secTopY = 3 * (i // 3), 3 * (j // 3)
            for x in range(secTopX, secTopX + 3):
                for y in range(secTopY, secTopY + 3):
                    if grid[x][y] == e:
                        return False
            return True
    return False


def solveSudoku(grid, i=0, j=0):
    i, j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1, 10):
        if isValid(grid, i, j, e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            grid[i][j] = 0
    return False


def show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
  
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
    s = 0
    for i in corners:
        x, y = i.ravel()
        if x == max_x or x == min_x or y == min_y or y == max_y:
            s += 1
            print(s)
    if(s>4 or s<4):
        return 1
    return 0


def get_corners(im):
    im = morphology.dilation(im)
    
    corners = cv2.goodFeaturesToTrack(im,300,0.08,50)
    max_x,max_y,min_x,min_y=0,0,10000,10000
    for i in corners:
        x, y = i.ravel()
        #cv2.circle(im, (x, y), 3, 255, -1)
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
    cv2.imshow("",im)
    if check_straight(corners,max_x,max_y,min_x,min_y)==1:
        return []

    pts = np.array([pt1,pt2,pt3,pt4], dtype="float32")
    return pts


def preprocessing(image):
    pts = get_corners(image)
    if pts==[]:
        return image
    else:
        return four_point_transform(image,pts)

    
def extract(path):
    pre_img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
 
    base_img = preprocessing(pre_img)
    threshold, img = cv2.threshold(base_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bw = img
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
        print("UWAGA, wykryto " + str(len(rects)) + " pól, nie 81")

    rects.sort(key=lambda x: x[1] * 9 + x[0])

    model = tf.keras.models.load_model('digits10epochs.model')
    def dilate_and_resize(img, steps=2):
        for i in range(steps):
            img = morphology.dilation(img)
        img = cv2.resize(img, SIZE)
        img = 255 - img
        # img = ((img/255) ** 5) * 255
        print(img)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()
        img = img.reshape(28, 28, 1)

        return img

    sudoku = []
    row = []

    for i, r in enumerate(rects):
        x, y, w, h = r

        chunk = img_bw[y: y + h, x:x + w]
        show(chunk)
        new_img = dilate_and_resize(chunk, steps=0)
        data = np.array([new_img]).astype('float32')
        prediction = model.predict(data)
        print(np.argmax(prediction))
        row.append(np.argmax(prediction))
        if (i+1) % 9 == 0:
            sudoku.append(row)
            row = []
    return np.array(sudoku).T.tolist()

sudoku = extract("data/sud55.jpg")
print(sudoku)


solveSudoku(sudoku)

print_field(sudoku)

