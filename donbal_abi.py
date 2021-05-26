import numpy as np
import cv2 as cv
import random as rd

cap = cv.VideoCapture(0)
u = 1
x = y = 0
z = 0
file = 'image' + str(rd.randint(0, 100000)) + '.jpg'
while cap.isOpened():
    if u == 1:
        p = 1
        u = 2
    else:
        p = 2
        u = 1
    _, frame = cap.read()
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_range = np.array([100, 150, 150])
    upper_range = np.array([130, 255, 255])
    mask = cv.inRange(frame_hsv, lower_range, upper_range)
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    mask_eroion = cv.erode(mask, kernel, iterations=2)
    contours, _ = cv.findContours(mask_eroion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv.contourArea, reverse=True)
    cv.drawContours(frame, contours_sorted, 0, (0, 100, 255), thickness=3)
    if len(contours_sorted) != 0:
        (x, y), _ = cv.minEnclosingCircle(contours_sorted[0])
        cv.namedWindow('vebcam', cv.WINDOW_NORMAL)
    x, y = int(x), int(y)
    if x == 0 & y == 0:
        cv.imshow('vebcam', frame)
        car = cv.waitKey(5)
        if car == ord('q'):
            break
        elif car == ord('c'):
            u = 1
            x = y = z = 0
            continue
        elif car == ord('s'):
            cv.imwrite(file, frame)
    if p == 1:
        x1, y1 = x, y
    if z == 1:
        numpy = [[x1, y1], [x, y]]
        # pts = np.array([[x1, y1], [x, y]])
        z = 2
    elif z == 0:
        z = 1
    else:
        # if x != 0 & y != 0: numpy.append([x, y])
        numpy.append([x, y])
        try:
            numpy.remove([0, 0])
            numpy.remove([0, 0])
        except:
            pass
        # pts = np.array([[x1, y1], [x, y]]) + pts
        pts = np.array(numpy)
        cv.polylines(frame, [pts], False, (0, 100, 255), thickness=5)

    cv.imshow('eroion', mask_eroion)
    cv.imshow('inrange', mask)
    cv.imshow('vebcam', frame)
    car = cv.waitKey(5)
    if car == ord('q'):
        break
    elif car == ord('c'):
        u = 1
        x = y = z = 0
        continue
    elif car == ord('s'):
        cv.imwrite(file, frame)
# print('mask', mask.shape, 'mask_eroion', mask_eroion.shape, 'vebcam', frame.shape)

cap.release()
cv.destroyAllWindows()
