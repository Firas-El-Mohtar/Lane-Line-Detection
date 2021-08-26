import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

#read a video
cap = cv.VideoCapture(r'opencv\lane line\test2 (1).mp4')

while True:
    #read image
    #img=cv.imread(r'opencv\lane line\test_image.png')
    success, img = cap.read()
    #gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #blur
    blur = cv.GaussianBlur(gray, (3, 3), 1)

    #canney
    canney = cv.Canny(gray, 50, 150)

    #region of interest
    ##get the help of matiplot lib
    #plt.imshow(canney)
    #plt.show()
    #points(580,304), (270,700), (1013,700)
    #create a triangle
    ##blank canva
    canva = np.zeros_like(canney)
    ##points of interest
    p1, p2, p3 = [270, 700], [540, 304], [1013, 700]
    pts = np.array([p1, p2, p3])
    ##fill poly
    mask = cv.fillPoly(canva, [pts], 255)

    #bitwise_and
    roi = cv.bitwise_and(mask, canney)

    #hough transform (returns set of lines)
    lines = cv.HoughLinesP(roi, 1, math.radians(
        1), 100, minLineLength=20, maxLineGap=200)

    #print(lines)
    #draw the lines
    canva2 = np.zeros_like(img)
    ##for every line
    for line in lines:
        cv.line(canva2, (line[0, 0], line[0, 1]),
                (line[0, 2], line[0, 3]), (255, 0, 0), thickness=10)

    #display on real image
    img = cv.addWeighted(img, 0.5, canva2, 0.5, 0)

    cv.imshow("canney", canney)
    cv.imshow("canva", canva)
    cv.imshow("roi", roi)
    cv.imshow("canva2", canva2)
    cv.imshow("img", img)
    cv.waitKey(1)
