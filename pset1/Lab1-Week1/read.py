import cv2 as cv

img = cv.imread('imgs/opencv_logo.png')
cv.imshow('opencv_logo', img)
cv.waitKey()