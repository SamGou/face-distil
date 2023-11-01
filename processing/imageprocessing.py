import cv2 as cv
import imutils

im = cv.imread('data/ingame_images/0.png')
im = imutils.resize(im,width = 256)
cv.imwrite("data/ingame_images/1_resized.png",im)