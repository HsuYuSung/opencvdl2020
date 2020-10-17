from cv2 import cv2 as cv
import sys

img = cv.imread("Uncle_Roger.jpg")

if img is None:
    sys.exit("Could not read the image.")


cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("q"):
    cv.destroyWindow("Display window")
