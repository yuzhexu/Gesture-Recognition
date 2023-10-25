import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)  # ID number of our webcam is 0
detector = HandDetector(maxHands=1)  # Detection of 1 hand at a time

offset = 20
imgsize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, width, height = hand['bbox']
        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+height+offset, x-offset:x+width+offset]

        imgCropShape = imgCrop.shape


        aspectRatio = height/width

        if aspectRatio>1:
            const = imgsize/height
            widthCal = math.ceil(const*width)
            imgResize = cv2.resize(imgCrop,(widthCal, imgsize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgsize-widthCal)/2)
            imgWhite[:,widthGap: widthCal + widthGap] = imgResize

        else:
            const = imgsize / width
            heightCal = math.ceil(const * height)
            imgResize = cv2.resize(imgCrop, (imgsize, heightCal))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgsize - heightCal) / 2)
            imgWhite[heightGap: heightCal + heightGap, :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    cv2.waitKey(1)         # Delay of 1ms in capturing image
