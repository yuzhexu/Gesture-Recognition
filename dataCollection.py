import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # ID number of our webcam is 0
detector = HandDetector(maxHands=1)  # Detection of 1 hand at a time

offset = 20  # Offset to have more pixels for clear identification of data
imgsize = 300  # bound size for the image
counter = 0  # for calculating no of images captured
folder = "C:\\Users\harsh\PycharmProjects\GestureDetection\Data\\2"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, width, height = hand['bbox']  # bounding box
        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255  # multiply by 255 to make image white
        imgCrop = img[y - offset:y + height + offset, x - offset:x + width + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = height / width

        if aspectRatio > 1:
            const = imgsize / height
            widthCal = math.ceil(const * width)
            imgResize = cv2.resize(imgCrop, (widthCal, imgsize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgsize - widthCal) / 2)
            imgWhite[:, widthGap: widthCal + widthGap] = imgResize

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
    key = cv2.waitKey(1)  # Delay of 1ms in capturing image

    # Saving images for data collection with "s"

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)
