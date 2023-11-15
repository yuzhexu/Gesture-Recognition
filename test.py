import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # ID number of our webcam is 0
detector = HandDetector(maxHands=1)  # Detection of 1 hand at a time
classifier = Classifier("C:\\Users\harsh\PycharmProjects\GestureDetection\Model\keras_model.h5",
                        "C:\\Users\harsh\PycharmProjects\GestureDetection\Model\labels.txt")

offset = 20  # Offset to have more pixels for clear identification of data
imgsize = 300  # bound size for the image
counter = 0  # for calculating no of images captured
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z']

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)



        else:
            const = imgsize / width
            heightCal = math.ceil(const * height)
            imgResize = cv2.resize(imgCrop, (imgsize, heightCal))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgsize - heightCal) / 2)
            imgWhite[heightGap: heightCal + heightGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x, y), (x + width, y + height), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)  # Delay of 1ms in capturing image
