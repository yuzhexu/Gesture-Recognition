import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=2)

offset = 20
imgsize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    # Create two blank images for displaying
    imgLeft = np.zeros((imgsize, imgsize, 3), np.uint8)
    imgRight = np.zeros((imgsize, imgsize, 3), np.uint8)

    if hands:
        for hand in hands:
            x, y, width, height = hand['bbox']
            x1, y1 = max(x - offset, 0), max(y - offset, 0)
            x2, y2 = min(x + width + offset, img.shape[1]), min(y + height + offset, img.shape[0])
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                continue

            # Update the corresponding window according to the type of hand
            if hand['type'] == 'Left':
                imgLeft = cv2.resize(imgCrop, (imgsize, imgsize))
            elif hand['type'] == 'Right':
                imgRight = cv2.resize(imgCrop, (imgsize, imgsize))

    # Show left and right hand windows
    cv2.imshow("Left Hand", imgLeft)
    cv2.imshow("Right Hand", imgRight)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Loading models
interpreter = tf.lite.Interpreter(model_path="ASL_Model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Assume img is an image captured from the camera
# Resize the image to match the model input
img_resized = cv2.resize(img, (512, 512)) # Resize according to model requirements

# Normalize pixel values
img_normalized = img_resized / 255.0 # Assume normalized to 0-1 range

# Preprocessed image
input_data = np.expand_dims(img_normalized, axis=0).astype(np.float32)

# Run model inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
# Get the model output
output_data = interpreter.get_tensor(output_details[0]['index'])