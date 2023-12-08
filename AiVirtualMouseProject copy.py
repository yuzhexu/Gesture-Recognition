import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()
print(wScr, hScr)
lastClickTime = 0  # 用于双击功能的时间记录
rightClickFlag = False  # 用于控制右键点击的标志

try:
    while True:
        # 1. Find hand Landmarks
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        # 2. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]  # Index Finger
            x2, y2 = lmList[12][1:]  # Middle Finger
        performRightClick = False

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # 在访问 fingers 列表之前检查其长度
        if len(fingers) ==5:
            if sum(fingers) == 0:
                continue  # 跳过循环的剩余部分
            # 4. Moving Mode: Index and Middle Finger are up
            if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                # Convert Coordinates and Move Mouse
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                pyautogui.moveTo(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY
                length, _, _ = detector.findDistance(8, 12, img)
                if length < 40:  # Threshold for fingers being close
                    if not isDoubleClick:
                        lastClickTime = time.time()
                        isDoubleClick = True
                    else:
                        if time.time() - lastClickTime < 0.3:
                            pyautogui.doubleClick()
                            isDoubleClick = False
                else:
                    isDoubleClick = False
            # 5. Left Click Mode: Only Index Finger is up
            elif fingers[1] == 1 and fingers[2] == 0:
                pyautogui.click()

            # 6. Right Click Mode: Only Middle Finger is up
            elif fingers[1] == 0 and fingers[2] == 1 and sum(fingers) == 1:
                performRightClick = True

            # 7. Drag Mode: Both Index and Middle Fingers are bent
            elif fingers[1] == 0 and fingers[2] == 0:
                pyautogui.mouseDown()
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                pyautogui.moveTo(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY
            else:
                pyautogui.mouseUp()
        if performRightClick and not rightClickPerformed:
            pyautogui.rightClick()
            rightClickPerformed = True
        elif not performRightClick:
            rightClickPerformed = False


            # 9. Frame Rate
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
            (255, 0, 0), 3)

            # 10. Display
            cv2.imshow("Image", img)
            cv2.waitKey(1)

except KeyboardInterrupt:
    print("Interrupted by user, closing...")
    cap.release()
    cv2.destroyAllWindows()