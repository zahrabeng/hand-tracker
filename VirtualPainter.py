import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

def main():
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector(detection_con=0.85)
    xp=0
    yp=0
    brushThickness = 15
    imgCanvas = np.zeros((400, 680, 3), np.uint8)

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # tip of index finger
            x1,y1=lmList[8][1:]
            # tip of middle finger
            x2,y2=lmList[12][1:]

            fingers = detector.fingersUp()
            if fingers[1] and fingers[2]==False:
                cv2.circle(img, (x1, y1), 15, (233, 37, 0), cv2.FILLED)
                if xp==0 and yp==0:
                    xp, yp = x1, y1
                cv2.line(img, (xp, yp), (x1, y1), (233, 37, 0), brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), (233, 37, 0), brushThickness)

                xp, yp = x1, y1

        cv2.imshow("Image", img)
        cv2.imshow("Canvas", imgCanvas)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()
