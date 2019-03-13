
# This idea aims at detecting the laesr spot automatically
# ref: https://github.com/andrewnagyeb/LaserPointerTracking/blob/master/track_laser.py

import cv2
import numpy as np

def Method_HSVthresh():

    cap = cv2.VideoCapture(1)

    while(1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([102,70,36])
        upper_red = np.array([143,255,255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # cv2.circle(frame, maxLoc, 20, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Track Laser', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def Method_Bright():

    cap = cv2.VideoCapture(1)
    pts = []

    while (1):
        # Take each frame for post processing
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        cv2.circle(frame, maxLoc, 5, (0, 0, 0), 2)
        cv2.imshow('Track Laser', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Test the max bright method
    Method_Bright()

    # Test the hsv threshold method
    # Method_HSVthresh()