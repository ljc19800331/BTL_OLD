'''
ref: https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/  -- video capture vision
'''

import numpy as np
import cv2
from mss import mss
from PIL import Image
import matplotlib.pyplot as plt
import time

class IV_USVision():

    def __init__(self):

        self.path = 1
        self.img_path = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data'
        self.sct = mss()                        # Define the screenshot number
        self.monitor = self.sct.monitors[1]     # Define the monitor number

        left = self.monitor["left"] + self.monitor["width"] * 5 // 100  # 5% from the left
        top = self.monitor["top"] + self.monitor["height"] * 5 // 100   # 5% from the top
        right = left + 1000                     # 400px width
        lower = top + 800                       # 400px height
        self.bbox = (left, top, right, lower)   # Define the bounding box

    def ScreenShot(self):

        sct = self.sct
        monitor = self.monitor
        sct_img = sct.grab(self.bbox)
        img = np.array(sct_img)
        img_test = img[:, :, 0:3]

        # Check the fourth dimension
        D_four = img[:, :, 2]
        print("The fourth dimension data is ", D_four)

        # Show the image
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img_test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def RealtimeShot(self):

        sct = self.sct
        monitor = self.monitor

        while 1:
            sct_img = sct.grab(self.bbox)
            img = np.asarray(sct_img)
            print(img.shape)
            cv2.imshow('screen', np.array(img))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    def VideoCapture(self):

        # Define the basic parameters
        sct = self.sct
        monitor = self.monitor
        frame_array = []

        frame_flag = 0
        # print(frame_flag)
        while frame_flag < 1000:
            sct_img = sct.grab(self.bbox)
            img = np.asarray(sct_img)
            img = img[:,:,0:3]
            height, width, layers = img.shape
            size = (width, height)
            print(img)
            # cv2.imshow('screen', np.array(img))
            frame_array.append(img)
            frame_flag += 1

        PathOut = '/home/mgs/PycharmProjects/BTL_GS/BTL/video.avi'
        fps = 30
        # cv2.VideoWriter_fourcc(*'DIVX')
        # cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(PathOut, cv2.VideoWriter_fourcc(*'XVID'), fps, size)

        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()

    def ScreenRecordEfficient(self):

        # 800x600 windowed mode
        mon = {"top": 40, "left": 0, "width": 800, "height": 640}

        title = "[MSS] FPS benchmark"
        fps = 0
        sct = self.sct
        last_time = time.time()

        while time.time() - last_time < 1:
            img = np.asarray(sct.grab(mon))
            fps += 1

            cv2.imshow(title, img)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

        return fps

if __name__ == "__main__":

    test = IV_USVision()
    # test.VideoCapture()
    # test.ScreenShot()
    # fps = test.ScreenRecordEfficient()
    # print(fps)      # The fps can reached to 33 -- solved the problem
    test.RealtimeShot()