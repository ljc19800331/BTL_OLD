
# This code targets at the realsense

import cv2
import vtk
import numpy as np

def test_region():

    # Read the image
    color_image = cv2.imread('/home/maguangshen/PycharmProjects/BTL/open5d/data/color.png')