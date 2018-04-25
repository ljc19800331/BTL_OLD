# This is the function for object segmentation
# Segment the background color
# Watershred segmentation

import cv2
import numpy as np
import matplotlib.pyplot as plt

class sec_seg:

    def __init__(self):
        # imgpath = '/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/test_3.png'
        imgpath = []
        img = []

    def LoadImg(self):
        self.img = cv2.imread(self.imgpath)

    def ImgViz(self):
        plt.figure(1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))    # The imshow function is targeted for RGB images
        plt.show()

    def ImgThreshold(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
        print(self.imgpath)
        for i in xrange(6):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    def ImgAdaptThreshold(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        img = cv2.medianBlur(img,5)
        ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]
        for i in xrange(4):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    def ImgWaterThreshold(self):

        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        plt.imshow(thresh, cmap = 'gray')

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(gray, markers)
        gray[markers == -1] = [255,0,0]

        plt.imshow(sure_bg, cmap = 'gray')

if __name__ == '__main__':

    # Test load image, viz image
    imgpath = '/home/maguangshen/PycharmProjects/BTL_GS/Data/Data_img/IMG_3780.JPG'
    test = sec_seg()
    test.imgpath = imgpath
    img = test.LoadImg()
    # test.ImgViz()

    # Test threshold method
    test.ImgThreshold()

    # Test adaptive thresholding method
    test.ImgAdaptThreshold()

    # Test the Watershred threshold method
    test.ImgWaterThreshold()


