'''
1. data preprocessing
    1. crop the images
    2. take the gradient
2. Find the vessel
    1. calculate the scores and find the centroids
    2. EKF if possible -- how to model the whole ellipse model -- tracking the contour of the vessel
3. Local histogram equalization
    1. http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html
'''

from PIL import Image
import os.path, sys

class IV_vessel():

    def __init__(self):
        self.img_readpath = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/IV_vessel_2/'
        self.img_savepath = '/home/mgs/PycharmProjects/BTL_GS/BTL_Data/IV_vessel_2_croped/'

    def ImgCrop(self):

        path = self.img_readpath
        dirs = os.listdir(path)
        print(dirs)

        for item in dirs:
            fullpath = os.path.join(path, item)
            if os.path.isfile(fullpath):
                im = Image.open(fullpath)
                f, e = os.path.splitext(fullpath)       # f: file name e: file extension -- .png etc
                imCrop = im.crop((453, 115, 953, 871))  # corrected
                imCrop.save(f + '_Cropped.png', "PNG", quality=100)
                # print(imCrop.shape)
                print(f)
                # print(e)

    def ImgGradient(self):

        # Basic preprocessing skills for the images
        a = 1

if __name__ == "__main__":

    test = IV_vessel()
    test.ImgCrop()