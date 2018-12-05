import open3d
import numpy as np
import BTL_Register
import BTL_DataConvert

# Algorithm:
# 1. 3D cost function -- 3D grid with the multimodality image
# 2. 3D imwarp function -- faster way
# 3. 3D optimization -- still the same
# 4. Test for success matching -- visualization and others
# 5. Test code for each function

# Ref: https://bic-berkeley.github.io/psych-214-fall-2016/resampling_with_ndimage.html   -- tutorial about the figure
# Ref:

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
import nibabel as nib
from rotations import *

def MutualInf_Img(obj_A, obj_B):

    # Get the histogram image
    Hist_A = obj_A.ravel()
    Hist_B = obj_B.ravel()

    # The joint histogram
    Hist_2d = JointHist(Hist_A, Hist_B)
    pxy = Hist_2d / float(np.sum(Hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def JointHist(Hist_A, Hist_B):
    # Read the one dimensional histogram
    # fig, axes = plt.subplots(1, 2)

    # The hist for object A
    # hist_A = np.histogram(Hist_A.ravel(), bins = len(Hist_A))
    # axes[0].hist(Hist_A, bins = 20)
    # axes[0].set_title('T1 slice histogram')

    # The hist for object B
    # hist_B = np.histogram(Hist_B.ravel(), bins = len(Hist_B))
    # axes[1].hist(Hist_B, bins = 20)
    # axes[1].set_title('T2 slice histogram')
    # plt.show()

    # Joint histogram -- 2D
    hist_2d, x_edges, y_edges = np.histogram2d(Hist_A.ravel(), Hist_B.ravel(), 20)
    # hist_2d.T[0][0] = 0
    # plt.imshow(hist_2d.T, origin='lower')
    # plt.show()

    return hist_2d

# Load the data
np.set_printoptions(precision=4)
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
img = nib.load('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/ds107_sub012_t1r2.nii')
data = img.get_data()

M_x = x_rotmat(0)  # radius 1.57 = 180 degrees
M_y = y_rotmat(0)
M_z = z_rotmat(0)
M = M_x * M_y * M_z
translation = [0, 0, 0]  # voxel

I = data[..., 0]  # I is the first volume
I_new = affine_transform(I, M, translation, output_shape = (65, 100, 100), order = 1)
# print("The object of data is ", data)
print("The shape of the data is ", data.shape)
print("The shape of I is ", I.shape)

# Apply the transform to an original volume
M_x = x_rotmat(0)
M_y = y_rotmat(0)
M_z = z_rotmat(0)

M = M_x * M_y * M_z

print("The transformation of M is ", M)

translation = [1, 2, 3]
# K = affine_transform(I, M, translation, order=1)
# 3D interpolation -- if possible -- with scipy
# K = affine_transform(I, M, translation, order=1)
# Shape determined by the original image
K = affine_transform(I_new, M_x, translation, output_shape = (65, 100, 100), order=1)
print("The output of shape K is ", K.shape)
plt.imshow(K[:, :, 20])
plt.show()

# The mutual information with two 3D images
obj_A = I
obj_B = K
Hist_A = obj_A.ravel()
Hist_B = obj_A.ravel()
print("The histogram of A is ", Hist_A)
print("The shape of the histogram of A is ", Hist_A.shape)

MI = MutualInf_Img(obj_A, obj_B)

print("The mutual information between object A and B is ", MI)

# The joint histogram of the two objects
hist_2d, x_edges, y_edges = np.histogram2d(Hist_A.ravel(), Hist_B.ravel(), 20)
hist_2d.T[0][0] = 0
plt.imshow(hist_2d.T, origin = 'lower')
plt.show()

# Show the slices


