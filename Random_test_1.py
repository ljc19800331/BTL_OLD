
# %reload_ext signature
# %matplotlib inline

# Ref: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
# Ref: http://nbviewer.jupyter.org/github/empet/Plotly-plots/blob/master/Plotly-Slice-in-volumetric-data.ipynb  -- show the slice
# Ref: https://github.com/matplotlib/matplotlib/issues/3919

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from BTL_Register import *

def plot_3D_array_slices(obj_A, obj_B):

    n_x, n_y, n_z = obj_A.shape
    Img_test_A = obj_A[:, :, n_z // 2]
    Img_test_B = obj_B[:, :, n_z // 2]

    ShowPair(Img_test_A, Img_test_B)

    # Input is a 3D array
    # min_val = obj_A.min()
    # max_val = obj_A.max()
    # n_x, n_y, n_z = obj_A.shape
    # colormap = plt.cm.YlOrRd
    #
    # fig = plt.figure()
    #
    # x_cut = obj_A[n_x//2, :, :]
    # Y, Z = np.mgrid[0:n_y, 0:n_z]
    # X = n_x//2 * np.ones((n_y, n_z))
    #
    # plt.subplot(2, 2, 1)
    # plt.imshow(X)
    #
    # y_cut = obj_A[:, n_y // 2, :]
    # plt.subplot(2, 2, 2)
    # plt.imshow(y_cut)
    #
    # z_cut = obj_A[:, :, n_z//2]
    # plt.subplot(2, 2, 3)
    # plt.imshow(z_cut)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = colormap((x_cut-min_val)/(max_val-min_val)), shade=False)
    # ax.set_title("x slice")

    # y_cut = obj_A[:, n_y//2, :]
    # X, Z = np.mgrid[0:n_x, 0:n_z]
    # Y = n_y//2 * np.ones((n_x, n_z))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colormap((y_cut-min_val)/(max_val-min_val)), shade=False)
    # ax.set_title("y slice")
    #
    # z_cut = obj_A[:, :, n_z//2]
    # X, Y = np.mgrid[0:n_x, 0:n_y]
    # Z = n_z//2 * np.ones((n_x, n_y))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colormap((z_cut-min_val)/(max_val-min_val)), shade=False)
    # ax.set_title("z slice")

    plt.show()

# Read the .mat file from Matlab
data_A = sio.loadmat('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/obj_A.mat')
obj_A = data_A['obj_A']

data_B = sio.loadmat('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/obj_B.mat')
obj_B = data_B['obj_B']

x_res = [5.0050e-01, 5.5381e-01, -2.3367e-04, 6.5032e+00, 4.3415e+00, 9.1080e-07]
x_res = [0, 0, 0, 6.5032e+00, 4.3415e+00, 9.1080e-07]
x_res = [0, 0, 0, 6.5032e+00, 4.3415e+00, 9.1080e-07]

affine_paras = x_res
rotx = affine_paras[0]
roty = affine_paras[1]
rotz = affine_paras[2]
tx = affine_paras[3]
ty = affine_paras[4]
tz = affine_paras[5]

M_x = x_rotmat(rotx)
M_y = y_rotmat(roty)
M_z = z_rotmat(rotz)
M = M_x * M_y * M_z
translation = [tx, ty, tz]

print(M)

Img_out = affine_transform(obj_B, M_y, translation, output_shape = obj_B.shape, order = 1)

obj_C = TransformImg3D(obj_B, x_res)

n_x, n_y, n_z = obj_A.shape
Img_test_A = obj_A[:, :, n_z // 2]
Img_test_B = obj_B[:, :, n_z // 2]
Img_test_C = obj_C[:, :, n_z // 2]

plt.figure(1)
plt.subplot(131)
plt.imshow(Img_test_A)
plt.subplot(132)
plt.imshow(Img_test_B)
plt.subplot(133)
plt.imshow(Img_test_C)
plt.show()

# plot_3D_array_slices(obj_A, obj_B)
# plot_3D_array_slices(obj_A, obj_C)

translation = [10, 10, 0]
[ 9.9789e+00  9.9048e+00 -2.0859e-03]

# The correct MI is  1.1687880838433808
# The histogram of A is  [0.3053 0.3928 0.3839 ... 0.3186 0.2815 0.2147]
# The shape of the histogram of A is  (8675289,)
# The mutual information between object A and B is  0.41028154221425894
# The current cost function is  -0.41028154221425894
# The current cost function is  -1.095636484556351
# The current cost function is  -1.099365297374111
# The current cost function is  -1.102627095715212
# The current cost function is  -1.106053243614103
# The current cost function is  -1.1060629654035543
# The current cost function is  -1.1060627456798202
# The current cost function is  -1.1060632767371077
# The final result is  [ 9.9991  9.888  -9.9989]
#
# The shape of the dictionary is  (729, 7)
# The shape of the dictionary is  (729, 7)
# The correct MI is  1.1687880838433808
# The histogram of A is  [0.3053 0.3928 0.3839 ... 0.3186 0.2815 0.2147]
# The shape of the histogram of A is  (8675289,)
# The mutual information between object A and B is  0.4712318039841886
# The current cost function is  -0.4712318039841886
# The current cost function is  -0.6601039600825638
# The current cost function is  -0.6601069805491233
# The current cost function is  -0.6601068612346884
# The current cost function is  -0.6601065580044513
# The current cost function is  -0.6601065565979456
# The final result is  [ 7.6542  5.5481 -0.0655]