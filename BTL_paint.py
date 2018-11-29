# Draw the region manually in the image -- with human loop
# Draw a rectangular or circular region

import numpy as np
import cv2
import matplotlib.pyplot as plt

global ix, iy, drawing, mode, P_circle
drawing = False
mode = True
ix, iy = -1, -1
P_ROI = []
P_circle = np.zeros((1,2))

# Mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, P_circle

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing == True:

            if mode == True:

                P_ROI.append([x, y])

                # print(P_ROI)
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

                # Get the pixel coordinates within the circle radius
                r = 5
                x_center = x
                y_center = y

                idx = np.where( np.sqrt(np.square(xv - x) + np.square(yv - y)) <= r )

                idx_x = idx[0]
                idx_y = idx[1]

                pixel_circle = np.zeros((len(idx_x), 2))
                pixel_circle[:, 0] = idx_x
                pixel_circle[:, 1] = idx_y
                print(pixel_circle)

                temp = P_circle
                P_circle = np.append(temp, pixel_circle, axis = 0)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

img = cv2.imread('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Brain_cortex.jpg', 1)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

[h, w] = img_grey.shape

# Generate the pixel grid
x = np.linspace(0, h-1, h)
y = np.linspace(0, w-1, w)
xv, yv = np.meshgrid(y, x)

img_mask = np.zeros((img_grey.shape))
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == 27:
        break

# Mask (of course replace corners with yours)
points = np.array([P_ROI], dtype = np.int32)
points_circle = P_circle

pixel_use = np.array(points_circle, dtype=np.int32)
pixel_use = pixel_use[1:,:]         # Delete the first value
points = pixel_use

h = img.shape[0]
w = img.shape[1]

mask = np.zeros(img_grey.shape, dtype = np.uint8)

for item in pixel_use:
    mask[item[0], item[1]] = 255

# Apply the mask
masked_image = cv2.bitwise_and(img_grey, mask)

# Return the mask indices
mask_idx = np.where(masked_image != 0)
x_roi = mask_idx[1]
y_roi = mask_idx[0]
ROI = np.zeros([len(x_roi), 2])
ROI[:,0] = x_roi
ROI[:,1] = y_roi
ROI = np.array(ROI, dtype = np.int32)

print(ROI)
cv2.imwrite('/home/mgs/PycharmProjects/BTL_GS/BTL_Data/Brain_texture.jpg', masked_image)
#　cv2.imwrite('C:/Users/gm143/Documents/MATLAB/BTL/Data/exp_1/pos_12/before/mask.png', mask)

cv2.imshow('masked image', masked_image)
cv2.waitKey()
cv2.destroyAllWindows()