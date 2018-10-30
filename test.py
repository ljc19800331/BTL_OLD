import numpy as np
import cv2
import sys
sys.path.insert(0,'C:/Users/gm143/TumorCNC_brainlab/BTL/')
sys.path.insert(0,'C:/Users/gm143/TumorCNC_brainlab/Data/NonrotateMap')
import BTL_PNP

# Load the data
test = BTL_PNP.PNP()
MTI_vtx, MTI_tex, colorvec = test.GetMtiTexColor()
Img_roi = test.BTL_Draw()

# Get the colorize point cloud
TARGET = np.zeros([len(Img_roi), 3])
COLOR = np.zeros([len(Img_roi), 3])
r = 2
for idx, item in enumerate(Img_roi):
    # print(idx)
    p1 = [item[0] - r, item[1] - r]
    p2 = [item[0] + r, item[1] + r]
    x1 = p1[0]; y1 = p1[1]
    x2 = p2[0]; y2 = p2[1]
    idx_use = np.where( (MTI_tex[:, 0] >= y1) & (MTI_tex[:, 0] <= y2) & (MTI_tex[:, 1] >= x1) & (MTI_tex[:, 1] <= x2)  )
    tex_use = MTI_tex[idx_use,:]
    vtx_use = MTI_vtx[idx_use,:]
    color_use = colorvec[idx_use,:]
    if len(color_use) == 0:
        COLOR[idx, :] = np.array([0, 0, 0])
    else:
        COLOR[idx, :] = np.mean(np.asarray(color_use, dtype = np.int32), axis = 1)
    TARGET[idx,:] = np.mean(np.asarray(vtx_use), axis = 1)

