# This function is used for the data preprocessing
# structure: standard format -- npy
# txt2vtk:
# vec2mat:
# mat2vec:
# npy2vtk:
# vtkAngle: transform the point sets based on the x,y,z angles
# This module includes all the format exchange

import numpy as np
from stl import mesh
import SEC_VIZ








if __name__ == '__main__':
    txt_x = '/home/maguangshen/PycharmProjects/BTL_GS/others/VIZ tool/Data/brain_x.txt'
    txt_y = '/home/maguangshen/PycharmProjects/BTL_GS/others/VIZ tool/Data/brain_y.txt'
    txt_z = '/home/maguangshen/PycharmProjects/BTL_GS/others/VIZ tool/Data/brain_z.txt'

    # test txt2vtk
    vtk_scan, scan_x, scan_y, scan_z = txt2vtk(txt_x, txt_y, txt_z)
    print(scan_x[1:2]); print('\n')
    print(scan_y[1:2]); print('\n')
    print(scan_z[1:2])
    SEC_VIZ.vtk_pc([vtk_scan])

    # test vec2mat

    # test npy2vtk

