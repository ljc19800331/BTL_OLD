import math
import numpy as np
import SEC_PRE
import SEC_VIZ

class sec_map:

    def __init__(self):
        # Basic parameters for the model
        self.scan = []
        self.brain = []
        self.model_x = 3.346  # inch
        self.model_y = 4.547  # inch
        self.model_z = 1.571  # inch
        self.res = 0.5  # The size of the scan box
        self.pixel_res = 100  # Change during the time
        self.step = 100  # inch

    def Txt2PtNpy(self, txt_x, txt_y, txt_z):

        # initialize
        a_x = []; a_y = []; a_z = []
        x = []; y = []; z= []

        # Load the data
        scan_L1x = open(txt_x, "r")
        for line in scan_L1x:
            a_x = line.split()
            x.append(float(a_x[0]))

        scan_L1y = open(txt_y, "r")
        for line in scan_L1y:
            a_y = line.split()
            y.append(float(a_y[0]))

        scan_L1z = open(txt_z, "r")
        for line in scan_L1z:
            a_z = line.split()
            z.append(float(a_z[0]))

        # Normalize the data
        data = x
        data_min = [min(x)] * len(x)
        x = [data - data_min for data, data_min in zip(data, data_min)]
        x = np.asanyarray(x)

        data = y
        data_min = [min(y)] * len(y)
        y = [data - data_min for data, data_min in zip(data, data_min)]
        y = np.asanyarray(y)

        data = z
        data_min = [min(z)] * len(z)
        z = [data - data_min for data, data_min in zip(data, data_min)]
        z = np.asanyarray(z)

        # close the txt files
        scan_L1x.close(); scan_L1y.close(); scan_L1z.close()

        # Save to npy
        npy_data = np.zeros((len(x), 3), dtype=np.float32)
        npy_data[:,0] = x
        npy_data[:,1] = y
        npy_data[:,2] = z

        # convert to the point cloud -- vtk format
        vtk_data = SEC_VIZ.VtkPointCloud()
        for k in range(len(x)):
            point = ([x[k], y[k], z[k]])
            #print(point)
            vtk_data.addPoint(point)

        return vtk_data, npy_data, x, y, z

    def pt2Len(self, npy_data, flag_type, flag_Len):

        pt_xyz = npy_data

        pt_xyz[:, 0] = pt_xyz[:, 0] - np.min(pt_xyz[:, 0])
        pt_xyz[:, 1] = pt_xyz[:, 1] - np.min(pt_xyz[:, 1])
        pt_xyz[:, 2] = pt_xyz[:, 2] - np.min(pt_xyz[:, 2])

        x_range = np.max(pt_xyz[:, 0]) - np.min(pt_xyz[:, 0])
        y_range = np.max(pt_xyz[:, 1]) - np.min(pt_xyz[:, 1])
        z_range = np.max(pt_xyz[:, 2]) - np.min(pt_xyz[:, 2])

        if flag_type == 'brain':

            if flag_Len == 'cm':
                x_len = self.model_x
                y_len = self.model_y
                z_len = self.model_z
                c_x = x_len / x_range
                c_y = y_len / y_range
                c_z = z_len / z_range
                pt_xyz[:, 0] = pt_xyz[:, 0] * c_x
                pt_xyz[:, 1] = pt_xyz[:, 1] * c_y
                pt_xyz[:, 2] = pt_xyz[:, 2] * c_z
            elif flag_Len == 'inch':
                x_len = self.model_x
                y_len = self.model_y
                z_len = self.model_z
                c_x = x_len / x_range
                c_y = y_len / y_range
                c_z = z_len / z_range
                pt_xyz[:, 0] = pt_xyz[:, 0] * c_x
                pt_xyz[:, 1] = pt_xyz[:, 1] * c_y
                pt_xyz[:, 2] = pt_xyz[:, 2] * c_z

        if flag_type == 'scan':

            if flag_Len == 'cm':
                c = 2.54
                pt_xyz[:, 0] = pt_xyz[:, 0] * c
                pt_xyz[:, 1] = pt_xyz[:, 1] * c
                pt_xyz[:, 2] = pt_xyz[:, 2] * c

            elif flag_Len == 'inch':
                c = 1
                pt_xyz[:, 0] = pt_xyz[:, 0] * c
                pt_xyz[:, 1] = pt_xyz[:, 1] * c
                pt_xyz[:, 2] = pt_xyz[:, 2] * c

        return pt_xyz

if __name__ == '__main__':

    test = sec_map()
    # test Txt2PtNpy
    txtbrain_x = '/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_x.txt'
    txtbrain_y = '/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_y.txt'
    txtbrain_z = '/home/maguangshen/PycharmProjects/BTL_GS/Data/brain_z.txt'
    # vtk_brain, npy_brain, npy_brain_x, npy_brain_y, npy_brain_z = test.Txt2PtNpy(txtbrain_x, txtbrain_y, txtbrain_z)
    # print(npy_brain_x[1:3])
    # print('\n')
    # print(npy_brain_y[1:3])
    # print('\n')
    # print(npy_brain_z[1:3])
    # print('\n')
    # print(npy_brain)
    # print('\n')
    #
    # # test AglTransform
    # R_tform = AglTransform(180, 0, 0)
    # print(R_tform)
    #
    # # test Pc2Origin
    # npy_origin = Npy2Origin(npy_brain)
    # vtk_origin = sec_pre.npy2vtk(npy_origin)
    # sec_viz.vtk_pc(vtk_origin)
    # print(np.min(npy_origin[:, 0]))
    # print(np.min(npy_origin[:, 1]))
    # print(np.min(npy_origin[:, 2]))

    # test Brain_Init
    theta = [180, 180, 0]
    vtk_brain, npy_brain, x_brain, y_brain, z_brain = test.Brain_Init(txtbrain_x, txtbrain_y, txtbrain_z, theta)
    SEC_VIZ.vtk_pc(vtk_brain)
