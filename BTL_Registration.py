# Intraoperative and preoperative registration with brain MRI model
import vtk
import numpy as np
import BTL_VIZ
import cv2
import open3d
from scipy.interpolate import griddata
import copy
from open3d import *

class BrainRegis():

    def __init__(self):
        self.filename_brain = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/BrainData/brain_shell.stl'
        self.filename_brain_x = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_x.txt'
        self.filename_brain_y = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_y.txt'
        self.filename_brain_z = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_z.txt'
        self.brain_npy = np.load('/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_npy.npy')
        self.filename_brain_pcd = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_pcd.pcd'
        self.filename_scan_pcd = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/scan_pcd.pcd'
        self.filename_brain_ply = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain.ply'
        self.filename_scan_ply = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/scan.ply'

        self.filename_scan_x = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/ScanData(NonRotate)/8/Scan_x'
        self.filename_scan_y = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/ScanData(NonRotate)/8/Scan_y'
        self.filename_scan_z = '/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/ScanData(NonRotate)/8/Scan_z'

    def SurfaceMatch(self):
        # This is for the surface matching algorithms
        a = 1

    def xyz2npy(self, filename_x, filename_y, filename_z):

        # Read the point cloud of the brain model with tumor
        x_file = open(filename_x, "r")
        obj_x = x_file.readlines()
        OBJ = np.zeros([len(obj_x), 3])

        x_file = open(filename_x, "r")
        for idx, line in enumerate(x_file):
            a_x = line.split()
            OBJ[idx, 0] = float(a_x[0])

        y_file = open(filename_y, "r")
        for idx, line in enumerate(y_file):
            a_y = line.split()
            OBJ[idx, 1] = float(a_y[0])

        z_file = open(filename_z, "r")
        for idx, line in enumerate(z_file):
            a_z = line.split()
            OBJ[idx, 2] = float(a_z[0])

        return OBJ
        # np.save('/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_npy', brain)

    def CutRegion(self):

        # Crop some regions from the object for registration
        source_load = open3d.read_point_cloud('/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain.ply')

        source = self.xyz2npy(self.filename_brain_x, self.filename_brain_y, self.filename_brain_z)

        brain_npy = self.NpyNormalized(source)

        source = brain_npy * 1

        # open3d.draw_geometries([source_load])

        # source = np.transpose(np.asarray([source_load.points]))
        # length = source.shape[1]
        # print(source.shape)
        # source = source.reshape((length, 3))
        ratio = 0.6
        print(source.shape)
        x_min = np.min(source[:, 0])
        x_max = np.max(source[:, 0]) * (ratio - 0.3) + np.min(source[:, 0])
        print(x_min)
        print(np.min(source[:, 0]))
        y_min = np.min(source[:, 1])
        y_max = np.max(source[:, 1]) * (ratio - 0.1) + np.min(source[:, 1])
        print(y_min)
        print(y_max)
        z_min = np.min(source[:, 2])
        z_max = np.max(source[:, 2]) * (ratio + 0.3) + np.min(source[:, 2])
        print(z_min)
        print(z_max)

        idx_roi = np.where( (source[:, 0] > x_min) & (source[:, 0] < x_max) & \
                            (source[:, 1] > y_min) & (source[:, 1] < y_max) & \
                            (source[:, 2] > z_min) & (source[:, 2] < z_max))
        print(idx_roi)
        ratio = 0.25              # crop 0.25 part of the region

        xyz = source[idx_roi, :]
        length = xyz.shape[1]
        xyz = xyz.reshape((length, 3))
        print(xyz.shape)
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(xyz)
        open3d.write_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_cube.ply", pcd)
        open3d.draw_geometries([pcd])

    def Test(self):
        actor_stl = BTL_VIZ.ActorStl(self.filename_brain)
        BTL_VIZ.VizActor([actor_stl])

    def ShowPc(self):
        brain = self.brain_npy
        vtk_data = BTL_VIZ.npy2vtk(brain)
        BTL_VIZ.VizVtk([vtk_data])

    def ShowPcPair(self):
        # a = 1
        brain_pcd = open3d.read_point_cloud(self.filename_scan_pcd)
        open3d.draw_geometries([brain_pcd])

    def ShowMesh(self):

        filename_mesh = self.filename_brain_ply
        mesh = open3d.read_triangle_mesh(filename_mesh)
        print(np.asarray(mesh.vertices))
        print(np.asarray(mesh.triangles))
        open3d.draw_geometries([mesh])

    def NpyNormalized(self, data_npy):
        # Normalized the numpy data
        data_npy[:,0] = data_npy[:,0] - np.min(data_npy[:,0])
        data_npy[:,1] = data_npy[:,1] - np.min(data_npy[:,1])
        data_npy[:,2] = data_npy[:,2] - np.min(data_npy[:,2])
        return data_npy

    def Npy2Pcd(self):

        # Load the data
        brain_npy = self.xyz2npy(self.filename_brain_x, self.filename_brain_y, self.filename_brain_z)
        scan_npy = self.xyz2npy(self.filename_scan_x, self.filename_scan_y, self.filename_scan_z)
        scan_npy = scan_npy * 25.4 # inch to mm

        # Normalized the data
        brain_npy = self.NpyNormalized(brain_npy)
        scan_npy = self.NpyNormalized(scan_npy)

        # Convert the data format to the pcd version
        brain_pcd = open3d.PointCloud()
        scan_pcd = open3d.PointCloud()
        brain_pcd.points = open3d.Vector3dVector(brain_npy)
        scan_pcd.points = open3d.Vector3dVector(scan_npy)
        open3d.draw_geometries([brain_pcd, scan_pcd])

        # open3d.write_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain.ply", brain_pcd)
        # open3d.write_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/scan.ply", scan_pcd)

        return brain_pcd, scan_pcd

    def ShowDepth(self):
        # Show the depth images
        brain_npy = self.xyz2npy(self.filename_brain_x, self.filename_brain_y, self.filename_brain_z)
        x_obj = brain_npy[:,0]
        y_obj = brain_npy[:,1]
        x_range = np.max(x_obj) - np.min(x_obj)
        y_range = np.max(y_obj) - np.min(y_obj)
        xy_ratio = y_range / x_range
        print("The ratio of x and y is ", xy_ratio)
        Nx = 100
        Ny = np.asarray(Nx * xy_ratio, dtype = np.int32)
        print("The Nx is ", Nx)
        print("The Ny is ", Ny)

        mesh_x, mesh_y = np.meshgrid(Nx, Ny)
        grid_x, grid_y = np.mgrid[np.min(x_obj): np.max(x_obj): np.max(x_obj) / Nx, \
                         np.min(y_obj): np.max(y_obj): np.max(y_obj) / Ny]
        points = brain_npy[:, 0:2]
        values = brain_npy[:, 2]
        z = griddata(points, values, (grid_x, grid_y), method='nearest')
        z_norm = (z - z.min()) / (z.max() - z.min())
        img = open3d.Image((z_norm * 255).astype(np.uint8))
        open3d.draw_geometries([img])

    def CoarseRegistration(self):

        # Coarse registration using projection image
        a = 1

    def TestSurfaceMatching(self):
        a = 1
        # Load the model and the scene
        # model = open3d.read_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/test_1.ply")
        # scene = open3d.read_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/test_4.ply")
        # open3d.draw_geometries([scene])

        modelp = cv2.ppf_match_3d.loadPLYSimple('test_1.ply', 0)
        detector = cv2.ppf_match_3d_PPF3DDetector(0.025, 0.05)
        detector.trainModel(modelp)
        print(modelp.shape)

        # Register the model to the scene
        # ppf_detector = cv2.ppf_match_3d_PPF3DDetector(0.025, 0.5)

    def TestICP(self):
        a = 1

    def Test3DFeature(self):
        a = 1

    def TestColorICP(self):
        a = 1

    def TestFPFH(self):
        a = 1

    def TestPrjImg(self):
        a = 1

    def Test3DSift(self):
        a = 1

class FPFH:

    def __init__(self):
        self.a = 1

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        open3d.draw_geometries([source_temp, target_temp])

        # open3d.draw_geometries([source])
        # print(source)

    def preprocess_point_cloud(self, pcd, voxel_size):

        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = open3d.voxel_down_sample(pcd, voxel_size)
        radius_normal = voxel_size * 2

        print(":: Estimate normal with search radius %.3f." % radius_normal)
        open3d.estimate_normals(pcd_down, open3d.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = open3d.compute_fpfh_feature(pcd_down, open3d.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return pcd_down, pcd_fpfh

    def prepare_dataset(self, voxel_size):

        print(":: Load two point clouds and disturb initial pose.")

        # source = open3d.read_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/scan_pcd.pcd")
        # target = open3d.read_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_pcd.pcd")

        source = open3d.read_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain.pcd")
        target = open3d.read_point_cloud("/home/maguangshen/PycharmProjects/BTL_GS/BTL_Data/brain_cube.pcd")

        # Intial the translation
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                                 [1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
        source.transform(trans_init)
        self.draw_registration_result(source, target, np.identity(4))
        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = open3d.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            open3d.TransformationEstimationPointToPoint(False), 4,
            [open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
             open3d.RANSACConvergenceCriteria(4000000, 500))
        return result

    def refine_registration(self, source, target, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = open3d.registration_icp(source, target, distance_threshold,
                                  self.result_ransac.transformation,
                                  open3d.TransformationEstimationPointToPlane())
        return result

    def Test_FPFH(self):

        # This is for testing the fast point feature histogram
        voxel_size = 0.05          # means 5cm for the dataset -- 5 mm for

        # Prepare the dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(voxel_size)

        # Ransac fast registration
        self.result_ransac = self.execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

        # Visualize the results
        self.draw_registration_result(source_down, target_down, self.result_ransac.transformation)

        # ICP results for fine registration
        result_icp = self.refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)

        print(result_icp)

        # Draw the final result
        self.draw_registration_result(source, target, result_icp.transformation)

if __name__ == "__main__":

    # test = FPFH()
    # test.Test_FPFH()

    test = BrainRegis()
    test.CutRegion()