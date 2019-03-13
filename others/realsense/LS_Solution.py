# Ref: http://nghiaho.com/uploads/code/rigid_transform_3D.py_
import numpy as np
import math
import random
import BTL_DataConvert as DC
import BTL_VIZ as BV
import vtk
import pcl

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

class CamCalibrate():

    def __init__(self):

        self.N = 900 # number of dataset
        self.A = np.zeros((self.N, 3))  # The format of dataset
        self.noise_scale = 0.00008

        # Random rotation and translation
        self.R = np.mat(np.random.rand(3, 3))
        self.t = np.mat(np.random.rand(3, 1))

    def GenData(self):

        # Test with random data
        # Generate the data sets
        x = np.linspace(0, 20, 30)
        y = np.linspace(0, 20, 30)
        xv, yv = np.meshgrid(x, y)
        x_use = xv.reshape(len(x) * len(y), )
        y_use = yv.reshape(len(x) * len(y), )
        z = np.ones((len(x) * len(y), 1)).reshape(len(x) * len(y), )

        # Add the noise data
        noise = 0.0008 * np.asarray(random.sample(range(0, 1000), self.N))

        # Add the noise to the point cloud
        npy_source = np.zeros((len(x) * len(y), 3))
        npy_source[:, 0] = x_use
        npy_source[:, 1] = y_use
        npy_source[:, 2] = z + noise

        return npy_source

    def test_LS(self):

        npy_source = self.GenData()

        # Random rotation and translation
        R = self.R
        t = self.t

        # make R a proper rotation matrix, force orthonormal
        U, S, Vt = np.linalg.svd(R)
        R = U * Vt

        # remove reflection
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = U * Vt

        # Choose the dataset
        # A = np.mat(np.random.rand(self.N,3))
        A = npy_source
        B = R * A.T + np.tile(t, (1, self.N))

        noise = self.noise_scale * np.asarray(random.sample(range(0, 1000), self.N))
        npy_noise = np.tile(noise, (3, 1))
        B = B + npy_noise
        B = B.T

        # Recover the transformation
        ret_R, ret_t = self.rigid_transform_3D(A, B)
        A2 = (ret_R * A.T) + np.tile(ret_t, (1, self.N))
        A2 = A2.T

        # Find the error
        err = A2 - B
        err = np.multiply(err, err)
        err = sum(err)
        rmse = np.sqrt(err/self.N)

        # Viz the result
        actor_A = BV.ActorNpyColor(A, [255,0,0])
        actor_A2 = BV.ActorNpyColor(A2, [0,255,0])
        actor_B = BV.ActorNpyColor(B, [0,0,255])
        origin_A = [0,0,0]
        axes_A = BV.ActorAxes(origin_A)
        origin_A2 = [1,1,1]
        axes_A2 = BV.ActorAxes(origin_A2)
        BV.VizActor([actor_A, actor_A2, actor_B, axes_A, axes_A2])

        # Print the result
        print "LS_err = ", err
        print "LS_rmse = ", rmse

    def test_ICP(self):

        npy_source = self.GenData()

        # Random rotation and translation
        R = self.R
        t = self.t

        # make R a proper rotation matrix, force orthonormal
        U, S, Vt = np.linalg.svd(R)
        R = U * Vt

        # remove reflection
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = U * Vt

        # Choose the dataset
        # A = np.mat(np.random.rand(self.N,3))
        A = npy_source
        B = R * A.T + np.tile(t, (1, self.N))
        noise = self.noise_scale * np.asarray(random.sample(range(0, 1000), self.N))
        npy_noise = np.tile(noise, (3, 1))
        B = B + npy_noise
        B = B.T

        # Recover the transformation
        source = pcl.PointCloud(A.astype(np.float32))
        target = pcl.PointCloud(B.astype(np.float32))
        icp = source.make_IterativeClosestPoint()
        converged, transf, estimate, fitness = icp.icp(source, target, max_iter=1000)
        ret_R = transf[0:3,0:3]
        ret_t = transf[3, 0:3]
        A2 = np.dot(npy_source, np.linalg.inv(ret_R)) + np.tile(ret_t.T, (self.N, 1))
        # A2 = (np.matmul(A, np.linalg.inv(ret_R.T))) + np.tile(ret_t.T, (self.N, 1))
        # A2 = A2

        # Find the error
        err = A2 - B
        err = np.multiply(err, err)
        err = sum(err)
        rmse = np.sqrt(err / self.N)

        # Viz the result
        actor_A = BV.ActorNpyColor(A, [255, 0, 0])
        actor_A2 = BV.ActorNpyColor(A2, [0, 255, 0])
        actor_B = BV.ActorNpyColor(B, [0, 0, 255])
        BV.VizActor([actor_A, actor_A2, actor_B])

        # Print the result
        print "tranf = ", transf
        print "R = ", R
        print "t = ", t
        print "err = ", err
        print "rmse = ", rmse

    def rigid_transform_3D(self, A, B):

        assert len(A) == len(B)

        N = A.shape[0]  # total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))

        # dot is matrix multiplication for array -- H is the covariance matrix
        H = np.transpose(AA) * BB

        # solve the SVD problem
        U, S, Vt = np.linalg.svd(H)

        # Calculate the rotation matrix
        R = Vt.T * U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            print "Reflection detected"
            Vt[2, :] *= -1
            R = Vt.T * U.T

        # Calculate the t vector
        # for npy source
        t = (-np.matmul(R, centroid_A.T)).T + centroid_B.T
        # for random data
        # t = (-np.matmul(R, centroid_A.T)) + centroid_B.T

        print "t ="
        print t

        return R, t

if __name__ == "__main__":

    test = CamCalibrate()
    test.test_LS()

    # test.test_ICP()