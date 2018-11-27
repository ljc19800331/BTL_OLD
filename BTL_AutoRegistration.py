import cv2
import numpy as np
import vtk
import open3d
import BTL_Registration
import BTL_PatchGrid

# Algoirhtms
# 1. get the patch block
# 2. get the projected image for this block
# 3. Register the whole image
# 4. Compress the image data
# 5. Pending: AutoEncoder program -- code a new program based on this idea
# 6. 3D point cloud for date compression -- AutoEncoder or PCA -- which is better?
# 7. Extended gaussian image for cortex registration -- based on color-weighted? -- 3D feature descriptor for this problem
# 8. Automated registration based on colorized region registration -- this is difficult? -- RGBD local to global registration

class AutomatedRegis():
    # Automated registration This is important for the learning
    #
    def __init__(self):

        # Test the results
        self.test = BTL_Registration.BrainRegis()
        self.brain_pcd, self.scan_pcd = self.test.Npy2Pcd()

        # Open3d([brain_pcd, scan_pcd])
        self.brain_npy = np.asarray(self.brain_pcd.points)
        self.scan_npy = np.asarray(self.scan_pcd.points)

    def DividePatch(self):

        # Divide the 3D model into several patches
        a = 1

        # Find the pc in the grid cube

        # Save the results

    def MultiCamera(self):

        a = 1

        # Initialize the discrete points

        # Define the focus point

        # Define the camera viewpoint

        # Capture the depth patch

        # Register the point cloud for mutual information

if __name__ == "__main__":
    test = AutomatedRegis()
    test.LoadPC()