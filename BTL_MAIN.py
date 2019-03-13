'''
Automatic registration using simulated brain image data
Algorithm:
1. Generate the brain cortical surface stl/plt file
   Change different images and generate different models
2. Threshold out the brain vessels and extract the salient features
3. Based on the salient features -- develop an automated way to detect the features
4. Capture the training data set based on different camera poses -- vtk version
5. Divide the regions into several subregions (classes)
6. Train the neural network model -- finished the first step
'''

