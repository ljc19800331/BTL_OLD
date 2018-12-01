# This is the script for dictionary learning
# ref: https://blog.csdn.net/baimafujinji/article/details/80872099
# ref: https://www.jianshu.com/p/f6e5d1cd21b9
# ref: https://blog.csdn.net/tiaxia1/article/details/80264228
# ref: https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf

# The matching algoritihm for the dictionary
# Problem statement: Given a new vector, find the closest encoder for this feature vector represented by the dictionary.
# Look at: https://www.cnblogs.com/theonegis/p/7653425.html
# Matching algorithm: https://www.cnblogs.com/theonegis/p/7653425.html

print(__doc__)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from keras.preprocessing.image import load_img

# load an image from file
image = load_img('lena_gray_256.tif')

from keras.preprocessing.image import img_to_array

# convert the image pixels to a numpy array
image = img_to_array(image)
image = image[:, :, 0]
print("original shape", image.shape)

# Normalized the result
image = image.astype('float32')
image /= 255
plt.imshow(image, cmap='gray')

# Add the gaussian noise
noise = np.random.normal(loc=0, scale=0.05, size=image.shape)
x_test_noisy1 = image + noise
x_test_noisy1 = np.clip(x_test_noisy1, 0., 1.)
plt.imshow(x_test_noisy1, cmap='Greys_r')

# Extract all reference patches from the original image -- train the dictionary with the patches
print('Extracting reference patches...')
patch_size = (5, 5)
data = extract_patches_2d(image, patch_size)
print(data.shape)

# PATCH TO VECTOR
data = data.reshape(data.shape[0], -1)
print(data.shape)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)

# #############################################################################
# Learn the dictionary from reference patches
print('Learning the dictionary...')
dico = MiniBatchDictionaryLearning(n_components=144, alpha=1, n_iter=500)
V = dico.fit(data).components_

print(V.shape)

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:144]):
    plt.subplot(12, 12, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from patches\n', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# #############################################################################
# Extract noisy patches and reconstruct them using the dictionary

print('Extracting noisy patches... ')
data = extract_patches_2d(x_test_noisy1, patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept

print('Orthogonal Matching Pursuit\n2 atoms' + '...')
reconstructions = x_test_noisy1.copy()

dico.set_params(transform_algorithm='omp', **{'transform_n_nonzero_coefs': 2})
code = dico.transform(data)
patches = np.dot(code, V)

patches += intercept
patches = patches.reshape(len(data), *patch_size)

reconstructions = reconstruct_from_patches_2d(patches, (256, 256))

plt.imshow(reconstructions, cmap='Greys_r')




















