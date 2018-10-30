#from sklearn.svm import LinearSVC
#from sklearn.externals import joblib
import glob
import os
from skimage.feature import hog
import numpy as np
import cv2

pos_im_path = 'C:\\Users\\Richa Agrawal\\Desktop\\computer vision\\video\\re-svm\\data\\images\\pos_person'
neg_im_path = 'C:\\Users\\Richa Agrawal\\Desktop\\computer vision\\video\\re-svm\\data\\images\\neg_person'


samples = []
labels = []    

# Get positive samples
for filename in glob.glob(os.path.join(pos_im_path, '*.png')):
    img = cv2.imread(filename, 0)
    hist =  hog(img, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(2, 2),block_norm='L1', visualise=False,transform_sqrt=False,feature_vector=True,normalise=None)
    samples.append(hist)
    labels.append(1)

# Get negative samples
for filename in glob.glob(os.path.join(neg_im_path, '*.jpg')):
    img = cv2.imread(filename, 0)
    hist =  hog(img, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(2, 2),block_norm='L1', visualise=False,transform_sqrt=False,feature_vector=True,normalise=None)
    samples.append(hist)
    labels.append(0)

# Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)


# Shuffle Samples
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]    

# Create SVM classifier
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF) # cv2.ml.SVM_LINEAR
# svm.setDegree(0.0)
svm.setGamma(5.383)
# svm.setCoef0(0.0)
svm.setC(2.67)
# svm.setNu(0.0)
# svm.setP(0.0)
# svm.setClassWeights(None)

# Train
svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
svm.save('C:\\Users\\Richa Agrawal\\Desktop\\computer vision\\video\\re-svm\\data\\models\\svm_data.dat')