#HOG features
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
#import config
pos_im_path = 'C:\\Users\\Richa Agrawal\\Desktop\\computer vision\\video\\re-svm\\data\\images\\pos_person'
neg_im_path = 'C:\\Users\\Richa Agrawal\\Desktop\\computer vision\\video\\re-svm\\data\\images\\neg_person'
min_wdw_sz = [64, 128]
step_size = [10,10]
orientations = 9
pixels_per_cell = [6,6]
cells_per_block = [2, 2]
visualize = False
normalize = True
pos_feat_ph = 'C:\\Users\\Richa Agrawal\\Desktop\\computer vision\\video\\re-svm\\data\\features\\pos'
neg_feat_ph = 'C:\\Users\\Richa Agrawal\\Desktop\\computer vision\\video\\re-svm\\data\\features\\neg'
model_path = 'C:\\Users\\Richa Agrawal\\Desktop\\computer vision\\video\\re-svm\\data\\models'
threshold = .3


    
def extract_features():
    des_type = 'HOG'

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)

    print("Calculating the descriptors for the positive samples and saving them")
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        #print im_path
        
        image = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd =hog(image, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L1', visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print("Positive features saved in {}".format(pos_feat_ph))

    print("Calculating the descriptors for the negative samples and saving them")
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd =hog(im, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L1', visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_ph, fd_name)
    
        joblib.dump(fd, fd_path)
    print("Negative features saved in {}".format(neg_feat_ph))

    print("Completed calculating features from training images")

if __name__=='__main__':
    extract_features()
