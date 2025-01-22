from PIL import Image
from skimage.feature import greycomatrix, greycoprops
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import scipy
from scipy import ndimage
import glob
import math
import pandas as pd
from itertools import islice
import pdb
import mahotas.features as mf
import os
import skimage
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

cv2.ocl.setUseOpenCL(False)


def haralick_feature_extraction(images, image_array=True):
    """
    This function finds the Haralick features from the image(s) for ML classification
    Notes:
        Authors, Ahmad

    Args:
        images (str): directory where the images are located
                      or
                      an image numpy array

        image_array (bool): True(default), when the numpy object of image is passed
                            false, when the path of the image(s) is passed
    Returns:
        Four np arrays, feature vectors for all images in direction:
            1.   0 degrees
            2.  45 degrees
            3.  90 degrees
            4. 135 degrees
            5. list of filenames and class

    """
    feature_0 = np.ndarray((0,13), int); feature_45 = np.ndarray((0,13), int)
    feature_90 = np.ndarray((0,13), int); feature_135 = np.ndarray((0,13), int)
    first_loop = True
    f = ['init']
    img_file_names = []
    if not image_array:
        f = sorted(glob.glob(images))
        files_name =  [elem.split("/")[-2:] for elem in f]

    for i in range(len(f)):
        if not image_array:
            n_img = Image.open(f[i]).convert('L') #greyscale 'L'
            n_img = np.array(n_img.getdata()).reshape((n_img.size))
            #n_img = sp.misc.imread(f[i], mode='F').astype(np.uint8)
            f14 = mf.haralick(n_img,compute_14th_feature=True)
            img_file_names.append(files_name[i])
        else:
            f14 = mf.haralick(images,compute_14th_feature=True)

        #glcm = sf.greycomatrix(n_img,[1], [0,np.pi/4, np.pi/2, 3*np.pi/4])
        if first_loop :
            feature_0 = np.array(f14[0, :]); feature_45 = np.array(f14[1, :])
            feature_90 = np.array(f14[2, :]); feature_135 = np.array(f14[3, :])
            first_loop = False
        else:
            feature_0 = np.append(feature_0,np.array(f14[0,:]),axis=0)
            feature_45 = np.append(feature_45, np.array(f14[1, :]), axis=0)
            feature_90 = np.append(feature_90, np.array(f14[2, :]), axis=0)
            feature_135 = np.append(feature_135, np.array(f14[3, :]), axis=0)

    feature_0 = feature_0.reshape( i+1, 14)
    feature_45 = feature_45.reshape( i+1, 14)
    feature_90 = feature_90.reshape( i+1, 14)
    feature_135 = feature_135.reshape( i+1, 14)
    #svd_data(feature_0)
    #X = pca_data(feature_0)
    #clust = hierarchical_clust(X,files_name,3)
    #print ("feature extraction")
    return feature_0, feature_45, feature_90, feature_135, img_file_names


def global_feat(img,length_mult = 2):
    """
    calculates the global feature by horizontally summing the intnsity and combining rows
    Args:
        img: a 2D numpy array
        length_mult: how much to compress infomation:
            for example, when img is 250x250, a length_mult of 2 will have the vector be (125,)
    """
    vector = []
    #print(img.shape)
    for i in range(0,len(img),length_mult):
        tmp = 0
        for j in range(len(img)):
            tmp += int(img[i][j])+ int(img[i+1][j])
        vector.append(tmp)
    return np.array(vector)/np.linalg.norm(np.array(vector))


def global_feat_figure(img,length_mult = 1):
    """
    Replicates figure in PVSC 2020 paper
    """
    vector = []
    #print(img.shape)
    for i in range(0,len(img)-1,length_mult):
        tmp = 0
        for j in range(len(img)):
            tmp += int(img[i][j])+ int(img[i+1][j])
        vector.append(tmp)
    vec = np.array(vector)
    vec = vec/np.linalg.norm(vec)
    plt.plot(vec)
    plt.xlabel("Image index")
    plt.ylabel("Normalized sum of intensity")
    plt.show()
    plt.imshow(ndimage.rotate(img, 90), cmap = 'Greys_r')
    plt.axis('off')
    #f,ax = plt.subplots(1,2)
    #ax[0].plot(vec)
    #ax[0].set_aspect('equal')
    #ax[1].imshow(img)
    #ax[1].axis('off')
    #plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def predictBoW(model, bow, names, pca= True,  csv_path = "test.csv",model_type = 'agg'):
    if pca:
        bow = do_pca(bow, n_comp = 2)
    agg = None
    if model_type == 'agg':
        agg = model.fit_predict(bow)
    elif model_type == 'gmm':       
        agg = model.predict(bow)
    elif model_type == 'spectral':
        agg = None
        agg = model.fit_predict(bow)
    df = pd.DataFrame(list(zip(names, list(agg))), columns = ['filename','class']) # magic
    df.to_csv(csv_path) # saves a csv with filename and classification.
    return df

def fitBoW(bow,names = None, pca = True, dist = None, clusters = 4, model_type = 'agg'):
    """
    Fit the bag of words with a given ML clustering model
    Args:
        bow: a NumPy bag of words (output of predictBoW)
        names: the image names. Not used.
        pca:bool, use PCA or not
        dist:bool use a distance metric or not. Generally not useful
        clusters:int the number of classes/clusters you want to have
        model_type: string, agg= hierarchial clustering, gmm = Gaussian Mixture Model, spectral = spectral clustering
    """
    if pca:
        bow = do_pca(bow, n_comp = 2)
    if model_type == 'agg':
        if dist is None:
            model = AgglomerativeClustering(n_clusters = clusters).fit(bow) # now cluster the visual histograms for classification
        else:
            model = AgglomerativeClustering(n_clusters = None, distance_threshold = dist).fit(bow)
    elif model_type == 'gmm':
        model = GaussianMixture(n_components = clusters).fit(bow)
        
    elif model_type == 'spectral':
        model = SpectralClustering(n_clusters=clusters, 
                                           eigen_solver='arpack', 
                                           affinity="nearest_neighbors").fit(bow)
    return model

def constructBoW(filepath, feature = 'kaze', clusters = 4, hist_len = 100, dist= None, global_vec_len = 125, haralick = False, custom = True):
    """
    This function constructs a bag of visual words model using local features
    such as KAZE, or FAST.

    Authors: Ben (bgp12)
    Args:
        filepath:str, a path to the top level directory for the images,
                      which can then be contained in subdirectories
        csv_path:str, where to save the output .csv file
        clusters:int, the number of clusters to output, e.g. 2 (good, bad)
        hist_len:int, the length of the histogram, usually 100
        dist:float, optional, use distance rather then cluster number for HC.
        pca:bool, use PCA on the bag or words or not
        global_vec_len:int, length to make custom descriptor
        haralick:bool, use Haralick features or not.
        custom:bool, use custom descriptor or not.
    """ 
    if feature == 'kaze':
        N = 64
        detector = computer = cv2.KAZE_create()
    if feature == 'fast':
        detector = cv2.FastFeatureDetector_create() #threshold=2
        #detector.setNonmaxSuppression(0) # this line causes FAST to produce more features
        computer = cv2.ORB_create()
        N = 32
    if feature == 'orb':
        detector = computer = cv2.ORB_create()
        N = 64
    if feature == 'daisy': #TODO: not yet fully implemented
        N = 104
    path = '{fp}/**/*.jpg'.format(fp=filepath) # gets all jpegs in subdirectory
    if custom:
        bow = np.empty((0,hist_len + global_vec_len))
    elif haralick:
        bow = np.empty((0,hist_len + 14)) #number of haralick features
    else: # TODO: Add more GLCM/Haralick features
        bow = np.empty((0,hist_len))
    names = [] #for labeling
    kmeans = MiniBatchKMeans(n_clusters = hist_len)
    all_des = []
    all_global = []
    all_haralick = []
    fails = 0
    i = 0
    failures = []
    imgpaths = []
    for img in glob.glob(path,recursive = True):#gets each individual image
        if "Blank" in img:
            continue
        imgpaths.append(img)
        gray = cv2.imread(img,0)
        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        if feature == 'daisy':
            des = skimage.feature.daisy(gray, normalization = 'l2').flatten()
            print(des.shape)
        else:    
            kp = detector.detect(gray ,None)
            _,des = computer.compute(gray,kp)
        if des is None: # TODO: make this result in a histogram of all zeros
            print("failure: " + img)
            fails += 1
            failures.append(img)
            continue
        
        all_des.append(des)
        if custom:
            all_global.append(global_feat(gray))
        #print(feature_45.flatten())
        if haralick:
            feat = glcm(gray)
            all_haralick.append(feat)
            #feature_0, feature_45, feature_90, feature_135, img_file_names = haralick_feature_extraction(gray, image_array=True)
            #all_haralick.append(feature_45.flatten())
        i+=1
        #print(i)
        names.append(img)
    print("failures= " + str(fails))
    all_des = np.array(all_des) #all_des = descriptors attatched to each img
    descriptors = np.empty((0,N)) # descriptors = all descriptors together
    for img_des in all_des:
        for k in img_des:
            descriptors = np.vstack((descriptors, k))
    print(descriptors.shape)
    kmeans.fit(descriptors)
    """
    In previous versions, we fit each descriptor as we found it.
    However, this created certain errors in how KMeans processes
    samples. The workaround is to allow KMeans to access all descriptors
    at once, as we use batch mode KMeans anyways. This may cause a slight
    increase in memory usage, but whatever.
    """
    if not custom:
        all_global = np.zeros((len(all_des)))
    if not haralick:
        all_haralick = np.zeros((len(all_des)))
    #bow = np.empty((0,hist_len))
    print(len(all_global), len(all_haralick), len(all_des))
    for img_des,global_des, haralick_des in zip(all_des,all_global, all_haralick):
        hist = np.zeros((hist_len)) #create histogram for each image
        for des in img_des:
            cluster_result = kmeans.predict(des.reshape((1,-1))) # KMeans likes col vectors for response
            hist[cluster_result] += 1.0
    
        l2_norm =  hist/np.linalg.norm(hist) # l2 normalization for frequency
        if custom:
            feature_vector =  np.append(l2_norm.flatten(),global_des.flatten())
        elif haralick:
            feature_vector = np.append(l2_norm.flatten(),haralick_des.flatten())
        else:
            feature_vector = l2_norm.flatten()
    
        bow = np.vstack((bow,feature_vector)) #this bag of words creates a "global" descriptor for the image based off the SIFT keypoints
    print(bow.shape)
    return bow,names,failures

def do_pca(bow, n_comp = 2):
    pca = PCA(n_components=n_comp, svd_solver='full')
    return pca.fit_transform(bow)

def glcm(img):
    return greycomatrix(img, [2], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=4)


def show_sequential(csv,classes = 4, samples = 16):
    """
    Shows a stiched-together image of some images in a cluster
    
    Args:
        csv: a path to a csv that is the output of previous functions
        classes: number of clusters
        samples (integer): a perfect square for number of samples
    """
    df = pd.read_csv(csv)   
    df = df.sort_values(by=['class'])
    n = samples
    filenames = df.groupby(['class']).head(n)['filename'].tolist()
    filenames = [filenames[i*n:(i+1)*n] for i in range((len(filenames)+n-1)//n)]
    #print(filenames)

    nrow = int(math.sqrt(samples))
    ncol = int(math.sqrt(samples))
    x,y = cv2.imread(filenames[0][0],0).shape
    #imgs = []
    #print(len(filenames))
    for j in range(len(filenames)):
        cvs = Image.new("P",(x*nrow,y*ncol) )
        class_i=filenames[j]
        print(class_i[:3])
        for i in range(len(class_i)):
            px,py = x*int(i/nrow), y*(i%nrow)
            cvs.paste(Image.open(class_i[i],"r"),(px,py))
        plt.imshow(cvs, cmap = "Greys_r")
        plt.show()
        

def eval_accuracy(csv_path = 'fasttest4Class.csv'):
    """
    This function attempts to transform cluster labels -> classes
    This depends highly on how the cluster algorithum chooses labels
    and is not deterministic
    This basically always needs to be done manually, by hand.
    
    
    You either label all images at the start, or the classes at the end
    
    Returns accuracy: #successful classifications/total number of samples
    """
    #show_sequential(csv_path, classes = 4, samples = 49)
    res = pd.read_csv(csv_path)
    names = list(res['filename'])
    class_names = [os.path.dirname(x).split("/")[-1] for x in names]
    res['class_true'] = class_names
    res['class_true'] = res['class_true'].str[-1].astype(int)
    res = res.drop(res.columns[[0]], axis = 1)
    res.columns = ['file','class_pred','class_true' ]
    
    res.to_csv(csv_path[:-4]+'cleaned.csv')
    
    res_fixed = []
    # for now, hardcoded bindings
    for row in res.values.tolist():
        file, class_pred, class_true = tuple(row)
        fixed_pred = None
        if class_pred == 0:
            fixed_pred = 3#2
        elif class_pred == 1:
            fixed_pred = 2
        elif class_pred == 2:
            fixed_pred = 1#0
        elif class_pred == 3:
            fixed_pred = 0#1
        elif class_pred == 4:
            fixed_pred = 0
        res_fixed.append([file,fixed_pred,class_true])
    df = pd.DataFrame(res_fixed)
    df.columns= ['file','class_pred','class_true' ]
    #res.loc[res['class_pred'] == 2, 'class_pred'] = 0

    acc = len((np.where(df['class_true'] == df['class_pred']))[0]) / len(df['class_true'])
    return acc

def image_resize_l(img_file,img_len, img_wdth):
    """
    This function resize the number of pixels in an image
    :param img_file: Path of an image file
    :param img_len:  Length of the output image
    :param img_wdth: Width of the output image
    :return: re-sized image
    """
    sz = (img_len, img_wdth)
    img = cv2.imread(img_file,0)
    #img = cv2.imread(img_file)
    img = cv2.resize(img, sz).astype('float32') / 255.0
    #img = img.astype('float32') / 255.0
    return img


def read_input_images(dir_count, dir_path,image_len, image_wdth,nprocs=3):
    """
    This function reads input images of different class from corresponding directories.
    Directories Should have the name as 0,1,2,3 ....and so on. Only integers.
    Args:
        dir_count (int) : Number of different directories to read data from
        dir_path  (str) : Path of the directories
        image_len (int) : Length of the images
        image_wdth(int) : Width of the images
        nprocs: Number of parallel process, default = 3
    Returns:
        x (nparray) : data
        y (nparray) : labels
    """
    X_train = []
    Y_train = []
    #read images directory wise and assign labels based on the folder name
    for i in range(dir_count):
        path = os.path.join(dir_path+str(i)+"/*.jpg")
        files = glob.glob(path)
        X_train.extend(Parallel(n_jobs=nprocs)(delayed(image_resize_l)(im_file,image_len,image_wdth) for im_file in files))
        Y_train.extend([i] * len(files))
    x = np.array(X_train)
    y = np.array(Y_train)
    return x, y

def data_partition(x,y,test_size_percent = 0.2,kfolds=1):
    """
    Args:
        x (nparray) : numpy array of image data
        y (nparray) : labels
        kfolds (int): Most;y it would remain 1, but it can be changed
    Returns:
        X_train (nparray) : Training data
        Y_train (nparray) : Training label
        X_test  (nparray) : Test data
        Y_train (nparray) : Test label
    """
    s = StratifiedShuffleSplit(n_splits=kfolds, test_size = test_size_percent, random_state=0)
    for train_index, test_index in s.split(x,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
    return X_train, X_test, Y_train, Y_test

def change_dimension(X_train,X_test):
    """
    Updates the 3d nparray to 2d nparray for training
    Args:
        X_train: Input training image data vector 3-D
        X_test:  Test image data 3-D vector
    Returns:
        X_train (2d nparray) : Training data
        X_test (2d nparray) : test data
    """
    #image reshape to convert 3-D numpy vector to 2-D
    img_size = X_train[0].shape[0] * X_train[0].shape[1]
    X_train = X_train.reshape(len(X_train), img_size)
    X_test = X_test.reshape(len(X_test), img_size)
    return X_train, X_test

def kfold_partition(x, y, kfolds=5):
    """
    Partitions the data into multiple fold, helpful for crossvalidation in ML 
    Args:
        x (numpy) : Dataset
        y (numpy) : Labels
        kfolds (int): Number of folds
    Returns:
        k pairs of training and testing indices
        train_index (vector) : vector of training index
        test_index (vector) : vector of test index
    """
    train_index = []
    test_index = []
    s = StratifiedKFold(n_splits=kfolds, random_state=None, shuffle=True)
    # for loop to run number of times equal to value of kfolds
    for train_i, test_i in s.split(x, y):
        train_index.append(train_i)
        test_index.append(test_i)
    return train_index, test_index


def logistic_regression(X_train, y_train,X_test,y_test,lmbda=1e-5,iter=10000):
    """
     Trains the image dataset for logistic regression.
     Args:
         x (numpy) : Dataset
         y (numpy) : Labels
         kfolds (int): Number of folds
     Returns:
         k pairs of training and testing indices
    """
    #from sklearn.metrics import accuracy_score
    #from sklearn import linear_model
    clf=linear_model.LogisticRegression(C=lmbda, max_iter=iter,n_jobs=-1)
    clf.fit(X_train,y_train)
    #X_test = test_dataset.reshape(len(test_dataset),len(test_dataset[0])*len(test_dataset[0]))
    pred_train= [(clf.predict(row.reshape(1,-1)))[0] for row in X_train]
    pred_lst = [(clf.predict(row.reshape(1,-1)))[0] for row in X_test]
    test_accuracy = accuracy_score(y_test,pred_lst)
    train_accuracy = accuracy_score(y_train,pred_train)
    return test_accuracy, train_accuracy


def support_vector_machine(X_train, y_train,X_test,y_test,gamma_vector=[1e-5,1e-2, 0.05, 0.5],
                           c_vector=[0.001,0.01,0.1, 10, 25, 50],c_v=5):
    """
    Implemnts SVM for the image dataset
    Args:
        X_train (nparray): numpy vector, input training data
        y_train (nparray : numpy vector, training label
        X_test (nparray) : numpy vector, test data
        y_test (nparray) : numpy vector, test label
    Returns:
        y_test (nparray) : Test labels
        y_pred (nparray) : Prediction labels
    """
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_vector, 'C': c_vector}]
    #clf_svc = GridSearchCV(SVC(C=25), tuned_parameters, cv=10, n_jobs=-1)
    #score = ['precision', 'recall']
    #clf_svc = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf_svc = GridSearchCV(SVC(), tuned_parameters, cv=c_v)
    clf_svc.fit(X_train, y_train)
    clf_svc.best_params_
    means = clf_svc.cv_results_['mean_test_score']
    stds = clf_svc.cv_results_['std_test_score']
    scores = means.reshape(len(c_vector),len(gamma_vector))
    # for mean, std, params in zip(means, stds, clf_svc.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    y_pred =  clf_svc.predict(X_test)
    #pred_svc_train = [(clf_svc.predict(row.reshape(1, -1)))[0] for row in X_train]
    #pred_svc_test = [(clf_svc.predict(row.reshape(1, -1)))[0] for row in X_test]
    return y_test, y_pred

def confusion_classification(y_test,y_pred):
    """
    This function gives classification report and confusion matrix
    Args:
        y_test (vector) :test labels
        y_pred (vector) :pred labels
    Returns:
        classification_report (nparray): classifcation report
        confusion_matrix (nparray): confusion matrix
    """
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    return classification_report(y_test, y_pred),confusion_matrix(y_test, y_pred)


def CNN(train_path,test_path, batch_size = 128, N_classes = 4, 
        val_split = 0.33, n_epochs = 20):
    """
    Build a CNN model to classify EL/PL images
    Folder structure should look like:
    
    train
        L c0
        L c1
        L c2
    test
        L c0
        L c1
        L c2

    Parameters
    ----------
    train_path : string
        Path to trining data.
    test_path : string
        path to validation data.
    batch_size : int, optional
        batch size. The default is 128.
    N_classes : int, optional
        number of classes. The default is 4.
    val_split : float, [0,1], optional
        Percentage of dataset to use as validation. The default is 0.33.
    n_epochs : int, optional
        number of training epochs. The default is 20.

    Returns
    -------
    model : keras.Model
        A trained Keras model.

    """
    
    train_datagen = ImageDataGenerator(rescale=1/255,
                                horizontal_flip=False,
                                validation_split=val_split)
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=(250, 250),
                                                        batch_size=batch_size,
                                                        #classes = ['Cracked', 'NotCracked'],
                                                        class_mode='binary',
                                                        color_mode='grayscale',
                                                        subset='training')
    
    valid_generator = train_datagen.flow_from_directory(test_path,
                                                        target_size=(250, 250),
                                                        batch_size=batch_size,
                                                        #classes = ['Cracked', 'NotCracked'],
                                                        class_mode='binary',
                                                        color_mode='grayscale',
                                                        subset='validation')
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(250, 250, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(N_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=RMSprop(lr=0.0001),
                  metrics=['acc'])
    total_sample=train_generator.n
    history = model.fit_generator(
            train_generator, 
            steps_per_epoch=int(total_sample/batch_size),  
            epochs=n_epochs,
            validation_data=valid_generator,
            verbose=1)
    return model

#if __name__ == '__main__':

    #global_feat_figure(cv2.imread("/mnt/rstor/CSE_MSE_RXF131/hpc-members/axk962/datasets/1801-18-EL_SunEdison_Sorted/Corroded/sa40002_00-06-06-02-sedh01-c22.jpg",0))

    #BoW('/mnt/rstor/CSE_MSE_RXF131/hpc-members/axk962/datasets/1801-18-EL_SunEdison_Sorted/',csv_path='FASTSpeedTest6Cluster.csv', N = 64, clusters = 6, pca = True, haralick=False, custom=False)
    #BoW('C:\\Users\\Ben\Documents\\1907French-etal-SDLE-EL-ImageDataSet-forML\\train\\',csv_path='testman.csv', clusters = 4, pca = True, haralick=False, custom=False)
    #BoW('C:\\Users\Ben\\Documents\\quick_test\\',csv_path='testman.csv', clusters = 4, pca = True, haralick=False, custom=False)
    #csv_path = 'testman2.csv'
    #BoW('/home/ben/jpv_1/',csv_path=csv_path, clusters = 3, feature = 'fast', pca = True, haralick=False, custom=True)
    #def constructBoW(filepath,N = 64 ,csv_path = "./", feature = 'kaze', clusters = 4, hist_len = 100, dist= None,pca = False, global_vec_len = 125, haralick = False, custom = True):
    
    #print(eval_accuracy())
    
    #bow,names,fails = constructBoW('/home/ben/1912-Single_brand_dampheat_EL_cell_images/train_set/', feature = 'fast', clusters = 5)
    #bow,names,fails = constructBoW('/home/ben/1912-Single_brand_dampheat_EL_cell_images/train_set/', feature = 'fast', clusters = 5)
    #model = fitBoW(bow,names, model_type = 'agg', pca = True, clusters = 5)
    #preds = predictBoW(model,bow,names,model_type = 'agg', csv_path = 'jpvagg2.csv', pca = True)
    #show_sequential('jpvagg2.csv', samples = 49)
    #eval_accuracy(csv_path = 'jpvagg.csv')
    #eval_accuracy('jpvagg.csv')
