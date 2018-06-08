# Author: Juan Fernando Rincón Cardeño
# email: jrinco15@eafit.edu.co
import cv2
import numpy as np

from os import listdir
from os.path import isfile, join

from sklearn.cluster import KMeans
import pickle
from sklearn.externals import joblib
import numpy as np

filePath = '../test/'
siftPath = '../sift5/'
clusterNum = 300
folderNames = listdir(filePath)
folderCount = 0
for folder in folderNames:
    mypath = filePath + folder
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    exist = False
    count = 0
    folderCount +=1
    for file in filenames:
        img = cv2.imread(file)
        img = cv2.resize(img, (250,250))
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv2.drawKeypoints(gray,kp,img)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
        the_filename = siftPath + folder + '/'+str(count)+ '.sav'
        with open(the_filename, 'wb') as f:
            pickle.dump(des, f)
        count+=1
        print ('folder ' + str(folderCount) + '/' + str (len(folderNames)) + ': '+ str(count) + '/' + str(len(filenames)))

    print ('finished folder '+ folder)
