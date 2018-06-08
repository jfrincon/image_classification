# Author: Juan Fernando Rincón Cardeño
# email: jrinco15@eafit.edu.co
import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn.externals import joblib
import pickle

from sklearn import svm

from os import listdir

kmeans_path = 'kmeans3.sav'
kmeans = pickle.load(open(kmeans_path, 'rb'))
clusterNum = 50

svm_path = 'SVM_model10'
svc = pickle.load(open(svm_path, 'rb'))

names = ['19','auditorio','18','38','biblioteca','26','idiomas','dogger','agora','admisiones']

def tagImage(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (250,250))
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,img)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)

    histogram = [0]*clusterNum
    for elem in des:
        histogram[kmeans.predict([elem])[0]]+= 1
    maxElem = max(histogram)

    for elem in range(len(histogram)):
        histogram[elem] = histogram[elem]/maxElem

    prediction = svc.predict([histogram])[0]

    print (names[int(prediction)-1])
    return (prediction)

filePath = '../test/26/'
folderNames = listdir(filePath)
for name in folderNames:
    print (filePath + name)
    tagImage(filePath + name)
