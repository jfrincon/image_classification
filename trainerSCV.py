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


filename = 'kmeans3.sav'
kmeans = pickle.load(open(filename, 'rb'))
labels = kmeans.labels_
clusterNum = 50

filePath = '../sift5/'
folderNames = listdir(filePath)

histograms = []
folderNum = 0
for folder in folderNames:
    mypath = filePath + folder
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    count = 0
    folderNum+=1
    for file in filenames[:]:
        count+=1
        with open(file, 'rb') as f:
            my_list = pickle.load(f)
            hist = [0]*clusterNum
            for elem in my_list:
                hist[kmeans.predict([elem])[0]]+= 1
            maxElem = max(hist)

            for elem in range(len(hist)):
                hist[elem] = hist[elem]/maxElem
            hist +=  [folderNum]
            #print (hist)
            histograms += [hist]
            print ('folder ' + str(folderNum) + '/' + str (len(folderNames)) + ': '+ str(count) + '/' + str(len(filenames)))

print (folderNames)

with open('histograms_test1', 'wb') as f:
    pickle.dump(histograms, f)

#print(kmeans.cluster_centers_)
