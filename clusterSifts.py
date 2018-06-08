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

filePath = '../sift3/'
clusterNum = 50
folderNames = listdir(filePath)
desList = []

folderNum = 0
for folder in folderNames[:]:
    folderNum+=1
    mypath = filePath + folder
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    exist = False
    count = 0
    for file in filenames:
        count +=1
        with open(file, 'rb') as f:
                my_list = pickle.load(f)
                for elem in my_list:
                    desList+= [elem]
        print ('folder ' + str(folderNum) + '/' + str (len(folderNames)) + ': '+ str(count) + '/' + str(len(filenames)))

print ('finished folders, training kmeans')
print (len(desList))
kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(desList)

print ('finished kmeans, saving model')
filename = 'kmeans3.sav'
pickle.dump(kmeans, open(filename, 'wb'))
