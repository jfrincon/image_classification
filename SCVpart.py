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

from sklearn import svm

file = 'histograms10'

with open(file, 'rb') as f:
    print ("reading")
    my_list = pickle.load(f)
    clf = svm.SVC()

    X = []
    y = []
    Xtest = []
    Ytest = []
    count = 0
    for val in my_list:
        max_elem = max(val[:-1])
        for i in range(len(val[:-1])):
            val[i] = val[i] / max_elem
        if count % 10  == 0:
            Xtest += [val[:-1]]
            Ytest += [val[-1]]
        else:
            X+= [val[:-1]]
            y+= [val[-1]]
        count += 1
    #print (Xtest)

    print ("training model")
    clf.fit(X, y)
    Yprime = clf.predict(Xtest)
    #print (Ytest)
    #print (Yprime)
    count = 0
    errors = [0]*11
    total = [0]*11
    for i in range(len(Ytest)):
        total[Ytest[i]]+=1
        if Ytest[i] != Yprime[i]:
            count+=1
            errors[Ytest[i]]+=1
    print (len(X))
    print ((len(Ytest) - count)/len(Ytest))
    print (total)
    print (errors)

    mix = [errors[i+1]/total[i+1] for i in range(len(total[1:]))]
    print (mix)
    print ("saving model")
    with open('SVM_Model2', 'wb') as load:
        pickle.dump(clf, load, protocol = 2)
