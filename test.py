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

file = 'histograms_test1'

svm_path = 'SVM_model10'
svm = pickle.load(open(svm_path, 'rb'))

with open(file, 'rb') as f:
    print ("reading")
    my_list = pickle.load(f)

    X = []
    y = []
    for val in my_list:
        X+= [val[:-1]]
        y+= [val[-1]]
    #print (Xtest)

    print ("training model")

    Yprime = svm.predict(X)
    #print (Ytest)
    #print (Yprime)
    count = 0
    errors = [0]*11
    total = [0]*11
    for i in range(len(y)):
        total[y[i]]+=1
        if y[i] != Yprime[i]:
            count+=1
            errors[y[i]]+=1
    print (len(X))
    print ((len(y) - count)/len(y))
    print (total[1:])
    print (errors[1:])
    mix = [errors[i+1]/total[i+1] for i in range(len(total[1:]))]
    print (mix)
