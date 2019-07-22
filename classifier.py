# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:26:28 2019

@author: stefa
"""

from keras.models import model_from_json
import os 
import numpy as np 
import pickle
from tensorflow.keras.utils import normalize


def loadkNN():
    model = pickle.load(open(os.path.join(os.getcwd(), "model.sav"), 'rb'))
    return model
        
def loadPretrained():
    json_file = open(os.path.join(os.getcwd(), 'model.json'), 'r')
    json_model = json_file.read()
    json_file.close()
    model = model_from_json(json_model)
    model.load_weights(os.path.join(os.getcwd(), 'weights.h5'))
    return model
        
        
def classifyData(featuresList, model):
    
    npFeaturesList=np.array(featuresList, dtype='float64')
    normalizedFeaturesList = normalize(npFeaturesList, axis=1)
    predictionVect=model.predict(normalizedFeaturesList)
    prediction=int(np.round(np.average(np.argmax(predictionVect, axis=0))))
    return prediction
        
        
        
        
        
        
    
