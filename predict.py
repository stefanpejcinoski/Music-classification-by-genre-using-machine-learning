import os
from classifier import loadPretrained, classifyData
from dataPreprocessor import loadDataAndSplit, featureExtractor

default_path="tempdata"
classes=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

data, fs=loadDataAndSplit(os.path.join(os.path.join(os.getcwd(), default_path), "tmp.wav"), 30)
dataFeatures=featureExtractor(data, fs)
model=loadPretrained()
prediction=classifyData(dataFeatures, model)
print("Prediciton is:"+classes[prediction])