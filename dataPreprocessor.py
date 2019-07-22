# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:07:54 2019

@author: stefa
"""

from scipy.io import wavfile 
import numpy as np
import librosa


def loadDataAndSplit(loc, length):
    fs, wav=wavfile.read(loc)
    segmentLength=length*fs
    monowav=wav[:,0]/2+wav[:,1]/2
   
    
    filelist=[]
    for x in np.arange(int(segmentLength/2), len(wav)-int(1.5*segmentLength), segmentLength, dtype=int):
        
        filelist.append(monowav[x:x+segmentLength])
        
    return filelist, fs



def featureExtractor(wavlist, fs):
    dataFeatures=[]
    
    for x in wavlist:
        
        sampleFeatures=[]
        chroma_stft = librosa.feature.chroma_stft(y=x, sr=fs)
        rmse = librosa.feature.rmse(y=x)
        spec_cent = librosa.feature.spectral_centroid(y=x, sr=fs)
        spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=fs)
        rolloff = librosa.feature.spectral_rolloff(y=x, sr=fs)
        zcr = librosa.feature.zero_crossing_rate(x)
        mfcc = librosa.feature.mfcc(y=x, sr=fs)
        sampleFeatures.append(np.mean(chroma_stft))
        sampleFeatures.append(np.mean(rmse))
        sampleFeatures.append(np.mean(spec_cent))
        sampleFeatures.append(np.mean(spec_bw))
        sampleFeatures.append(np.mean(rolloff))
        sampleFeatures.append(np.mean(zcr))
        for e in mfcc:
            sampleFeatures.append(np.mean(e))
        
        
        dataFeatures.append(sampleFeatures)
    
    return dataFeatures
    
        

