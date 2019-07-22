# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:56:50 2019

@author: stefa
"""
import shutil
from tkinter import * 
from tkinter.ttk import *
import os
from tkinter import filedialog
import subprocess
from classifier import loadPretrained, classifyData, loadkNN
from dataPreprocessor import loadDataAndSplit, featureExtractor

fname=None
default_path="tempdata"

classes=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]




master=Tk()
v = StringVar()
Label(master, textvariable=v).pack()
v.set("Select music file then click Get genre")
v2 = StringVar()
Label(master, textvariable=v2).pack()


def callback():
    if fname==0:
        return
    v2.set(" ")
    try:  
        if os.path.exists(os.path.join(os.getcwd(), default_path)) is True:
            shutil.rmtree(os.path.join(os.getcwd(), default_path))
            os.mkdir(default_path)
        else:
            os.mkdir(default_path)
    except:
        v.set("Error while creating directories..")
    
    converter(fname)
  

def loadfile():
    global fname
    name=filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Open music/video file",
            filetypes=(("Music files", "*.mp3"),("Music files", "*.mp4"),("Music files", "*.wma"),("Music files", "*.flac"),("Music files", "*.ogg"), ("All files", "*.*")))
    if len(name)<1:
        return
    fname=name
    v2.set(os.path.basename(fname))
    

    
  

def converter(fname):
   
    v.set( "Converting..")
    
    cmds = [os.path.join(os.path.join(os.getcwd(), "converter"), "ffmpeg.exe"), '-i', fname, os.path.join(os.path.join(os.getcwd(), default_path), "tmp.wav")]
    prc = subprocess.Popen(cmds)
    prc.wait()
   
  
    v.set( "Done.. Calling classifier..")
    classification()
   
    
    
    
    
    
def classification():
    data, fs=loadDataAndSplit(os.path.join(os.path.join(os.getcwd(), default_path), "tmp.wav"), 30)
    dataFeatures=featureExtractor(data, fs)
    model=loadPretrained()
    prediction=classifyData(dataFeatures, model)
    v.set( "Genre is:"+" "+classes[prediction])
    v2.set(os.path.basename(fname))
    
  
	
b1 = Button(master, text="Open file", width=10, command=loadfile)
b1.pack()
b = Button(master, text="Get genre", width=10, command=callback)
b.pack()
master.title("Genre detector")
master.geometry("260x100")
master.maxsize(260, 100)
master.mainloop() 





