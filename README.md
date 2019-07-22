# Music classification by genre using machine learning 

I wrote this project for a college subject assignment. It uses two machine learning models (you can switch between them in the code)
kNN and a dense neural network model. The GUI part i wrote for fun and for learning Tkinter. The feature extraction and the dense model 
are both based on a [Medium article](https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d) on the Towards Data Science 
blog. The model is trained on the [GTZAN dataset](http://marsyas.info/downloads/datasets.html), which consists of 1000 audio tracks, 
each one 10 seconds long. There are 10 genres represented in the dataset with 100 tracks per genre. 

## Getting started 

There are a few things you need to do first, in order to run the code yourself

### Prerequisites

#### Python 

This project was written for Python 3.7, i cannot guarantee that it will work with other versions.
If you don't already have Python , the easiest way to install for Windows and Mac users
is to download Anaconda for the proper Python version. Linux users on the other hand can just run:

```
sudo apt-get install python3.7
sudo apt-get install python3-pip
``` 
in their terminal of choice

#### Libraries

When installing libraries in linux, in order to install them for python 3 
you need to use pip3 instead of just pip

This project depends on several libraries in order to work as intended:


##### LibROSA 

This library handles all the feature extraction from the audio files. 
If not currently installed on your system, run:

```
pip install librosa
``` 

in a Python shell.

##### Scikit Learn

This library is used to create and fit the kNN model and process it's output.
If not currently installed on your system, run:

```
pip install scikit-learn
```

in a Python shell.

##### Keras

This library is used to create and train the dense neural network model and to
normalize the input training data. 
If not currently installed on your system, run:

```
pip install tensorflow keras
```

in a Python shell.

##### Pandas

This library is used to manipulate the training data generated from processing
the audio files.
If not currently installed on your system, run:

```
pip install pandas
```

in a Python shell.

##### Tkinter

This library is used to create the GUI for the classifier.
It should come with every python distribution by default , 
but if not, run:

```
pip install python-tk
```

in a Python shell.


### Running the app

The application is launched by calling GUI.py from your
Python shell, for example:

```
python "path to project folder"/GUI.py
```

#### Supported music file types

The feature extraction code is made to work only on .wav files.
In order to broaden the range of supported file types, after 
loading the music file, the code calls an instance of ffmpeg
to convert the input file to .wav. The code does not require
you to have ffmpeg installed on your system and present in
your PATH. The project itself comes with ffmpeg and the code
calls it from a local directory in the project folder.
Therefore the app supports all the filetypes that ffmpeg supports
which means it supports all popular audio file formats(mp3, flac, aac..etc).

#### Training the model further

The results from the provided model are unfortunately unsatisfactory at best, with 
a 69% accuracy on the test dataset and terrible results in practical examples of 
categorizing actual popular songs into genres. In order to train a new model you 
need to either download the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset or provide 
your own dataset but format it the same way that this dataset is formatted.

## Project future and contributing

This project was only made for fun and learning, it is not meant to be used to provide
any sort of music categorization services, only as a means to get a laugh or two from 
it's terrible predictions. I will not continue to develop this project but you can feel
free to make any sort of changes you want (retrain the model, add features to the extractor..etc).