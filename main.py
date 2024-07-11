import tensorflow as tf
from record_audio import PAStreamParams
from resize import resize
from fetch_audio import get_wav_file, get_recording, get_random_file
import Convolution
import Convolution2
import dense
import Normalize
import Flatten
import MaxPooling2D
import spectrogram
import numpy as np
"""
[0.01203623 0.01131216 0.04545065 0.00425149 0.01260863 0.44165179
 0.10015564 0.3725334 ]
"""
'''
def main():
    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
    print("\nGETTING AUDIO")
    image, label = get_wav_file('mini_speech_commands_sample/*/*.wav', labels)
    print("\nTURING AUDIO TO SPECTOGRAM AND RESIZE")
    image = spectrogram.get_spectrogram(image)
    image = resize(image, 32, 32)
    print("\nNORMALIZATION")
    image = Normalize.NormalizeSingle(image)
    print("CONVOLUTION 1")
    Convolution.Conv1(image)
    print("\nCONVOLUTION 2")
    Convolution2.Conv2()
    print("\nMAXPOOLING AND FLATTEN")
    image = MaxPooling2D.maxPool2D(2, 2, strides=2)
    image = Flatten.flatten(image)
    print("\nFIRST DENSE")
    image = dense.dense_1(image)
    print("\nSECOND DENSE")
    image = dense.dense_2(image)
    print("\nRESULT",image)
    print('SUM OF RESULTS:',np.sum(image))
    print("WINNER IS:", np.max(image))

    CM = np.zeros(8,8)
    
    return
'''
def main():
    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
    CM = np.zeros(shape=(8,8))
    for correct in range(8):
        for afile in range(7):
            print("\nGETTING AUDIO")
            fileIndex = (correct+1)*(afile+1)
            image, label = get_wav_file('mini_speech_commands_sample/*/*.wav', labels, fileIndex)
            print('AUDIOFILE SHAPE:',np.shape(image))
            print("\nTURING AUDIO TO SPECTOGRAM AND RESIZE")
            image = spectrogram.get_spectrogram(image)
            print(np.shape(image))
            #image = image[...,np.newaxis]
            image = resize(image, 32, 32)
            print("\nNORMALIZATION")
            image = Normalize.NormalizeSingle(image)
            print("CONVOLUTION 1")
            Convolution.Conv1(image)
            print("\nCONVOLUTION 2")
            Convolution2.Conv2()
            print("\nMAXPOOLING AND FLATTEN")
            image = MaxPooling2D.maxPool2D(2, 2, strides=2)
            image = Flatten.flatten(image)
            print("\nFIRST DENSE")
            image = dense.dense_1(image)
            print("\nSECOND DENSE")
            image = dense.dense_2(image)
            print("\nRESULT",image)
            print('SUM OF RESULTS:',np.sum(image))
            winner = np.max(image)
            print("WINNER IS:", winner)
            winnerIndex = np.where(image==winner)[0][0]
                    
            CM[correct,winnerIndex] += 1
            print(CM)

    return





if __name__=="__main__":
    main()
 