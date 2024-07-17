import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
import csv

def main():
    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
    print("\nGETTING AUDIO")
    # stream_params = PAStreamParams()
    # image = get_recording(duration=1, stream_params=stream_params)
    image, label = get_wav_file('mini_speech_commands_sample/*/*.wav', labels, 0)
    print("\nTURNING AUDIO TO SPECTOGRAM AND RESIZE")
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
    output_array = np.zeros([14,14,64])
    for i in range(14):
        for j in range(14):
            for k in range(64):
                output_array[i][j][k] = image[k][i][j]

    image = Flatten.flatten(output_array)

    print("\nFIRST DENSE")
    image = dense.dense_1(image)   

    print("\nSECOND DENSE")
    result = dense.dense_2(image)

    # print(f'\nCOMMAND WAS "{label}"\n')
    print("RESULT \n", result)
    print("MAX RESULT", np.max(result))



# def main():
#     labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
#     CM = np.zeros(shape=(8,8))
#     correct = 0
#     for index in range(56):
#         print("\nGETTING AUDIO")
#         image, label = get_wav_file('mini_speech_commands_sample/*/*.wav', labels, index+1)
#         print('AUDIOFILE SHAPE:',np.shape(image))
#         print("\nTURING AUDIO TO SPECTOGRAM AND RESIZE")
#         image = spectrogram.get_spectrogram(image)
#         print(np.shape(image))
#         #image = image[...,np.newaxis]
#         image = resize(image, 32, 32)
#         print("\nNORMALIZATION")
#         image = Normalize.NormalizeSingle(image)
#         print("CONVOLUTION 1")
#         Convolution.Conv1(image)
#         print("\nCONVOLUTION 2")
#         Convolution2.Conv2()
#         print("\nMAXPOOLING AND FLATTEN")
#         image = MaxPooling2D.maxPool2D(2, 2, strides=2)
#         output_array2 = np.zeros([14,14,64])
#         for i in range(14):
#             for j in range(14):
#                 for k in range(64):
#                     output_array2[i][j][k] = image[k][i][j]
#         image = Flatten.flatten(output_array2)
#         print("\nFIRST DENSE")
#         image = dense.dense_1(image)
#         print("\nSECOND DENSE")
#         image = dense.dense_2(image)
#         print("\nRESULT",image)
#         print('SUM OF RESULTS:',np.sum(image))
#         winner = np.max(image)
#         print("WINNER IS:", winner)
#         winnerIndex = np.where(image==winner)[0][0]
        
#         if(index<7 and index>=0):
#                 correct = 0
#         elif(index <14 and index>=7):
#                 correct = 1
#         elif(index <21 and index>=14):
#                 correct = 2
#         elif(index <28 and index>=21):
#                 correct = 3
#         elif(index <35 and index>=28):
#                 correct = 4
#         elif(index <42 and index>=35):
#                 correct = 5
#         elif(index <49 and index>=42):
#                 correct = 6
#         elif(index <56 and index>=49):
#                 correct = 7
                
#         CM[correct,winnerIndex] += 1
#         print(CM)

#     return



if __name__=="__main__":
    main()