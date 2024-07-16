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

    model = tf.keras.models.load_model('./model_export.keras')

    tf_resized = tf.keras.layers.Resizing(32,32)
    tf_resized.build(input_shape=(124,129,1))

    tf_normalize = tf.keras.layers.Normalization()

    conv1Filters = model.layers[2].get_weights()[0]
    # print("1 FILTERS \n", conv1Filters.shape)
    conv1Biases =  model.layers[2].get_weights()[1]
    # print("1 BIASES \n", conv1Biases)
    conv1_layer = tf.keras.layers.Conv2D(32, 3, activation='relu', trainable=False)
    conv1_layer.build(input_shape=(None, 32, 32, 1))
    conv1_layer.set_weights([conv1Filters, conv1Biases])

    conv2Filters = model.layers[3].get_weights()[0]
    # print("2 FILTERS \n", conv2Filters.shape)
    conv2Biases =  model.layers[3].get_weights()[1]
    # print("2 BIASES \n", conv2Biases)
    conv2_layer = tf.keras.layers.Conv2D(64, 3, activation='relu', trainable=False)
    conv2_layer.build(input_shape=(None, 30, 30, 32))
    conv2_layer.set_weights([conv2Filters, conv2Biases])

    tf_maxpooling = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, trainable=False)
    tf_maxpooling.build(input_shape=(None, 28, 28, 64))

    tf_flatten = tf.keras.layers.Flatten(trainable=False)
    tf_flatten.build(input_shape=(None, 14, 14,64))

    tf_dense1_weights = model.layers[7].get_weights()[0]
    tf_dense1_biases = model.layers[7].get_weights()[1]
    tf_dense2_weights = model.layers[9].get_weights()[0]
    tf_dense2_biases = model.layers[9].get_weights()[1]

    tf_dense1 = tf.keras.layers.Dense(128, activation='relu', trainable=False)
    tf_dense1.build(input_shape=(12544,))
    tf_dense1.set_weights([tf_dense1_weights, tf_dense1_biases])

    tf_dense2 = tf.keras.layers.Dense(8, trainable=False)
    tf_dense2.build(input_shape=(128,))
    tf_dense2.set_weights([tf_dense2_weights, tf_dense2_biases])



    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
    print("\nGETTING AUDIO")
    image, label = get_wav_file('mini_speech_commands_sample/*/*.wav', labels)
    print("\nTURING AUDIO TO SPECTOGRAM AND RESIZE")
    image = spectrogram.get_spectrogram(image)
    tf_image = tf_resized(image)
    image = resize(image, 32, 32)
    # print("RESIZED")
    # print("TF IMAGE \n", tf_image)
    # print("IMAGE \n", image)

    print("\nNORMALIZATION")
    tf_normalize.adapt(tf_image)
    tf_image = tf_normalize(tf_image)
    image = Normalize.NormalizeSingle(image)
    # print("TF IMAGE \n", tf_image)
    # print("IMAGE \n", image)

    print("CONVOLUTION 1")
    tf_conv2d_input = tf.reshape(tf_image, [1, 32, 32, 1])
    tf_conv2d_output = conv1_layer(tf_conv2d_input)
    tf_conv2d_output = np.asarray(tf_conv2d_output)
    # print("tf conv2d output shape \n", tf_conv2d_output.shape)
    tf_conv2d_output_3d = tf.reshape(tf_conv2d_output, [32,30,30])
    # print("tf conv2d output shape after\n", tf_conv2d_output_3d.shape)
    counter = 1
    for i in tf_conv2d_output_3d:
         # Creating a directory for filter output if one does not exist
        if not os.path.exists('Conv2D_Test'):
            os.makedirs('Conv2D_Test')
        np.savetxt(f'Conv2D_Test/testOutput{counter}.csv',i,delimiter=',')
        counter += 1

    Convolution.Conv1(image)

    print("\nCONVOLUTION 2")
    tf_conv2d_output2 = conv2_layer(tf_conv2d_output)
    tf_conv2d_output2 = np.asarray(tf_conv2d_output2)
    # print("conv2d output 2 shape \n", tf_conv2d_output2.shape)
    tf_conv2d_output2_reshaped = tf.reshape(tf_conv2d_output2, [64,28,28])
    counter = 1
    for i in tf_conv2d_output2_reshaped:
        # Creating a directory for filter output if one does not exist
        if not os.path.exists('Conv2D_Test2'):
            os.makedirs('Conv2D_Test2')
        np.savetxt(f'Conv2D_Test2/testOutput{counter}.csv',i,delimiter=',')
        counter += 1

    Convolution2.Conv2()

    # def get_conv2_outputs():
    #     Conv2Output = np.array([])
    #     for i in range(64):
    #         Conv2 = np.array([])
    #         Input = f'./Conv2_Output/Conv2Output{i+1}.csv' #Check the Output files location
            
    #         # Appending the rows in one filter to the empty filter array and then applying it on the image using the applySingleFilter function
    #         with open(Input,newline='')as csvfile:
    #             reader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
    #             for row in reader:
    #                 Conv2 = np.append(Conv2, row)
    #             Conv2 = Conv2.reshape(28,28)
    #             # print(f"get_conv2_outputs {i}\n", Conv2)
    #         Conv2Output = np.append(Conv2Output, Conv2)

    #     Conv2Output = Conv2Output.reshape(1, 28, 28, 64)
    #     # print("get_conv2_outputs \n", Conv2Output)
    #     return Conv2Output

    print("\nMAXPOOLING AND FLATTEN")
    # tf_maxpooling_input = get_conv2_outputs()
    tf_image = tf_maxpooling(tf_conv2d_output2)
    image = MaxPooling2D.maxPool2D(2, 2, strides=2)  
    # print("MAXPOOLING")
    # print("TF IMAGE \n", tf_image)
    # print("IMAGE \n", image)
    # print("IMAGE SHAPE", image.shape)

    image = Flatten.flatten(image)
    tf_image = tf_flatten(tf_image)
    # print("FLATTEN")
    # print("TF IMAGE \n", tf_image)
    # print("IMAGE \n", image)

    print("\nFIRST DENSE")
    tf_image = tf_dense1(tf_image)
    image = dense.dense_1(image)    
    # print("DENSE 1")
    # print("TF IMAGE \n", tf_image)
    # print("IMAGE \n", image)

    print("\nSECOND DENSE")
    tf_result = tf_dense2(tf_image)
    result = dense.dense_2(image)
    # print("DENSE 2")
    # print("TF IMAGE \n", tf_image)
    # print("IMAGE \n", image)

    print("TF Result \n", tf_result[0])
    print(np.max(tf_result[0]))

    print("Result \n", result)
    print(np.max(result))

# def main():
#     labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
#     CM = np.zeros(shape=(8,8))
#     for correct in range(8):
#         for afile in range(7):
#             print("\nGETTING AUDIO")
#             fileIndex = (correct+1)*(afile+1)
#             image, label = get_wav_file('mini_speech_commands_sample/*/*.wav', labels, fileIndex)
#             print('AUDIOFILE SHAPE:',np.shape(image))
#             print("\nTURING AUDIO TO SPECTOGRAM AND RESIZE")
#             image = spectrogram.get_spectrogram(image)
#             print(np.shape(image))
#             #image = image[...,np.newaxis]
#             image = resize(image, 32, 32)
#             print("\nNORMALIZATION")
#             image = Normalize.NormalizeSingle(image)
#             print("CONVOLUTION 1")
#             Convolution.Conv1(image)
#             print("\nCONVOLUTION 2")
#             Convolution2.Conv2()
#             print("\nMAXPOOLING AND FLATTEN")
#             image = MaxPooling2D.maxPool2D(2, 2, strides=2)
#             image = Flatten.flatten(image)
#             print("\nFIRST DENSE")
#             image = dense.dense_1(image)
#             print("\nSECOND DENSE")
#             image = dense.dense_2(image)
#             print("\nRESULT",image)
#             print('SUM OF RESULTS:',np.sum(image))
#             winner = np.max(image)
#             print("WINNER IS:", winner)
#             winnerIndex = np.where(image==winner)[0][0]
                    
#             CM[correct,winnerIndex] += 1
#             print(CM)

#     return



if __name__=="__main__":
    main()