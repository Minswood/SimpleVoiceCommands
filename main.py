import tensorflow as tf
from record_audio import PAStreamParams
from resize import resize
from fetch_audio import get_wav_file, get_recording
import Convolution
import Convolution2
import dense
import Normalize
import Flatten
import MaxPooling2D
import spectrogram
"""
[0.01203623 0.01131216 0.04545065 0.00425149 0.01260863 0.44165179
 0.10015564 0.3725334 ]
"""

if __name__=="__main__":
    dense1Weights, dense1Biases = dense.get_weights_and_biases('dense1Weights.csv', 'dense1Biases.csv')
    dense2Weights, dense2Biases = dense.get_weights_and_biases('dense2Weights.csv', 'dense2Biases.csv')

    dense1 = tf.keras.layers.Dense(128, activation='relu')
    dense2 = tf.keras.layers.Dense(8)

    dense1.build(input_shape=(12544,))
    dense2.build(input_shape=(128,))

    dense1.set_weights([dense1Weights, dense1Biases])
    dense2.set_weights([dense2Weights, dense2Biases])


    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
    image, label = get_wav_file('mini_speech_commands_sample/*/*.wav', labels)
    image = spectrogram.get_spectrogram(image)
    image = resize(image, 32, 32)
    image = Normalize.NormalizeSingle(image)
    Convolution.Conv1(image)
    Convolution2.Conv2()
    image = MaxPooling2D.maxPool2D(2, 2, strides=2)
    image = Flatten.flatten(image)

    # copy_image = image.copy()
    image = dense.dense_1(image)

    # tf_image = copy_image[tf.newaxis,...]
    # tf_image = dense1(tf_image)
    # print("own dense1 \n", image)
    # print("tf dense1 \n", tf_image)

    result = dense.dense_2(image)
    # tf_result = dense2(tf_image)
    print("own result \n", result)
    # print("Result \n", tf_result)