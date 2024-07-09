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
    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
    image, label = get_wav_file('mini_speech_commands_sample/*/*.wav', labels)
    image = spectrogram.get_spectrogram(image)
    image = resize(image, 32, 32)
    image = Normalize.NormalizeSingle(image)
    Convolution.Conv1(image)
    Convolution2.Conv2()
    image = MaxPooling2D.maxPool2D(2, 2, strides=2)
    image = Flatten.flatten(image)
    image = dense.dense_1(image, 128)
    image = dense.dense_2(image, 8)
    print(image)