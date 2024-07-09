import tensorflow as tf
import numpy as np
import fetch_audio
import spectrogram
from resize import resize
from keras import layers
import Normalize

model = tf.keras.models.load_model('./model_export.keras')
# model.summary()

labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
audio, labels = fetch_audio.get_wav_file('mini_speech_commands_sample/*/*.wav', labels)
audio = spectrogram.get_spectrogram(audio)
audio = resize(audio,32,32)
# print("after resize \n", audio)
layer = layers.Normalization()
layer.adapt(audio)
normalized_data = layer(audio)
own_normalized = Normalize.NormalizeSingle(audio)
print("tf normalized \n", normalized_data)
print("Own normalized \n", own_normalized)