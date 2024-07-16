import tensorflow as tf
import numpy as np
import fetch_audio
import spectrogram
from resize import resize
from keras import layers
import Normalize

model = tf.keras.models.load_model('./model_export.keras')

# for layer in model.layers:  print(layer.get_config(), layer.get_weights())
# model.summary()


labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
audio, labels = fetch_audio.get_wav_file('mini_speech_commands_sample/*/*.wav', labels)
audio = spectrogram.get_spectrogram(audio)

audio = audio[tf.newaxis,...]
# print(audio)
prediction = model(audio)
print(prediction[0])
print(np.max(prediction[0]))

# prediction = model.predict(audio)
# print("prediction", prediction)

# audio = resize(audio,32,32)

# # print("after resize \n", audio)
# layer = layers.Normalization(axis=None)
# layer.adapt(audio)
# normalized_data = layer(audio)
# own_normalized = Normalize.NormalizeSingle(audio)
# print("tf normalized \n", normalized_data)
# print("Own normalized \n", own_normalized)