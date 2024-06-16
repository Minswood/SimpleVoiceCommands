import wave 
import glob as glob
import numpy as np
import random

# Get the audio file from given path
# and convert it into numpy.float32 array
def get_wav_file(path, labels):
    audio_files = glob.glob(path)
    label_names = labels
    label_indices = []
    
    for i in range(0, len(labels), 1):
         label_indices = np.append(label_indices, [i]*(len(labels)-1))

    random_int = random.randint(0,55)
    filename = audio_files[random_int]
    # filename = 'mini_speech_commands_sample/Up/0ab3b47d_nohash_0.wav'
    print(filename)
    index = label_indices[random_int].astype(np.int64)
    label = label_names[index]
    print(label)
    file = wave.open(filename, "rb")
    samples = file.getnframes()
    audio = file.readframes(samples)
    # sampling frequency = how many data points per second
    # framerate = file.getframerate()

    audio_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_np_float32 = audio_np_int16.astype(np.float32)
    audio_normalised = normalize_audio(audio_np_float32)
    if(len(audio_normalised < 16000)):
        audio_normalised = pad_audio(audio_normalised)

    file.close()
    return audio_normalised, label

# Prep the audio file by making it a numpy array with 
# values scaled to between 1 and -1
def normalize_audio(audio):
    audio_normalised = audio / 32768
    print("audio normalized: ", audio_normalised.shape)
    return audio_normalised

def pad_audio(audio):
    zeros = np.zeros((16000 - len(audio)), dtype=np.float32)
    audio_padded = np.append(audio, zeros)
    print("padded: ", audio_padded.shape)
    return audio_padded