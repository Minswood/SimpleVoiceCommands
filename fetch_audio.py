import wave 
import glob as glob
import numpy as np
import random
from record_audio import PAStreamParams, Recorder

def get_random_file(path, random_int):
    try:
        audio_files = glob.glob(path)
        filename = audio_files[random_int]
    except FileNotFoundError as e:
        print("get_random_file:", str(e))
    except IndexError as e:
        print("get_random_file:", str(e))
    else: 
        return filename
    
def generate_label_indices(labels, num_items):
    label_indices = []
    for i in range(0, len(labels), 1):
         label_indices = np.append(label_indices, [i]*num_items)
    return label_indices

def read_file(filename):
    try:
        file = wave.open(filename, "rb")
        samples = file.getnframes()
        audio = file.readframes(samples)
    except Exception as e:
        print("read_file:", e)
    else:
        file.close()
        return audio

def check_audio_length(audio):
    if(len(audio) < 16000):
        audio = pad_audio(audio)
    elif(len(audio) > 16000):
        audio = crop_audio(audio)
    return audio

def get_wav_file(path, labels, fileIndex):
    num_items = 7 # Number of audio files in each speech command folder
    label_indices = generate_label_indices(labels, num_items)
    random_int = random.randint(0, (len(labels) * num_items) - 1)
    if fileIndex > 0:
        filename = get_random_file(path, fileIndex)
        index = label_indices[fileIndex-1].astype(np.int64)
    else:
        filename = get_random_file(path, random_int)
        index = label_indices[random_int].astype(np.int64)
        
    # filename = 'mini_speech_commands_sample/Up/0ab3b47d_nohash_0.wav'
    print(filename)
    
    label = labels[index]
    print(label)
    # print("Up")

    audio = read_file(filename)
    audio_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_np_float32 = audio_np_int16.astype(np.float32)
    audio_normalised = normalize_audio(audio_np_float32)

    audio_normalised = check_audio_length(audio_normalised)

    return audio_normalised, label

def get_recording(duration: int, stream_params: PAStreamParams):
    recorder = Recorder(stream_params)
    rec_audio_int16 = recorder.record(duration=duration)
    rec_audio_float32 = rec_audio_int16.astype(np.float32)
    rec_audio_normalised = normalize_audio(rec_audio_float32)
    rec_audio_normalised = check_audio_length(rec_audio_normalised)
    return rec_audio_normalised

def normalize_audio(audio):
    """Prep the audio by making it a numpy array with 
    values scaled to between 1 and -1.
    """
    audio_normalised = audio / 32768
    return audio_normalised

def pad_audio(audio):
    zeros = np.zeros((16000 - len(audio)), dtype=np.float32)
    audio_padded = np.append(audio, zeros)
    return audio_padded

def crop_audio(audio):
    audio_length = len(audio)
    diff = audio_length - 16000
    required_length = audio_length - diff
    audio_cropped = audio[:required_length]
    return audio_cropped