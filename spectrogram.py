import wave 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob as glob
import random
import scipy.fft


# Get the audio file from given path
# and convert it into numpy.float32 array
def get_wav_file():
    audio_files = glob.glob('mini_speech_commands_sample/*/*.wav')
    label_names = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
    label_indices = []
    
    for i in range(0,8,1):
         label_indices = np.append(label_indices, [i]*7)

    random_int = random.randint(0,55)
    filename = audio_files[random_int]
    # filename = 'mini_speech_commands_sample/Yes/0ac15fe9_nohash_0.wav'
    print(filename)
    index = label_indices[random_int].astype(np.int64)
    label = label_names[index]
    print(label)
    file = wave.open(audio_files[random_int], "rb")
    samples = file.getnframes()
    audio = file.readframes(samples)
    # sampling frequency = how many data points per second
    # framerate = file.getframerate()
    # print("framerate: %s", framerate)

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


def plot_waveform(waveform, label):
    plt.figure(figsize=(16, 10))
    plt.plot(waveform)
    plt.title(label)
    plt.yticks(np.arange(-1.2, 1.2, 0.2))
    plt.ylim([-1.1, 1.1])
    plt.show()

def get_coefficient(samples, n): 
    L = len(samples)
    k = np.arange(0, len(samples), 1)
    Xn = 0.0
    for i in range(len(samples)):
        Xn += (samples[i] * np.exp(1j * 2 * np.pi * k[i] * n / L)) 

    # Xn = np.sum(samples * np.exp(-2j * np.pi * k * n / L))
    return Xn

def get_dft(samples):
    coefficients = []
    for n in range(int(len(samples)/2)):
        coefficients.append(get_coefficient(samples, n))
    return coefficients

# get spectrogam function
def get_spectrogram(waveform, window_length=256, window_step=128):
    # waveform is the normalized audio data as np.array float32
    # window_lenght is the number of datapoints in each window
    # step is how many points to move the window over the waveform
    spectrogram = []
    steps = np.arange(0, len(waveform), window_step)
    for step in steps:
        # If stepping will make the window exceed the input array, stop
        if(step + window_length <= len(waveform)):
            # coefficients = scipy.fft.fft(waveform[step:step + window_length])
            coefficients = get_dft(waveform[step:step + window_length])
            spectrogram.append(coefficients)
        # else:
        #     diff = len(waveform) - (step + window_length)
        #     zeros = np.zeros(abs(diff))
        #     wf_array = waveform[step:]
        #     appended_array = np.append(wf_array, zeros)
        #     print("length of appended array ", len(appended_array))
        #     coefficients = get_dft(appended_array)
        #     spectrogram.append(coefficients)

    spectrogram = np.array(spectrogram)
    spectrogram = np.abs(spectrogram)
    spectrogram = spectrogram[..., np.newaxis]
    print(spectrogram.shape)
    step = 0
    return spectrogram

def tf_get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  print("tf waveform length ", len(waveform))
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128, window_fn=None)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
#   print(spectrogram.shape)
  return spectrogram

# Plotting function from Tenserflow Simple Audio tutorial
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def main():
    waveform, label = get_wav_file()
    # plot_waveform(waveform, label)
    spectrogram = get_spectrogram(waveform)
    tf_spectrogram = tf_get_spectrogram(waveform)

    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    # axes[0].plot(timescale, waveform.numpy())
    # axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])

    plot_spectrogram(spectrogram, axes[0])
    axes[0].set_title('Own Spectrogram   ' + label)
    plt.suptitle(label.title())

    plot_spectrogram(tf_spectrogram.numpy(), axes[1])
    axes[1].set_title('TF Spectrogram')
    plt.suptitle(label.title())
    plt.show()

if __name__=="__main__":
    main()