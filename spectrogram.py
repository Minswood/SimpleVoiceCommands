from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # For creating a spectrogram for comparison
import scipy.fft # For comparing to own function
from record_audio import PAStreamParams
from resize import resize
from fetch_audio import get_wav_file, get_recording

def dft(input_signal):
    N = len(input_signal)
    
    L = int(N/2) + 1
   
    coefficients = [] # The different frequencies in input signal (samples)
    windowed = []
    for i in range(0, N):
        window = 0.5 - 0.5 * (np.cos((2 * np.pi * i) / N)) # Applying Hanning window to smooth out "spectral leakage" caused by discontinuities in the non-periodic signals
        windowed.append(window)

    # iterate through the possible frequencies up until Nyquist limit
    for k in range(L):
        Xk = 0 # to store current frequency
        #iterate through the samples in input_signal
        for n in range(N):
            # Extract amplitude and phase information for kth frequency
            e = np.exp(-2j * np.pi * k * n / N) # Euler's formula: Xn * (np.cos(2 * np.pi * k * n / N) + 1j * np.sin(2 * np.pi * k * n / N))
            Xk += input_signal[n] * windowed[n] * e 
        coefficients.append(Xk)
    return np.array(coefficients)

def get_spectrogram(waveform, window_length=256, window_step=128):
    # waveform is the normalized audio data as np.array float32
    # window_lenght is the number of datapoints in each window
    # step is how many points to move the window over the waveform
    spectrogram = []
    try:
        steps = np.arange(0, len(waveform), window_step)
        for step in steps:
            # If stepping will make the window exceed the input array, stop
            if(step + window_length <= len(waveform)):
                coefficients = dft(waveform[step:step + window_length])
                spectrogram.append(coefficients)
    except Exception as e:
        print("Could not get spectrogram for waveform:", str(e))
    else:
        spectrogram = np.array(spectrogram)
        spectrogram = np.abs(spectrogram) # Get only magnitudes of the dft
        spectrogram = spectrogram[..., np.newaxis]
        print("spectrogram shape:", spectrogram.shape)
        return spectrogram

def tf_get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  print("tf waveform length ", len(waveform))
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128, )
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
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
  if np.size(spectrogram, 1) == 32:
      X = np.linspace(0, np.size(spectrogram, 1), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def main():
    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]

    waveform, label = get_wav_file('mini_speech_commands_sample/*/*.wav', labels)
    stream_params = PAStreamParams()
    rec_waveform = get_recording(duration=1, stream_params=stream_params)

    # plot_waveform(waveform, label)
    spectrogram = get_spectrogram(waveform)
    rec_spectrogram = get_spectrogram(rec_waveform)
    # tf_spectrogram = tf_get_spectrogram(waveform)
    resized_spec = resize(spectrogram, 32, 32)
    rec_resized_spec = resize(rec_spectrogram, 32, 32)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 4, wspace=0.4, hspace=0.4, height_ratios=[0.6, 0.6, 1])
    # ax0 = fig.add_subplot(gs[0, :])
    # timescale = np.arange(waveform.shape[0])
    # ax0.plot(timescale, waveform)
    # ax0.set_title("Waveform")
    # ax0.set_xlim([0, 16000])

    ax0 = fig.add_subplot(gs[0, :])
    plot_spectrogram(spectrogram, ax0)
    ax0.set_title("File spectrogram")
    plt.suptitle(label.title())
    ax0.set_xlim([0, 16000])

    ax1 = fig.add_subplot(gs[1, :])
    plot_spectrogram(rec_spectrogram, ax1)
    ax1.set_title("Recorded spectrogram")
    ax1.set_xlim([0, 16000])

    ax2 = fig.add_subplot(gs[2, : -3])
    plot_spectrogram(resized_spec, ax2)
    ax2.set_title("File resized")

    ax3 = fig.add_subplot(gs[2, -1 :])
    plot_spectrogram(rec_resized_spec, ax3)
    ax3.set_title("Recorded resized")
   
    plt.show()

if __name__=="__main__":
    main()