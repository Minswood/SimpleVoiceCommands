import wave 
import numpy as np
from dataclasses import dataclass, asdict
import pyaudio

@dataclass
class PAStreamParams:
    """The parameters needed for creating a PyAudio Stream.
    Frames_per_buffer = Number of frames saved at a time.
    format = PortAudio sample format.
    channels = Number of channels.
    Rate = Sampling rate. Must be the same as the one used in training the model.
    input = True, if Stream is input.
    """
    frames_per_buffer: int = 3200 
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    input: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

class Recorder:
    """Uses PyAudio to record from device microphone. 
    Attribute stream_params is a PAStreamParams object 
    containing the values for recording from a pyaudio Stream.
    """
    def __init__(self, stream_params: PAStreamParams) -> None:
        self.stream_params = stream_params
        self._pyaudio = None
        self._stream = None
        self._frames = []

    def record(self, duration: int=1):
        """Records audio from a microphone and returns a numpy.ndarray.
        By default the length of the recording is one second.
        """
        print("Start recording")
        
        self._create_stream()
        recorded_audio = self._write_frames_from_stream(duration)
        self._close_recording()
        print("Stop recording")
        recorded_audio = np.frombuffer(b''.join(self._frames), dtype=np.int16)
        return recorded_audio
        
    def _create_stream(self) -> None:
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())

    def _write_frames_from_stream(self, duration: int):
        for _ in range(int(self.stream_params.rate / self.stream_params.frames_per_buffer * duration)):
            data = self._stream.read(self.stream_params.frames_per_buffer)
            self._frames.append(data)
        return np.frombuffer(b''.join(self._frames), dtype=np.int16)

    def _close_recording(self) -> None:
        self._stream.close()
        self._pyaudio.terminate()