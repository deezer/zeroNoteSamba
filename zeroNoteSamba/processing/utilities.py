import librosa as audio_lib
import numpy as np


def convert_to_mono(signal):
    """
    Converts signal to mono.
    -- signal: 2D array with shape = 2 in dim1 or dim2
    """
    if len(signal.shape) == 2:
        if signal.shape[0] == 1:
            signal = np.reshape(signal, signal.shape[1])
        elif signal.shape[1] == 1:
            signal = np.reshape(signal, signal.shape[0])
        elif signal.shape[0] == 2:
            signal = (signal[0, :] + signal[1, :]) / 2
        else:
            signal = (signal[:, 0] + signal[:, 1]) / 2

    elif len(signal.shape == 1):
        return signal

    else:
        raise ("Signal is 3D+!")

    return signal


def convert_to_xxhz(f, sample_rate):
    """
    Function for downsampling wav files to sample_rate kHz. Writes re-named file to same directory.
    -- f: file name
    -- sample_rate: new sample rate
    """
    if f.endswith(".wav") or f.endswith(".mp3"):
        y, _ = audio_lib.load(f, sr=sample_rate)

        return y

    else:
        raise Exception("File is not a .wav or .mp3!")


def preprocess(fp):
    """
    Convert file to 16000 Hz.
    -- fp: file path
    """
    y = convert_to_xxhz(fp, 16000)
    y = np.reshape(y, y.shape[0])

    return y
