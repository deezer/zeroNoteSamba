import os
import numpy as np 
import matplotlib.pyplot as plt

import librosa as audio_lib
import librosa.display as display


def generate_XQT(signal, sample_rate, mode):
    """
    Generates a high-resolution XQT spectrogram.
    -- signal      : signal to compute XQT on
    -- sample_rate : self-explanatory 
    -- mode        : can be either vqt or cqt
    """
    hop_length  = 256
    first_note  = 'C0'
    octave_reso = 12
    num_octaves = 8
    eps         = 10e-10

    fmin = audio_lib.note_to_hz(first_note)

    if (mode == "cqt"):
        CQT  = audio_lib.cqt(y=signal, sr=sample_rate, hop_length=hop_length, fmin=fmin, 
                            n_bins=num_octaves*octave_reso, bins_per_octave=octave_reso)

        CQT = np.abs(CQT)
        CQT = np.log(CQT + eps)

        return CQT

    elif (mode == "vqt"):
        VQT  = audio_lib.vqt(y=signal, sr=sample_rate, hop_length=hop_length, fmin=fmin, 
                            n_bins=num_octaves*octave_reso, bins_per_octave=octave_reso)

        VQT = np.abs(VQT)
        VQT = np.log(VQT + eps)

        return VQT

    else:
        raise Exception("Mode can only be vqt or cqt!")
    

def plot_XQT(cqt, sample_rate, title=None, save=None):
    """
    Function for plotting CQT.
    -- cqt         : Constant Q-Transform matrix
    -- sample_rate : SR in Hz
    -- title       : plot title
    """
    fig, ax = plt.subplots()
    dB = audio_lib.amplitude_to_db(cqt, ref=np.max)
    img = display.specshow(data=dB, sr=sample_rate, x_axis='time', y_axis='cqt_note', ax=ax)

    if (title is not None):
        ax.set_title(title)
    else:
        ax.set_title('Power spectrum')

    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    if (save == None):
        plt.show()

    else:
        os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/" + save + '.pdf', dpi=200, format='pdf')

    return