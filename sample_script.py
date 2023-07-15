# Import functions and packages
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
from spleeter.separator import Separator

import zeroNoteSamba.processing.input_rep as IR
from zeroNoteSamba.models.models import Down_CNN

# Run Spleeter and create percussive and non-percussive tracks
separator = Separator("spleeter:4stems")

y, _ = librosa.load("audio_example.mp3", sr=44100, mono=True)
stems = separator.separate(waveform=y.reshape((len(y), 1)))

drums = (stems["drums"][:, 0] + stems["drums"][:, 1]) / 2
other = (
    stems["other"][:, 0]
    + stems["other"][:, 1]
    + stems["vocals"][:, 0]
    + stems["vocals"][:, 1]
    + stems["bass"][:, 0]
    + stems["bass"][:, 1]
) / 2

drums = librosa.resample(y=drums, orig_sr=44100, target_sr=16000)
other = librosa.resample(y=other, orig_sr=44100, target_sr=16000)

# Generate VQTs
vqt_postve = torch.from_numpy(IR.generate_XQT(drums, 16000, "vqt"))
vqt_anchor = torch.from_numpy(IR.generate_XQT(other, 16000, "vqt"))

vqt_postve = vqt_postve.reshape(1, 1, vqt_postve.shape[0], vqt_postve.shape[1])
vqt_anchor = vqt_anchor.reshape(1, 1, vqt_anchor.shape[0], vqt_anchor.shape[1])

# Load pretext task model weights
device = torch.device("cpu")
model = Down_CNN()
state_dict = torch.load("zeroNoteSamba/models/saved/shift_pret_cnn_16.pth", map_location=device)
model.pretext.load_state_dict(state_dict)
model.eval()

# Ouput percussive, non-percussive, and combined networks
percussive = model.pretext.postve(vqt_postve.float())
non_percussive = model.pretext.anchor(vqt_anchor.float())
combined = model(vqt_anchor.float(), vqt_postve.float())

# Play each track
ipd.display(ipd.Audio(drums, rate=16000))  # type: ignore
ipd.display(ipd.Audio(other, rate=16000))  # type: ignore
ipd.display(ipd.Audio(librosa.resample(y=y, orig_sr=44100, target_sr=16000), rate=16000))  # type: ignore

# Plot overlapped signals
vqt_len = vqt_postve.shape[3]

plt.plot(non_percussive.detach().numpy().reshape(vqt_len))
plt.plot(percussive.detach().numpy().reshape(vqt_len))
plt.title("Overlapped Embeddings")
plt.legend(["Non-percussive", "Percussive"])
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.ylim((-0.1, 1))
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.show()

x1 = np.linspace(0, int(len(drums) / 16000), len(drums))
x2 = np.linspace(0, int(len(drums) / 16000), vqt_len)

plt.plot(x1, drums)
plt.plot(x2, percussive.detach().numpy().reshape(vqt_len))
plt.title("Overlapped Percussive Signal and Embedding")
plt.legend(["Signal", "Embedding"])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.ylim((-1, 1))
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
plt.show()

plt.plot(x1, other)
plt.plot(x2, non_percussive.detach().numpy().reshape(vqt_len))
plt.title("Overlapped Non-percussive Signal and Embedding")
plt.legend(["Signal", "Embedding"])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.ylim((-1, 1))
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
plt.show()
