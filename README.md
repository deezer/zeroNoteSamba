# Zero-Note Samba: Self-Supervised Beat Tracking

<p align="center">
        <img src="https://github.com/deezer/zeroNoteSamba/blob/main/images/flowchart.png" width="400">
</p>

by [Dorian Desblancs](https://www.linkedin.com/in/dorian-desblancs), [Vincent Lostanlen](https://www.lostanlen.com/), and [Romain Hennequin](http://romain-hennequin.fr/En/index.html).

## About

This repository contains the code used to generate the ZeroNS results. All experiment settings can be found in the `zeroNoteSamba/configuration/config.yaml` file. These include learning rates and evaluation modes for each downstream dataset, for example. For the pretext task, one can change the batch size and temperature parameters among other elements.

## Getting Started

In order to explore the embeddings output by our ZeroNS model, one can start with the following:
```bash
# Clone and enter repository
git clone https://github.com/deezer/zeroNoteSamba.git
cd zeroNoteSamba

# Install dependencies
pip install poetry
poetry install

# Unzip model weights file
gzip -d zeroNoteSamba/models/saved/shift_pret_cnn_16.pth.gz

# Download sample audio example
wget https://github.com/deezer/spleeter/raw/master/audio_example.mp3
```
Note that Spleeter dependencies can be a little tricky to install. Please consult the package's [repository](https://github.com/deezer/spleeter) for proper installation instructions.

One can then get started with the following Python code snippet in order to explore the trained model's outputs (also present in `sample_script.py`):

```python
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
```

The resulting embeddings should resemble the following outputs:

- overlapped embeddings.
<p align="center">
        <img src="https://github.com/deezer/zeroNoteSamba/blob/main/images/overlapped.png" width="600">
</p>

- overlapped percussive signal and embedding.
<p align="center">
        <img src="https://github.com/deezer/zeroNoteSamba/blob/main/images/p_emb_sig.png" width="600">
</p>

- overlapped non-percussive signal and embedding.
<p align="center">
        <img src="https://github.com/deezer/zeroNoteSamba/blob/main/images/np_emb_sig.png" width="600">
</p>

The script and code for plotting the embeddings can be found in this [Colab notebook](https://colab.research.google.com/drive/1bdS_-SSQJalvLVNT4rtMRNeKFDuFnCAR?usp=sharing#scrollTo=avSYHOHNJEg-).

## Advanced Usage

The pretext task code can be found in `zeroNoteSamba/pretext.py` and `zeroNoteSamba/fma_loader.py`. All dataset processing for beat-tracking-related tasks can be found in `zeroNoteSamba/ballroom.py`, `zeroNoteSamba/gtzan.py`, `zeroNoteSamba/hainsworth.py`, and `zeroNoteSamba/smc_mirex.py`. All downstream tasks can be found in the following:
- beat tracking: `zeroNoteSamba/beat_down.py`.
- cross-dataset generalization: `zeroNoteSamba/cross_data.py`.
- few-shot beat tracking: `zeroNoteSamba/data_exp.py`.

The code to generate the information-theoretic measures of ZeroNS network embeddings can be found in `zeroNoteSamba/measures.py` and librosa's beat tracking method is in `zeroNoteSamba/old_school.py`. Finally, one can find the model and processing code in `zeroNoteSamba/models/` and `zeroNoteSamba/processing/`.

## Reference

If you use this repository, please consider citing:

```
@article{desblancs2023zero,
  title={Zero-Note Samba: Self-Supervised Beat Tracking},
  author={Desblancs, Dorian and Lostanlen, Vincent and Hennequin, Romain},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2023},
  publisher={IEEE}
}
```
