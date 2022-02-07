# Zero-Note Samba: Self-Supervised Beat Tracking

<p align="center">
        <img src="https://github.com/deezer/zeroNoteSamba/blob/main/images/flowchart.png" width="400">
</p>

by [Dorian Desblancs](https://www.linkedin.com/in/dorian-desblancs), [Vincent Lostanlen](https://www.lostanlen.com/), and [Romain Hennequin](http://romain-hennequin.fr/En/index.html).

## About

This repository contains the code used to generate the ZeroNS results. All experiment settings can be found in the `configuration/config.yaml` file. These include learning rates and evaluation modes for each downstream dataset, for example. For the pretext task, one can change the batch size and temperature parameters, among other elements.

## Getting Started

In order to explore the embeddings output by our ZeroNS model, one can install all the dependencies using 
```bash
pip install -r requirements.txt
```

after cloning into the repository. One can then get started with the following Python code snippet in order to explore the trained model's outputs:

```python
# Import functions and packages
import torch
import librosa
import IPython.display as ipd
from spleeter.separator import Separator

# Run Spleeter and create percussive and non-percussive tracks
separator = Separator('spleeter:4stems')

y, _ = librosa.load('/path/to/music/file', sr=44100)
stems = separator.separate(waveform=y)

drums = (stems['drums'][:, 0] + stems['drums'][:, 1]) / 2
other = (stems['other'][:, 0] + stems['other'][:, 1] \
        + stems['vocals'][:, 0] + stems['vocals'][:, 1] \
        + stems['bass'][:, 0] + stems['bass'][:, 1] ) / 2

drums = librosa.resample(drums, 44100, 16000)
other = librosa.resample(other, 44100, 16000)

drums_tensor = drums.reshape((1, 1, drums.shape[0], musdb.shape[1]))
other_tensor = other.reshape((1, 1, other.shape[0], musdb.shape[1]))

# Generate VQTs
vqt_postve = torch.from_numpy(IR.generate_XQT(drums_tensor, 16000, 'vqt'))
vqt_anchor = torch.from_numpy(IR.generate_XQT(other_tensor, 16000, 'vqt'))

# Load pretext task model weights
device = torch.device('cpu')
model = Down_CNN()
state_dict = torch.load("models/saved/shift_pret_cnn_16.pth", map_location=device)
model.pretext.load_state_dict(state_dict)
model.eval()

# Ouput percussive, non-percussive, and combined networks
percussive = model.pretext.postve(vqt_postve.float())
non_percussive = model.pretext.anchor(vqt_anchor.float())
combined = model(vqt_anchor.float(), vqt_postve.float())
```

The embeddings should resemble the following outputs:

- overlapped embeddings
<p align="center">
        <img src="https://github.com/deezer/zeroNoteSamba/blob/main/images/overlapped.png" width="800">
</p>

- overlapped percussive signal and embedding
<p align="center">
        <img src="https://github.com/deezer/zeroNoteSamba/blob/main/images/p_emb_sig.png" width="800">
</p>

- overlapped non-percussive signal and embedding
<p align="center">
        <img src="https://github.com/deezer/zeroNoteSamba/blob/main/images/np_emb_sig.png" width="800">
</p>

More code for getting started can be found in `drum_playground.ipynb` notebook.

## Advanced Usage

The pretext task code can be found in `pretext.py` and `fma_loader.py`. All processing for downstream tasks can be found in `ballroom.py`, `gtzan.py`, `hainsworth.py`, and `smc_mirex.py`. All downstream tasks can be found in the following:
- beat tracking: `beat_down.py`.
- cross-dataset generalization: `cross_data.py`.
- few-shot beat tracking: `data_exp.py`.

The code for information-theoretic measures on ZeroNS network can be found in `measures.py` and librosa's beat tracking method is in `old_school.py`. Finally, one can find the model and processing code in `models/` and `processing/`.

## Reference

If you use this repository, please consider citing:

```
To be completed upon publication of the paper.
```
