import yaml
import torch
import pickle
import numpy as np
import librosa as audio_lib

from spleeter.separator import Separator

# File imports
import processing.utilities as utils
import processing.input_rep as IR
import beat_down as BD
import data_exp as DE
import old_school as DP

import processing.source_separation as source_separation


if __name__ == "__main__":
    save = True

    # Load YAML file configuations
    stream = open("configuration/config.yaml", "r")
    ymldict = yaml.safe_load(stream)

    hainsworth_status = ymldict.get("hainsworth_status")

    print("Loading audio and pulses...")

    if save == False:
        # Load the separation model:
        model = ymldict.get("spl_mod")
        m = "spleeter:{}".format(model)
        separator = Separator(m)

        with open("hainsworth/data.txt", "r") as fp:
            songs = fp.readlines()

        wavs = []
        beats = []
        downs = []

        idx = 0
        for el in songs:
            if idx > 12:
                l = el.split("<sep>")

                wav = l[0]
                wav = wav.replace("\t", "")
                wav = wav.replace("\n", "")
                wav = wav.replace(" ", "")
                wavs.append(wav)

                beat = l[10]
                beat = beat.replace("\t", "")
                beat = beat.replace("\n", "")
                beat = beat.replace(" ", "")
                beats.append(beat)

                down = l[11]
                down = down.replace("\t", "")
                down = down.replace("\n", "")
                down = down.replace(" ", "")
                downs.append(down)

            idx += 1

        num_songs = 222

        signals = {}
        beat_pulse = {}
        down_pulse = {}

        real_beat_times = {}
        real_down_times = {}

        idx = 0
        for el in wavs:
            file_path = "hainsworth/wavs/" + el

            if hainsworth_status == "pretrained":
                sig = utils.convert_to_xxhz(file_path, 44100)

                print("{} -- {} :: {}".format(idx, file_path, len(sig)))

                temp_stems = source_separation.wv_run_spleeter(
                    sig, 44100, separator, model
                )

                anchor = None
                for name, sig in temp_stems.items():
                    if name == "drums":
                        possignal = np.zeros(sig.shape)
                        possignal[:, :] = sig[:, :]

                    else:
                        if anchor is None:
                            anchor = np.zeros(sig.shape)
                            anchor[:, :] = sig[:, :]

                        else:
                            anchor[:, :] += sig[:, :]

                anchor = utils.convert_to_mono(anchor)
                anchor = audio_lib.resample(anchor, 44100, 16000)
                possignal = utils.convert_to_mono(possignal)
                possignal = audio_lib.resample(possignal, 44100, 16000)

                sigs = np.zeros((anchor.shape[0], 2))
                sigs[:, 0] = anchor[:]
                sigs[:, 1] = possignal[:]

                signals[el] = sigs

                if idx == 0:
                    VQT1 = IR.generate_XQT(signals[el][:, 0], 16000, "vqt")
                    VQT2 = IR.generate_XQT(signals[el][:, 1], 16000, "vqt")

                    VQT = np.zeros((2, VQT1.shape[0], VQT1.shape[1]), dtype=float)
                    VQT[0, :, :] = VQT1[:, :]
                    VQT[1, :, :] = VQT2[:, :]

                    vqts = {}
                    print("VQT shape: ({} * {})".format(VQT.shape[1], VQT.shape[2]))
                    vqts[el] = torch.from_numpy(VQT).float()
                    d_pulse = torch.zeros(VQT.shape[2])
                    b_pulse = torch.zeros(VQT.shape[2])

                else:
                    VQT1 = IR.generate_XQT(signals[el][:, 0], 16000, "vqt")
                    VQT2 = IR.generate_XQT(signals[el][:, 1], 16000, "vqt")

                    VQT = np.zeros((2, VQT1.shape[0], VQT1.shape[1]))
                    VQT[0, :, :] = VQT1[:, :]
                    VQT[1, :, :] = VQT2[:, :]

                    print("VQT shape: ({} * {})".format(VQT.shape[1], VQT.shape[2]))
                    vqts[el] = torch.from_numpy(VQT).float()
                    d_pulse = torch.zeros(VQT.shape[2])
                    b_pulse = torch.zeros(VQT.shape[2])

            else:
                sig = utils.preprocess(file_path)

                print("{} -- {} :: {}".format(idx, file_path, len(sig)))

                signals[el] = sig[:]

                if idx == 0:
                    VQT = IR.generate_XQT(signals[el], 16000, "vqt")
                    vqts = {}
                    print("VQT shape: ({} * {})".format(VQT.shape[0], VQT.shape[1]))
                    vqts[el] = torch.from_numpy(VQT).float()
                    d_pulse = torch.zeros(VQT.shape[1])
                    b_pulse = torch.zeros(VQT.shape[1])

                else:
                    VQT = IR.generate_XQT(signals[el], 16000, "vqt")
                    print("VQT shape: ({} * {})".format(VQT.shape[0], VQT.shape[1]))
                    vqts[el] = torch.from_numpy(VQT).float()
                    d_pulse = torch.zeros(VQT.shape[1])
                    b_pulse = torch.zeros(VQT.shape[1])

            beat = beats[idx]
            down = downs[idx]

            beat = np.asarray(beat.split(","), dtype=float)
            down = np.asarray(down.split(","), dtype=int)

            beat_tmz = []
            down_tmz = []

            for xx in range(len(beat)):
                b = beat[xx] / 44100
                d = xx + 1

                beat_tmz.append(b)
                temp = int(round(62.5 * b))

                if temp >= len(b_pulse) - 2:
                    temp = len(b_pulse) - 2

                if temp == 0:
                    temp = 1

                if d in down:
                    down_tmz.append(b)
                    d_pulse[temp] = 1
                    d_pulse[temp - 1] = 0.5
                    d_pulse[temp + 1] = 0.5

                b_pulse[temp] = 1
                b_pulse[temp - 1] = 0.5
                b_pulse[temp + 1] = 0.5

            real_beat_times[el] = beat_tmz
            real_down_times[el] = down_tmz

            down_pulse[el] = d_pulse
            beat_pulse[el] = b_pulse

            idx += 1

        with open("data/Hainsworth/wavs.pkl", "wb") as handle:
            pickle.dump(wavs, handle, pickle.HIGHEST_PROTOCOL)

        if hainsworth_status == "pretrained":
            with open("data/Hainsworth/signals_spleeted.pkl", "wb") as handle:
                pickle.dump(signals, handle, pickle.HIGHEST_PROTOCOL)

        else:
            with open("data/Hainsworth/signals_original.pkl", "wb") as handle:
                pickle.dump(signals, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/Hainsworth/beat_pulses.pkl", "wb") as handle:
            pickle.dump(beat_pulse, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/Hainsworth/down_pulses.pkl", "wb") as handle:
            pickle.dump(down_pulse, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/Hainsworth/real_beat_times.pkl", "wb") as handle:
            pickle.dump(real_beat_times, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/Hainsworth/real_down_times.pkl", "wb") as handle:
            pickle.dump(real_down_times, handle, pickle.HIGHEST_PROTOCOL)

        if hainsworth_status == "pretrained":
            with open("data/Hainsworth/vqts_spleeted.pkl", "wb") as handle:
                pickle.dump(vqts, handle, pickle.HIGHEST_PROTOCOL)

        else:
            with open("data/Hainsworth/vqts_original.pkl", "wb") as handle:
                pickle.dump(vqts, handle, pickle.HIGHEST_PROTOCOL)

    else:
        with open("data/Hainsworth/wavs.pkl", "rb") as handle:
            wavs = pickle.load(handle)

        if hainsworth_status == "pretrained":
            with open("data/Hainsworth/signals_spleeted.pkl", "rb") as handle:
                signals = pickle.load(handle)

        else:
            with open("data/Hainsworth/signals_original.pkl", "rb") as handle:
                signals = pickle.load(handle)

        with open("data/Hainsworth/beat_pulses.pkl", "rb") as handle:
            beat_pulse = pickle.load(handle)

        with open("data/Hainsworth/down_pulses.pkl", "rb") as handle:
            down_pulse = pickle.load(handle)

        with open("data/Hainsworth/real_beat_times.pkl", "rb") as handle:
            real_beat_times = pickle.load(handle)

        with open("data/Hainsworth/real_down_times.pkl", "rb") as handle:
            real_down_times = pickle.load(handle)

        if hainsworth_status == "pretrained":
            with open("data/Hainsworth/vqts_spleeted.pkl", "rb") as handle:
                vqts = pickle.load(handle)

        else:
            with open("data/Hainsworth/vqts_original.pkl", "rb") as handle:
                vqts = pickle.load(handle)

    _exp = ymldict.get("hainsworth_exp")

    if hainsworth_status == "old-school":
        DP.dp_ellis(wavs, signals, real_beat_times)

    elif _exp == "beat":
        _ = BD.train_model(
            wavs, vqts, beat_pulse, real_beat_times, "hainsworth", ymldict
        )

    elif _exp == "perc":
        _ = DE.train_model(
            wavs, vqts, beat_pulse, real_beat_times, "hainsworth", ymldict
        )

    else:
        print("YAML file for Hainsworth data set has a bug in experiment definition!")
