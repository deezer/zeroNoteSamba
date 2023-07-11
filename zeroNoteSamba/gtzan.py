import os
import pickle

import beat_down as BD
import data_exp as DE
import jams
import librosa as audio_lib
import numpy as np
import old_school as DP
import processing.input_rep as IR
import processing.source_separation as source_separation

# File imports
import processing.utilities as utils
import torch
import yaml
from spleeter.separator import Separator

if __name__ == "__main__":
    saved = True

    # Load YAML file configuations
    stream = open("configuration/config.yaml", "r")
    ymldict = yaml.safe_load(stream)

    gtzan_status = ymldict.get("gtzan_status")

    print("Loading audio and pulses...")

    if saved == False:
        # Load the separation model:
        model = ymldict.get("spl_mod")
        m = "spleeter:{}".format(model)
        separator = Separator(m)

        al = os.listdir("gtzan/GTZAN/")
        bl = os.listdir("gtzan/GTZAN-Rhythm_v2_ismir2015_lbd/jams")

        signals = {}
        beat_pulse = {}
        down_pulse = {}

        real_beat_times = {}
        real_down_times = {}

        wavs = []

        idx = 0
        for el in al:
            if "mf" in el:
                continue
            else:
                wav_fps = os.listdir("gtzan/GTZAN/" + el)

                for fp in wav_fps:
                    full_fp = "gtzan/GTZAN/" + el + "/" + fp

                    wavs.append(fp)

                    if gtzan_status == "pretrained":
                        sig = utils.convert_to_xxhz(full_fp, 44100)

                        print("{} :: {} -- {}".format(idx, fp, len(sig)))

                        temp_stems = source_separation.wv_run_spleeter(sig, 44100, separator, model)

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

                        signals[fp] = sigs

                        if idx == 0:
                            VQT1 = IR.generate_XQT(signals[fp][:, 0], 16000, "vqt")
                            VQT2 = IR.generate_XQT(signals[fp][:, 1], 16000, "vqt")

                            VQT = np.zeros((2, VQT1.shape[0], VQT1.shape[1]), dtype=float)
                            VQT[0, :, :] = VQT1[:, :]
                            VQT[1, :, :] = VQT2[:, :]

                            vqts = {}
                            print("VQT shape: ({} * {})".format(VQT.shape[1], VQT.shape[2]))
                            vqts[fp] = torch.from_numpy(VQT).float()
                            d_pulse = torch.zeros(VQT.shape[2])
                            b_pulse = torch.zeros(VQT.shape[2])

                        else:
                            VQT1 = IR.generate_XQT(signals[fp][:, 0], 16000, "vqt")
                            VQT2 = IR.generate_XQT(signals[fp][:, 1], 16000, "vqt")

                            VQT = np.zeros((2, VQT1.shape[0], VQT1.shape[1]))
                            VQT[0, :, :] = VQT1[:, :]
                            VQT[1, :, :] = VQT2[:, :]

                            print("VQT shape: ({} * {})".format(VQT.shape[1], VQT.shape[2]))
                            vqts[fp] = torch.from_numpy(VQT).float()
                            d_pulse = torch.zeros(VQT.shape[2])
                            b_pulse = torch.zeros(VQT.shape[2])

                    else:
                        sig = utils.preprocess(full_fp)

                        print("{} :: {} -- {}".format(idx, fp, len(sig)))

                        signals[fp] = sig

                        if idx == 0:
                            VQT = IR.generate_XQT(signals[fp], 16000, "vqt")
                            vqts = {}
                            print("VQT shape: ({} * {})".format(VQT.shape[0], VQT.shape[1]))
                            vqts[fp] = torch.from_numpy(VQT).float()
                            d_pulse = torch.zeros(VQT.shape[1])
                            b_pulse = torch.zeros(VQT.shape[1])

                        else:
                            VQT = IR.generate_XQT(signals[fp], 16000, "vqt")
                            print("VQT shape: ({} * {})".format(VQT.shape[0], VQT.shape[1]))
                            vqts[fp] = torch.from_numpy(VQT).float()
                            d_pulse = torch.zeros(VQT.shape[1])
                            b_pulse = torch.zeros(VQT.shape[1])

                    beats = "gtzan/GTZAN-Rhythm_v2_ismir2015_lbd/jams/" + fp + ".jams"

                    jam_temp = jams.load(beats).search(namespace="beat")

                    beat_tmz = []
                    down_tmz = []

                    for annotation in jam_temp:
                        if annotation["sandbox"]["annotation_type"] == "beat":
                            for dic in annotation["data"]:
                                beat_tmz.append(dic[0])
                                temp = round(62.5 * dic[0])

                                if temp >= len(b_pulse) - 2:
                                    temp = len(b_pulse) - 2

                                if temp == 0:
                                    temp = 1

                                b_pulse[temp] = 1
                                b_pulse[temp - 1] = 0.5
                                b_pulse[temp + 1] = 0.5

                        elif annotation["sandbox"]["annotation_type"] == "downbeat":
                            for dic in annotation["data"]:
                                down_tmz.append(dic[0])
                                temp = round(62.5 * dic[0])

                                if temp >= len(b_pulse) - 2:
                                    temp = len(b_pulse) - 2

                                if temp == 0:
                                    temp = 1

                                d_pulse[temp] = 1
                                d_pulse[temp - 1] = 0.5
                                d_pulse[temp + 1] = 0.5
                        else:
                            continue

                    real_beat_times[fp] = beat_tmz
                    real_down_times[fp] = down_tmz

                    down_pulse[fp] = d_pulse
                    beat_pulse[fp] = b_pulse

                    idx += 1

        with open("data/GTZAN/wavs.pkl", "wb") as handle:
            pickle.dump(wavs, handle, pickle.HIGHEST_PROTOCOL)

        if gtzan_status == "pretrained":
            with open("data/GTZAN/signals_spleeted.pkl", "wb") as handle:
                pickle.dump(signals, handle, pickle.HIGHEST_PROTOCOL)

        else:
            with open("data/GTZAN/signals_original.pkl", "wb") as handle:
                pickle.dump(signals, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/GTZAN/beat_pulses.pkl", "wb") as handle:
            pickle.dump(beat_pulse, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/GTZAN/down_pulses.pkl", "wb") as handle:
            pickle.dump(down_pulse, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/GTZAN/real_beat_times.pkl", "wb") as handle:
            pickle.dump(real_beat_times, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/GTZAN/real_down_times.pkl", "wb") as handle:
            pickle.dump(real_down_times, handle, pickle.HIGHEST_PROTOCOL)

        if gtzan_status == "pretrained":
            with open("data/GTZAN/vqts_spleeted.pkl", "wb") as handle:
                pickle.dump(vqts, handle, pickle.HIGHEST_PROTOCOL)

        else:
            with open("data/GTZAN/vqts_original.pkl", "wb") as handle:
                pickle.dump(vqts, handle, pickle.HIGHEST_PROTOCOL)

    else:
        with open("data/GTZAN/wavs.pkl", "rb") as handle:
            wavs = pickle.load(handle)

        if gtzan_status == "pretrained":
            with open("data/GTZAN/signals_spleeted.pkl", "rb") as handle:
                signals = pickle.load(handle)

        else:
            with open("data/GTZAN/signals_original.pkl", "rb") as handle:
                signals = pickle.load(handle)

        with open("data/GTZAN/beat_pulses.pkl", "rb") as handle:
            beat_pulse = pickle.load(handle)

        with open("data/GTZAN/down_pulses.pkl", "rb") as handle:
            down_pulse = pickle.load(handle)

        with open("data/GTZAN/real_beat_times.pkl", "rb") as handle:
            real_beat_times = pickle.load(handle)

        with open("data/GTZAN/real_down_times.pkl", "rb") as handle:
            real_down_times = pickle.load(handle)

        if gtzan_status == "pretrained":
            with open("data/GTZAN/vqts_spleeted.pkl", "rb") as handle:
                vqts = pickle.load(handle)

        else:
            with open("data/GTZAN/vqts_original.pkl", "rb") as handle:
                vqts = pickle.load(handle)

    _exp = ymldict.get("gtzan_exp")

    if gtzan_status == "old-school":
        DP.dp_ellis(wavs, signals, real_beat_times)

    elif _exp == "beat":
        _ = BD.train_model(wavs, vqts, beat_pulse, real_beat_times, "gtzan", ymldict)

    elif _exp == "perc":
        _ = DE.train_model(wavs, vqts, beat_pulse, real_beat_times, "gtzan", ymldict)

    else:
        print("YAML file for Gtzan data set has a bug in experiment definition!")
