import os
import yaml
import pickle
import torch
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

    ballroom_status = ymldict.get("ballroom_status")

    print("Loading audio and pulses...")

    if save == False:
        # Load the separation model:
        model = ymldict.get("spl_mod")
        m = "spleeter:{}".format(model)
        separator = Separator(m)

        duplicates = [
            "Albums-AnaBelen_Veneo-11",
            "Albums-Fire-08",
            "Albums-Latin_Jam2-05",
            "Albums-Secret_Garden-01",
            "Albums-AnaBelen_Veneo-03",
            "Albums-Ballroom_Magic-03",
            "Albums-Latin_Jam-04",
            "Albums-Latin_Jam-08",
            "Albums-Latin_Jam-06",
            "Albums-Latin_Jam2-02",
            "Albums-Latin_Jam2-07",
            "Albums-Latin_Jam3-02",
            "Media-103402",
            "README",
        ]

        al = [
            "ChaChaCha/",
            "Jive/",
            "Quickstep/",
            "Rumba-American/",
            "Rumba-International/",
            "Rumba-Misc/",
            "Samba/",
            "Tango/",
            "VienneseWaltz/",
            "Waltz/",
        ]

        bl = os.listdir("BallroomData/BallroomAnnotations-master/")

        audio_list = []

        idx = 0
        for el in al:
            temp = os.listdir("BallroomData/" + el)

            for song in temp:
                status = False

                if "._" in song:
                    status = True

                for dup in duplicates:
                    if dup in song:
                        status = True

                if status == True:
                    continue
                else:
                    audio_list.append(["BallroomData/" + el, song])
                    idx += 1

        num_songs = len(audio_list)

        signals = {}
        beat_pulse = {}
        down_pulse = {}

        real_beat_times = {}
        real_down_times = {}

        wavs = []

        idx = 0

        while audio_list != []:
            dir, audio = audio_list.pop()

            wavs.append(audio)

            if ballroom_status == "pretrained":
                sig = utils.convert_to_xxhz(dir + audio, 44100)

                print("{} -- {} :: {} -- {}".format(idx, dir, audio, len(sig)))

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

                signals[audio] = sigs

                if idx == 0:
                    VQT1 = IR.generate_XQT(signals[audio][:, 0], 16000, "vqt")
                    VQT2 = IR.generate_XQT(signals[audio][:, 1], 16000, "vqt")

                    VQT = np.zeros((2, VQT1.shape[0], VQT1.shape[1]), dtype=float)
                    VQT[0, :, :] = VQT1[:, :]
                    VQT[1, :, :] = VQT2[:, :]

                    vqts = {}
                    print("VQT shape: ({} * {})".format(VQT.shape[1], VQT.shape[2]))
                    vqts[audio] = torch.from_numpy(VQT).float()
                    d_pulse = torch.zeros(VQT.shape[2])
                    b_pulse = torch.zeros(VQT.shape[2])

                else:
                    VQT1 = IR.generate_XQT(signals[audio][:, 0], 16000, "vqt")
                    VQT2 = IR.generate_XQT(signals[audio][:, 1], 16000, "vqt")

                    VQT = np.zeros((2, VQT1.shape[0], VQT1.shape[1]))
                    VQT[0, :, :] = VQT1[:, :]
                    VQT[1, :, :] = VQT2[:, :]

                    print("VQT shape: ({} * {})".format(VQT.shape[1], VQT.shape[2]))
                    vqts[audio] = torch.from_numpy(VQT).float()
                    d_pulse = torch.zeros(VQT.shape[2])
                    b_pulse = torch.zeros(VQT.shape[2])

            else:
                sig = utils.preprocess(dir + audio)

                print("{} -- {} :: {} -- {}".format(idx, dir, audio, len(sig)))

                signals[audio] = sig[:]

                if idx == 0:
                    VQT = IR.generate_XQT(signals[audio], 16000, "vqt")
                    vqts = {}
                    print("VQT shape: ({} * {})".format(VQT.shape[0], VQT.shape[1]))
                    vqts[audio] = torch.from_numpy(VQT).float()
                    d_pulse = torch.zeros(VQT.shape[1])
                    b_pulse = torch.zeros(VQT.shape[1])

                else:
                    VQT = IR.generate_XQT(signals[audio], 16000, "vqt")
                    print("VQT shape: ({} * {})".format(VQT.shape[0], VQT.shape[1]))
                    vqts[audio] = torch.from_numpy(VQT).float()
                    d_pulse = torch.zeros(VQT.shape[1])
                    b_pulse = torch.zeros(VQT.shape[1])

            beats = audio.replace(".wav", ".beats")

            with open("BallroomData/BallroomAnnotations-master/" + beats, "r") as fp:
                times = fp.readlines()

            beat_tmz = []
            down_tmz = []

            for t in times:
                temp = t.replace("\n", "")

                down = int(temp[-1:])
                beat = float(temp[:-2])

                beat_tmz.append(beat)
                temp = round(62.5 * beat)

                if temp >= len(b_pulse) - 2:
                    temp = len(b_pulse) - 2

                if temp == 0:
                    temp = 1

                if down == 1:
                    down_tmz.append(beat)
                    d_pulse[temp] = 1
                    d_pulse[temp - 1] = 0.5
                    d_pulse[temp + 1] = 0.5

                b_pulse[temp] = 1
                b_pulse[temp - 1] = 0.5
                b_pulse[temp + 1] = 0.5

            real_beat_times[audio] = beat_tmz
            real_down_times[audio] = down_tmz

            down_pulse[audio] = d_pulse
            beat_pulse[audio] = b_pulse

            idx += 1

        with open("data/Ballroom/wavs.pkl", "wb") as handle:
            pickle.dump(wavs, handle, pickle.HIGHEST_PROTOCOL)

        if ballroom_status == "pretrained":
            with open("data/Ballroom/signals_spleeted.pkl", "wb") as handle:
                pickle.dump(signals, handle, pickle.HIGHEST_PROTOCOL)

        else:
            with open("data/Ballroom/signals_original.pkl", "wb") as handle:
                pickle.dump(signals, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/Ballroom/beat_pulses.pkl", "wb") as handle:
            pickle.dump(beat_pulse, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/Ballroom/down_pulses.pkl", "wb") as handle:
            pickle.dump(down_pulse, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/Ballroom/real_beat_times.pkl", "wb") as handle:
            pickle.dump(real_beat_times, handle, pickle.HIGHEST_PROTOCOL)

        with open("data/Ballroom/real_down_times.pkl", "wb") as handle:
            pickle.dump(real_down_times, handle, pickle.HIGHEST_PROTOCOL)

        if ballroom_status == "pretrained":
            with open("data/Ballroom/vqts_spleeted.pkl", "wb") as handle:
                pickle.dump(vqts, handle, pickle.HIGHEST_PROTOCOL)

        else:
            with open("data/Ballroom/vqts_original.pkl", "wb") as handle:
                pickle.dump(vqts, handle, pickle.HIGHEST_PROTOCOL)

    else:
        with open("data/Ballroom/wavs.pkl", "rb") as handle:
            wavs = pickle.load(handle)

        if ballroom_status == "pretrained":
            with open("data/Ballroom/signals_spleeted.pkl", "rb") as handle:
                signals = pickle.load(handle)

        else:
            with open("data/Ballroom/signals_original.pkl", "rb") as handle:
                signals = pickle.load(handle)

        with open("data/Ballroom/beat_pulses.pkl", "rb") as handle:
            beat_pulse = pickle.load(handle)

        with open("data/Ballroom/down_pulses.pkl", "rb") as handle:
            down_pulse = pickle.load(handle)

        with open("data/Ballroom/real_beat_times.pkl", "rb") as handle:
            real_beat_times = pickle.load(handle)

        with open("data/Ballroom/real_down_times.pkl", "rb") as handle:
            real_down_times = pickle.load(handle)

        if ballroom_status == "pretrained":
            with open("data/Ballroom/vqts_spleeted.pkl", "rb") as handle:
                vqts = pickle.load(handle)

        else:
            with open("data/Ballroom/vqts_original.pkl", "rb") as handle:
                vqts = pickle.load(handle)

    _exp = ymldict.get("ballroom_exp")

    if ballroom_status == "old-school":
        DP.dp_ellis(wavs, signals, real_beat_times)

    elif _exp == "beat":
        _ = BD.train_model(wavs, vqts, beat_pulse, real_beat_times, "ballroom", ymldict)

    elif _exp == "perc":
        _ = DE.train_model(wavs, vqts, beat_pulse, real_beat_times, "ballroom", ymldict)

    else:
        print("YAML file for Ballroom data set has a bug in experiment definition!")
