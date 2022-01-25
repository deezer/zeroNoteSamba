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
import beat_down            as BD
import data_exp             as DE
import old_school           as DP

import processing.source_separation as source_separation


if __name__ == '__main__':
    save = True

    # Load YAML file configuations 
    stream  = open("configuration/config.yaml", 'r')
    ymldict = yaml.safe_load(stream)

    smc_status = ymldict.get("smc_status")

    print("Loading audio and pulses...")

    if (save == False):
        # Load the separation model:
        model     = ymldict.get("spl_mod")
        m         = 'spleeter:{}'.format(model)
        separator = Separator(m)

        al = os.listdir("SMC_MIREX/SMC_MIREX_Audio/")
        bl = os.listdir("SMC_MIREX/SMC_MIREX_Annotations_05_08_2014/")

        wavs = []
        beat_list = []

        for el in al:
            if (el[0] == '.'):
                continue
            else:
                wavs.append(el)

        for el in bl:
            if ("beats" in el or el[0] == '.'):
                continue
            else:
                beat_list.append(el)

        wavs.sort(reverse=True)
        beat_list.sort(reverse=True)

        num_songs = len(wavs)

        signals    = {}
        beat_pulse = {}

        real_beat_times = {}

        idx = 0

        with open('SMC/smc_wavs.pkl', 'wb') as handle:
            pickle.dump(wavs, handle, pickle.HIGHEST_PROTOCOL)

        for audio in wavs:
            beats = beat_list.pop(0)

            print("{} :: {}".format(audio, beats))

            if (smc_status == 'pretrained'):    
                temp_sig = utils.convert_to_xxhz("SMC_MIREX/SMC_MIREX_Audio/" + audio, 44100)

                temp_stems = source_separation.wv_run_spleeter(temp_sig, 44100, separator, model)

                anchor = None
                for name, sig in temp_stems.items():
                    if (name == 'drums'):     
                        possignal       = np.zeros(sig.shape)   
                        possignal[:, :] = sig[:, :]       
                        
                    else:
                        if (anchor is None):
                            anchor       = np.zeros(sig.shape)
                            anchor[:, :] = sig[:, :]

                        else:
                            anchor[:, :] += sig[:, :]

                anchor    = utils.convert_to_mono(anchor)
                anchor    = audio_lib.resample(anchor, 44100, 16000)
                possignal = utils.convert_to_mono(possignal)
                possignal = audio_lib.resample(possignal, 44100, 16000)

                sigs = np.zeros((anchor.shape[0], 2))
                sigs[:, 0] = anchor[:]
                sigs[:, 1] = possignal[:]

                signals[audio] = sigs

                if (idx == 0):
                    VQT1 = IR.generate_XQT(signals[audio][:, 0], 16000, "vqt")
                    VQT2 = IR.generate_XQT(signals[audio][:, 1], 16000, "vqt")

                    VQT  = np.zeros((2, VQT1.shape[0], VQT1.shape[1]), dtype=float)
                    VQT[0, :, :] = VQT1[:, :]
                    VQT[1, :, :] = VQT2[:, :]

                    vqts = {}
                    print("VQT shape: ({} * {})".format(VQT.shape[1], VQT.shape[2]))
                    vqts[audio] = torch.from_numpy(VQT).float()
                    b_pulse  = torch.zeros(VQT.shape[2])

                else:
                    VQT1 = IR.generate_XQT(signals[audio][:, 0], 16000, "vqt")
                    VQT2 = IR.generate_XQT(signals[audio][:, 1], 16000, "vqt")
                
                    VQT  = np.zeros((2, VQT1.shape[0], VQT1.shape[1]))
                    VQT[0, :, :] = VQT1[:, :]
                    VQT[1, :, :] = VQT2[:, :]
                    
                    print("VQT shape: ({} * {})".format(VQT.shape[1], VQT.shape[2]))
                    vqts[audio] = torch.from_numpy(VQT).float()
                    b_pulse  = torch.zeros(VQT.shape[2])

            else:
                sig = utils.preprocess("SMC_MIREX/SMC_MIREX_Audio/" + audio)
                signals[audio] = sig[:]

                if (idx == 0):
                    VQT = IR.generate_XQT(signals[audio], 16000, "vqt")
                    vqts = {}
                    print("VQT shape: ({} * {})".format(VQT.shape[0], VQT.shape[1]))
                    vqts[audio] = torch.from_numpy(VQT).float()
                    b_pulse  = torch.zeros(VQT.shape[1])

                else:
                    VQT = IR.generate_XQT(signals[audio], 16000, "vqt")
                    print("VQT shape: ({} * {})".format(VQT.shape[0], VQT.shape[1]))
                    vqts[audio] = torch.from_numpy(VQT).float()
                    b_pulse  = torch.zeros(VQT.shape[1])

            with open("SMC_MIREX/SMC_MIREX_Annotations_05_08_2014/" + beats, 'r') as fp:
                times = fp.readlines()

            tmz = []

            for t in times:
                temp = float(t.replace("\n", ""))
                tmz.append(temp)
                temp = round(62.5 * temp)

                if (temp >= 2499):
                    temp = 2499

                if (temp == 0):
                    temp = 1

                b_pulse[temp]   = 1
                b_pulse[temp-1] = 0.5
                b_pulse[temp+1] = 0.5

            real_beat_times[audio] = tmz

            beat_pulse[audio] = b_pulse

            idx += 1

        if (smc_status == 'pretrained'):
            with open('SMC/smc_signals_spleeted.pkl', 'wb') as handle:
                pickle.dump(signals, handle, pickle.HIGHEST_PROTOCOL)
        
        else:
            with open('SMC/smc_signals_original.pkl', 'wb') as handle:
                pickle.dump(signals, handle, pickle.HIGHEST_PROTOCOL)

        with open('SMC/smc_pulses.pkl', 'wb') as handle:
            pickle.dump(beat_pulse, handle, pickle.HIGHEST_PROTOCOL)

        with open('SMC/smc_real_times.pkl', 'wb') as handle:
            pickle.dump(real_beat_times, handle, pickle.HIGHEST_PROTOCOL)

        if (smc_status == 'pretrained'):
            with open('SMC/smc_vqts_spleeted.pkl', 'wb') as handle:
                pickle.dump(vqts, handle, pickle.HIGHEST_PROTOCOL)

        else:
            with open('SMC/smc_vqts_original.pkl', 'wb') as handle:
                pickle.dump(vqts, handle, pickle.HIGHEST_PROTOCOL)

    else:
        with open('SMC/smc_wavs.pkl', 'rb') as handle:
            wavs = pickle.load(handle)

        if (smc_status == 'pretrained'):
            with open('SMC/smc_signals_spleeted.pkl', 'rb') as handle:
                signals = pickle.load(handle)

        else:
            with open('SMC/smc_signals_original.pkl', 'rb') as handle:
                signals = pickle.load(handle)

        with open('SMC/smc_pulses.pkl', 'rb') as handle:
            beat_pulse = pickle.load(handle)

        with open('SMC/smc_real_times.pkl', 'rb') as handle:
            real_beat_times = pickle.load(handle)

        if (smc_status == 'pretrained'):
            with open('SMC/smc_vqts_spleeted.pkl', 'rb') as handle:
                vqts = pickle.load(handle)
        
        else:
            with open('SMC/smc_vqts_original.pkl', 'rb') as handle:
                vqts = pickle.load(handle)  

    _exp = ymldict.get("smc_exp")

    if (smc_status == 'old-school'):
        DP.dp_ellis(wavs, signals, real_beat_times)
    
    elif (_exp == 'beat'):
        _ = BD.train_model(wavs, vqts, beat_pulse, real_beat_times, "smc", ymldict)

    elif (_exp == 'perc'):
        _ = DE.train_model(wavs, vqts, beat_pulse, real_beat_times, "smc", ymldict)

    else:
        print("YAML file for SMC data set has a bug in experiment definition!")