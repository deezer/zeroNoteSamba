import pickle
import random
from typing import Any, Dict, List, SupportsFloat, Union

import numpy as np
import torch
import yaml
from typing_extensions import Buffer, SupportsIndex

from zeroNoteSamba.epochs import train_epoch, val_epoch
from zeroNoteSamba.loader import load_models
from zeroNoteSamba.models.models import DS_CNN, Down_CNN


def train_model(
    train_wavs: List[str],
    train_vqts: Dict[str, Any],
    train_masks: Dict[str, Any],
    train_real_times: Dict[str, Any],
    test_wavs: List[str],
    test_vqts: Dict[str, Any],
    test_masks: Dict[str, Any],
    test_real_times: Dict[str, Any],
    ymldict: Dict[str, Union[SupportsFloat, SupportsIndex, str, Buffer]],
) -> torch.nn.Module:
    """
    Function for training model on Hainsworth data set.
    -- train_wavs: list of training wav files (dictionary keys)
    -- train_vqts: training spectrograms of audio
    -- train_masks: training beat tracking pulse vectors
    -- train_real_times: training list with real beat tracking times in seconds
    -- test_wavs: list of test wav files (dictionary keys)
    -- test_vqts: test spectrograms of audio
    -- test_masks: test tracking pulse vectors
    -- test_real_times: test list with real beat tracking times in seconds
    -- ymldict: YAML parameters
    """
    # Load the Hainsworth stuff:
    _status = str(ymldict.get("cross_status"))
    _pre = str(ymldict.get("cross_pre"))
    _train_set = ymldict.get("cross_train_set")
    _lr = float(ymldict.get("cross_lr", ""))
    _eval = ymldict.get("cross_eval")

    threshold = False
    librosa = False

    if _eval == "threshold":
        threshold = True

    elif _eval == "librosa":
        librosa = True

    random.shuffle(train_wavs)

    cv_len = len(train_wavs) / 8

    split1 = train_wavs[0 : round(cv_len)]
    split2 = train_wavs[round(cv_len) : round(cv_len * 2)]
    split3 = train_wavs[round(cv_len * 2) : round(cv_len * 3)]
    split4 = train_wavs[round(cv_len * 3) : round(cv_len * 4)]
    split5 = train_wavs[round(cv_len * 4) : round(cv_len * 5)]
    split6 = train_wavs[round(cv_len * 5) : round(cv_len * 6)]
    split7 = train_wavs[round(cv_len * 6) : round(cv_len * 7)]
    split8 = train_wavs[round(cv_len * 7) :]

    splits = [split1, split2, split3, split4, split5, split6, split7, split8]

    f1 = []
    cmlc = []
    cmlt = []
    amlc = []
    amlt = []
    ig = []

    for jj in range(8):
        # Load everything
        criterion, optimizer, model = load_models(_status, _pre, _lr)

        val_counter = 0

        train_loss = []
        val_loss = []
        train_f1 = []
        val_f1 = []
        train_indices: List[str] = []

        for ii in range(8):
            if ii != jj:
                train_indices = train_indices + splits[ii]

        val_indices = splits[jj]

        random.shuffle(train_indices)

        best_f1 = 0.0

        # Train model
        for epoch in range(500):
            print("\n-- Epoch {} --".format(epoch))

            (model, optimizer, full_train_loss, train_f_measure, _, _, _, _, _,) = train_epoch(
                model,
                criterion,
                optimizer,
                _status,
                train_indices,
                train_real_times,
                train_vqts,
                train_masks,
                threshold,
                librosa,
            )

            print("\nMean training loss is {:.3f}.".format(full_train_loss))
            print("Mean F1-score is {:.3f}.".format(train_f_measure))

            full_val_loss, val_f_measure, _, _, _, _, _ = val_epoch(
                model,
                criterion,
                _status,
                val_indices,
                train_real_times,
                train_vqts,
                train_masks,
                threshold,
                librosa,
            )

            print("\nMean validation loss     is {:.3f}.".format(full_val_loss))
            print("Mean validation F1-score is {:.3f}.".format(val_f_measure))

            if val_f_measure > best_f1:
                mod_fp = "models/saved/cross_{}_{}.pth".format(_train_set, _status)
                best_f1 = val_f_measure
                torch.save(model.state_dict(), mod_fp)
                print("Saved model to " + mod_fp)
                val_counter = 0

            else:
                val_counter += 1

            train_loss.append(full_train_loss)
            val_loss.append(full_val_loss)
            train_f1.append(train_f_measure)
            val_f1.append(val_f_measure)

            if val_counter >= 20:
                break

        mod_fp = "models/saved/cross_{}_{}.pth".format(_train_set, _status)

        test_mod: torch.nn.Module

        if _status == "pretrained":
            test_mod = Down_CNN().cuda()
        else:
            test_mod = DS_CNN().cuda()

        state_dict = torch.load(mod_fp)
        test_mod.load_state_dict(state_dict)

        (full_test_loss, test_f_measure, test_cmlc, test_cmlt, test_amlc, test_amlt, test_info_gain,) = val_epoch(
            test_mod,
            criterion,
            _status,
            test_wavs,
            test_real_times,
            test_vqts,
            test_masks,
            threshold,
            librosa,
        )

        print("\n-- Test Set --")
        print("\nMean test loss     is {:.3f}.".format(full_test_loss))
        print("Mean beat F1-score is {:.3f}.".format(test_f_measure))
        print("Mean beat CMLC     is {:.3f}.".format(test_cmlc))
        print("Mean beat CMLT     is {:.3f}.".format(test_cmlt))
        print("Mean beat AMLC     is {:.3f}.".format(test_amlc))
        print("Mean beat AMLT     is {:.3f}.".format(test_amlt))
        print("Mean beat InfoGain is {:.3f}.".format(test_info_gain))

        f1.append(test_f_measure)
        cmlc.append(test_cmlc)
        cmlt.append(test_cmlt)
        amlc.append(test_amlc)
        amlt.append(test_amlt)
        ig.append(test_info_gain)

    f1_arr = np.asarray(f1)
    cmlc_arr = np.asarray(cmlc)
    cmlt_arr = np.asarray(cmlt)
    amlc_arr = np.asarray(amlc)
    amlt_arr = np.asarray(amlt)
    ig_arr = np.asarray(ig)

    print("\n-- 8-fold CV results --")
    print("\nBeat F1-score is {:.3f} +- {:.3f}.".format(np.mean(f1_arr), np.std(f1_arr)))
    print("Beat CMLC     is {:.3f} +- {:.3f}.".format(np.mean(cmlc_arr), np.std(cmlc_arr)))
    print("Beat CMLT     is {:.3f} +- {:.3f}.".format(np.mean(cmlt_arr), np.std(cmlt_arr)))
    print("Beat AMLC     is {:.3f} +- {:.3f}.".format(np.mean(amlc_arr), np.std(amlc_arr)))
    print("Beat AMLT     is {:.3f} +- {:.3f}.".format(np.mean(amlt_arr), np.std(amlt_arr)))
    print("Beat InfoGain is {:.3f} +- {:.3f}.".format(np.mean(ig_arr), np.std(ig_arr)))

    return model


if __name__ == "__main__":
    # Load YAML file configuations
    stream = open("zeroNoteSamba/configuration/config.yaml", "r")
    ymldict = yaml.safe_load(stream)

    _status = ymldict.get("cross_status")
    _train_set = ymldict.get("cross_train_set")

    print("Loading audio and pulses...")

    with open("data/GTZAN/wavs.pkl", "rb") as handle:
        test_wavs = pickle.load(handle)

    with open("data/GTZAN/beat_pulses.pkl", "rb") as handle:
        test_masks = pickle.load(handle)

    with open("data/GTZAN/real_beat_times.pkl", "rb") as handle:
        test_real_times = pickle.load(handle)

    if _status == "pretrained":
        with open("data/GTZAN/vqts_spleeted.pkl", "rb") as handle:
            test_vqts = pickle.load(handle)

    else:
        with open("data/GTZAN/vqts_original.pkl", "rb") as handle:
            test_vqts = pickle.load(handle)

    if _train_set == "hainsworth":
        with open("data/Hainsworth/wavs.pkl", "rb") as handle:
            train_wavs = pickle.load(handle)

        with open("data/Hainsworth/beat_pulses.pkl", "rb") as handle:
            train_masks = pickle.load(handle)

        with open("data/Hainsworth/real_beat_times.pkl", "rb") as handle:
            train_real_times = pickle.load(handle)

        if _status == "pretrained":
            with open("data/Hainsworth/vqts_spleeted.pkl", "rb") as handle:
                train_vqts = pickle.load(handle)

        else:
            with open("data/Hainsworth/vqts_original.pkl", "rb") as handle:
                train_vqts = pickle.load(handle)

    elif _train_set == "smc":
        with open("data/SMC/wavs.pkl", "rb") as handle:
            train_wavs = pickle.load(handle)

        with open("data/SMC/pulses.pkl", "rb") as handle:
            train_masks = pickle.load(handle)

        with open("data/SMC/real_times.pkl", "rb") as handle:
            train_real_times = pickle.load(handle)

        if _status == "pretrained":
            with open("data/SMC/vqts_spleeted.pkl", "rb") as handle:
                train_vqts = pickle.load(handle)

        else:
            with open("data/SMC/vqts_original.pkl", "rb") as handle:
                train_vqts = pickle.load(handle)

    elif _train_set == "ballroom":
        with open("data/Ballroom/wavs.pkl", "rb") as handle:
            train_wavs = pickle.load(handle)

        with open("data/Ballroom/beat_pulses.pkl", "rb") as handle:
            train_masks = pickle.load(handle)

        with open("data/Ballroom/real_beat_times.pkl", "rb") as handle:
            train_real_times = pickle.load(handle)

        if _status == "pretrained":
            with open("data/Ballroom/vqts_spleeted.pkl", "rb") as handle:
                train_vqts = pickle.load(handle)

        else:
            with open("data/Ballroom/vqts_original.pkl", "rb") as handle:
                train_vqts = pickle.load(handle)

    else:
        print("\nYAML file for cross data set has a bug in experiment definition!")

    _ = train_model(
        train_wavs,
        train_vqts,
        train_masks,
        train_real_times,
        test_wavs,
        test_vqts,
        test_masks,
        test_real_times,
        ymldict,
    )
