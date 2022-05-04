import torch
import random
import numpy as np

from tqdm import trange
from loader import load_models
from epochs import train_epoch, val_epoch
from models.models import DS_CNN, Down_CNN


def train_model(wavs, vqts, beat_pulse, real_beat_times, data_set, ymldict):
    """
    Function for training model on GTZAN data set.
    -- wavs : list of wav files (dictionary keys)
    -- vqts : spectrograms of audio
    -- beat_pulse : beat tracking pulse vectors
    -- real_beat_times : list with real beat tracking times in seconds
    -- data_set : data set we are running our experiment on
    -- ymldict : YAML parameters
    """
    # Load the experiment stuff:
    _status = ymldict.get("{}_status".format(data_set))
    _pre = ymldict.get("{}_pre".format(data_set))
    _exp = ymldict.get("{}_exp".format(data_set))
    _lr = ymldict.get("{}_lr".format(data_set))
    _eval = ymldict.get("{}_eval".format(data_set))

    threshold = False
    librosa = False

    if _eval == "threshold":
        threshold = True

    elif _eval == "librosa":
        librosa = True

    random.Random(16).shuffle(wavs)

    cv_len = len(wavs) / 8

    split = wavs[0 : round(cv_len * 6)]
    val_indices = wavs[round(cv_len * 6) : round(cv_len * 7)]
    test_indices = wavs[round(cv_len * 7) :]

    train_lens = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]

    for train_len in train_lens:

        f1 = []
        cmlc = []
        cmlt = []
        amlc = []
        amlt = []
        ig = []

        print("\nTrain set size is {}.".format(train_len))

        for jj in range(2):
            # Load everything
            criterion, optimizer, model = load_models(_status, _pre, _lr)

            val_counter = 0

            train_loss = []
            val_loss = []
            train_f1 = []
            val_f1 = []

            random.Random(16).shuffle(split)

            train_indices = split[0:train_len]

            best_f1 = 0.0

            # Train model
            for _ in trange(1):
                (
                    model,
                    optimizer,
                    full_train_loss,
                    train_f_measure,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = train_epoch(
                    model,
                    criterion,
                    optimizer,
                    _status,
                    train_indices,
                    real_beat_times,
                    vqts,
                    beat_pulse,
                    threshold,
                    librosa
                )

                full_val_loss, val_f_measure, _, _, _, _, _ = val_epoch(
                    model,
                    criterion,
                    _status,
                    val_indices,
                    real_beat_times,
                    vqts,
                    beat_pulse,
                    threshold,
                    librosa
                )

                if val_f_measure > best_f1:
                    mod_fp = "models/saved/{}_{}_{}.pth".format(data_set, _exp, _status)
                    best_f1 = val_f_measure
                    torch.save(model.state_dict(), mod_fp)
                    val_counter = 0

                else:
                    val_counter += 1

                train_loss.append(full_train_loss)
                val_loss.append(full_val_loss)
                train_f1.append(train_f_measure)
                val_f1.append(val_f_measure)

                if val_counter >= 20:
                    break

            print("\nBest validation F1-score is {}.".format(best_f1))

            mod_fp = "models/saved/{}_{}_{}.pth".format(data_set, _exp, _status)

            if _status == "pretrained":
                test_mod = Down_CNN().cuda()
            else:
                test_mod = DS_CNN().cuda()

            state_dict = torch.load(mod_fp)
            test_mod.load_state_dict(state_dict)

            (
                full_test_loss,
                test_f_measure,
                test_cmlc,
                test_cmlt,
                test_amlc,
                test_amlt,
                test_info_gain,
            ) = val_epoch(
                test_mod,
                criterion,
                _status,
                test_indices,
                real_beat_times,
                vqts,
                beat_pulse,
                threshold,
                librosa
            )

            print("\n-- Test Set {} --".format(jj))
            print("\nMean test loss     is {:.3f}.".format(full_test_loss))
            print("Mean beat F1-score is {:.3f}.".format(test_f_measure))

            f1.append(test_f_measure)
            cmlc.append(test_cmlc)
            cmlt.append(test_cmlt)
            amlc.append(test_amlc)
            amlt.append(test_amlt)
            ig.append(test_info_gain)

        f1 = np.asarray(f1)
        cmlc = np.asarray(cmlc)
        cmlt = np.asarray(cmlt)
        amlc = np.asarray(amlc)
        amlt = np.asarray(amlt)
        ig = np.asarray(ig)

        print("\n-- 8-fold CV results --")
        print("\nBeat F1-score is {:.3f} +- {:.3f}.".format(np.mean(f1), np.std(f1)))
        print("Beat CMLC     is {:.3f} +- {:.3f}.".format(np.mean(cmlc), np.std(cmlc)))
        print("Beat CMLT     is {:.3f} +- {:.3f}.".format(np.mean(cmlt), np.std(cmlt)))
        print("Beat AMLC     is {:.3f} +- {:.3f}.".format(np.mean(amlc), np.std(amlc)))
        print("Beat AMLT     is {:.3f} +- {:.3f}.".format(np.mean(amlt), np.std(amlt)))
        print("Beat InfoGain is {:.3f} +- {:.3f}.".format(np.mean(ig), np.std(ig)))

    return model
