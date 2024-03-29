import pathlib
import random
from typing import Any, Dict, List, SupportsFloat, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
from tqdm import tqdm
from typing_extensions import Buffer, SupportsIndex

from zeroNoteSamba.epochs import train_epoch, val_epoch
from zeroNoteSamba.loader import load_models
from zeroNoteSamba.models.models import DS_CNN, Down_CNN
from zeroNoteSamba.processing.evaluate import beat_tracking as eval


def train_model(
    wavs: List[str],
    inputs: Dict[str, Any],
    masks: Dict[str, Any],
    real_times: Dict[str, Any],
    data_set: str,
    ymldict: Dict[str, Union[SupportsFloat, SupportsIndex, str, Buffer]],
) -> torch.nn.Module:
    """
    Function for training model on Ballroom data set.
    -- wavs: list of files
    -- inputs: spectrograms of audio to feed to NN
    -- masks: pulse vectors
    -- real_times: list with real times in seconds
    -- data_set: data set we are running our experiment on
    -- ymldict: YAML parameters
    """
    # Load the experiment stuff:
    _status = str(ymldict.get("{}_status".format(data_set)))
    _pre = str(ymldict.get("{}_pre".format(data_set)))
    _exp = ymldict.get("{}_exp".format(data_set))
    _lr = float(ymldict.get("{}_lr".format(data_set), ""))
    _eval = ymldict.get("{}_eval".format(data_set))

    threshold = False
    librosa = False

    if _eval == "threshold":
        threshold = True

    elif _eval == "librosa":
        librosa = True

    random.shuffle(wavs)

    cv_len = len(wavs) / 8

    split1 = wavs[0 : round(cv_len)]
    split2 = wavs[round(cv_len) : round(cv_len * 2)]
    split3 = wavs[round(cv_len * 2) : round(cv_len * 3)]
    split4 = wavs[round(cv_len * 3) : round(cv_len * 4)]
    split5 = wavs[round(cv_len * 4) : round(cv_len * 5)]
    split6 = wavs[round(cv_len * 5) : round(cv_len * 6)]
    split7 = wavs[round(cv_len * 6) : round(cv_len * 7)]
    split8 = wavs[round(cv_len * 7) :]

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

        if (_status == "clmr" or _status == "pretrained" or _status == "vanilla") and (
            _pre == "finetune" or _pre == "frozen"
        ):
            val_counter = 0

            train_loss = []
            val_loss = []
            train_f1 = []
            val_f1 = []
            train_indices: List[str] = []

            for ii in range(8):
                if ii != jj:
                    train_indices = train_indices + splits[ii]

            test_indices = splits[jj]

            random.shuffle(train_indices)

            val_indices = train_indices[0 : round(cv_len)]
            train_indices = train_indices[round(cv_len) :]

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
                    real_times,
                    inputs,
                    masks,
                    threshold,
                    librosa,
                )

                print("\nMean training loss     is {:.3f}.".format(full_train_loss))
                print("Mean training F1-score is {:.3f}.".format(train_f_measure))

                full_val_loss, val_f_measure, _, _, _, _, _ = val_epoch(
                    model,
                    criterion,
                    _status,
                    val_indices,
                    real_times,
                    inputs,
                    masks,
                    threshold,
                    librosa,
                )

                print("\nMean validation loss     is {:.3f}.".format(full_val_loss))
                print("Mean validation F1-score is {:.3f}.".format(val_f_measure))

                if val_f_measure > best_f1:
                    mod_fp = "models/saved/{}_{}_{}.pth".format(data_set, _exp, _status)
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

            mod_fp = "models/saved/{}_{}_{}.pth".format(data_set, _exp, _status)

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
                test_indices,
                real_times,
                inputs,
                masks,
                threshold,
                librosa,
            )

            print("\n-- Test Set --")
            print("\nMean test loss     is {:.3f}.".format(full_test_loss))
            print("Mean test F1-score is {:.3f}.".format(test_f_measure))
            print("Mean test CMLC     is {:.3f}.".format(test_cmlc))
            print("Mean test CMLT     is {:.3f}.".format(test_cmlt))
            print("Mean test AMLC     is {:.3f}.".format(test_amlc))
            print("Mean test AMLT     is {:.3f}.".format(test_amlt))
            print("Mean test InfoGain is {:.3f}.".format(test_info_gain))

            f1.append(test_f_measure)
            cmlc.append(test_cmlc)
            cmlt.append(test_cmlt)
            amlc.append(test_amlc)
            amlt.append(test_amlt)
            ig.append(test_info_gain)

            if jj == 0:
                pathlib.Path("figures/{}/{}".format(data_set, _exp)).mkdir(parents=True, exist_ok=True)

            plt.figure()
            plt.plot(train_loss)
            plt.plot(val_loss)
            plt.legend(["Train", "Validation"])
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig(
                "figures/{}/{}/{}_loss_{}.pdf".format(data_set, _exp, _status, jj),
                dpi=300,
                format="pdf",
            )

            plt.figure()
            plt.plot(train_f1)
            plt.plot(val_f1)
            plt.legend(["Train", "Validation"])
            plt.ylim([0, 1])
            plt.xlabel("Epochs")
            plt.ylabel("F1-score")
            plt.savefig(
                "figures/{}/{}/{}_f1_{}.pdf".format(data_set, _exp, _status, jj),
                dpi=300,
                format="pdf",
            )

        elif (_status == "pretrained" or _status == "clmr" or _status == "vanilla") and _pre == "validation":
            model.eval()

            (
                l_full_test_loss,
                l_test_f_measure,
                l_test_cmlc,
                l_test_cmlt,
                l_test_amlc,
                l_test_amlt,
                l_test_info_gain,
            ) = ([], [], [], [], [], [], [])

            for _, wav in enumerate(tqdm(wavs)):
                with torch.no_grad():
                    times = real_times[wav]

                    if _status == "pretrained":
                        vqt = inputs[wav]
                        vqt1 = torch.reshape(vqt[0, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()
                        vqt2 = torch.reshape(vqt[1, :, :], (1, 1, vqt.shape[1], vqt.shape[2])).cuda()

                        msk = masks[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        output = model(vqt1, vqt2)

                        loss = criterion(output, msk)

                    else:
                        vqt = inputs[wav]
                        vqt = torch.reshape(vqt[:, :], (1, 1, vqt.shape[0], vqt.shape[1])).cuda()

                        msk = masks[wav]
                        msk = torch.reshape(msk, (1, msk.shape[0])).cuda()

                        # Apply model to full batch
                        output = model(vqt)

                        loss = criterion(output, msk)

                    l_full_test_loss.append(loss.item())

                    cpu_output = output.squeeze(0).cpu().detach().numpy()

                    res = eval(cpu_output, times, threshold=threshold, librosa=librosa)
                    l_test_f_measure.append(res[0])
                    l_test_cmlc.append(res[1])
                    l_test_cmlt.append(res[2])
                    l_test_amlc.append(res[3])
                    l_test_amlt.append(res[4])
                    l_test_info_gain.append(res[5])

            print("\n-- Full Set --")
            print("Mean loss     is {:.3f} +- {:.3f}.".format(np.mean(l_full_test_loss), np.std(l_full_test_loss)))
            print("Mean F1-score is {:.3f} +- {:.3f}.".format(np.mean(l_test_f_measure), np.std(l_test_f_measure)))
            print("Mean CMLC     is {:.3f} +- {:.3f}.".format(np.mean(l_test_cmlc), np.std(l_test_cmlc)))
            print("Mean CMLT     is {:.3f} +- {:.3f}.".format(np.mean(l_test_cmlt), np.std(l_test_cmlt)))
            print("Mean AMLC     is {:.3f} +- {:.3f}.".format(np.mean(l_test_amlc), np.std(l_test_amlc)))
            print("Mean AMLT     is {:.3f} +- {:.3f}.".format(np.mean(l_test_amlt), np.std(l_test_amlt)))
            print("Mean InfoGain is {:.3f} +- {:.3f}.".format(np.mean(l_test_info_gain), np.std(l_test_info_gain)))

            break

        else:
            raise ValueError("Problem with configuration file experiment arguments: {} and {}.".format(_status, _pre))

    if f1 != []:
        f1_arr = np.asarray(f1)
        cmlc_arr = np.asarray(cmlc)
        cmlt_arr = np.asarray(cmlt)
        amlc_arr = np.asarray(amlc)
        amlt_arr = np.asarray(amlt)
        ig_arr = np.asarray(ig)

        print("\n8-fold CV results:")
        print("\nF1-score is {:.3f} +- {:.3f}.".format(np.mean(f1_arr), np.std(f1_arr)))
        print("CMLC     is {:.3f} +- {:.3f}.".format(np.mean(cmlc_arr), np.std(cmlc_arr)))
        print("CMLT     is {:.3f} +- {:.3f}.".format(np.mean(cmlt_arr), np.std(cmlt_arr)))
        print("AMLC     is {:.3f} +- {:.3f}.".format(np.mean(amlc_arr), np.std(amlc_arr)))
        print("AMLT     is {:.3f} +- {:.3f}.".format(np.mean(amlt_arr), np.std(amlt_arr)))
        print("InfoGain is {:.3f} +- {:.3f}.".format(np.mean(ig_arr), np.std(ig_arr)))

    return model
