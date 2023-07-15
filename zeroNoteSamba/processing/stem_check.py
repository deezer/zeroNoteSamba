from typing import Any, Dict, Tuple

import librosa as audio_lib
import numpy as np
import numpy.typing as npt


def compute_rms(signal: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, float]:
    """
    Function for combining a signal's Root Mean Square (RMS) value.
    -- signal: input waveform
    """
    rms = audio_lib.feature.rms(y=signal, frame_length=2048, hop_length=512)

    # Compute mean and standard deviations of rms for stems
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)

    return rms, mean_rms, std_rms


def check_CL_clips(
    anchor: npt.NDArray[np.float32], positive: npt.NDArray[np.float32], lower_p: float, upper_p: float
) -> bool:
    """
    Function for thresholding anchor vs positive. Goal is to make sure drum clip has enough energy.
    -- anchor: selected stem combination
    -- positive: other stem combination
    -- lower_p: lower RMS percentage threshold
    -- upper_p: upper RMS percentage threshold
    """
    # Compute both stem and rest of signal RMS
    stem_rms, _, _ = compute_rms(anchor.T)
    ros_rms, _, _ = compute_rms(positive.T)

    # Thresholding
    rms_check1 = stem_rms[:] > ros_rms[:] / 2
    rms_check2 = stem_rms[:] < ros_rms[:] * 4
    rms_check1 = rms_check1.astype(int)[0]
    rms_check2 = rms_check2.astype(int)[0]
    rms_check = rms_check1[:] * rms_check2[:]
    rms_sum = np.sum(rms_check)
    rms_perc = rms_sum / len(rms_check)

    # print("RMS Clip Pow% : {}".format(rms_perc))

    if lower_p < rms_perc <= upper_p:
        return True

    else:
        return False


def check_drum_stem(stems: Dict[str, npt.NDArray[np.float32]], ymldict: Dict[str, Any]) -> bool:
    """
    Function for thresholding drums. Goal is to make sure drum clip has enough energy.
    -- stems: dictionary with stems and their names
    -- ymldict: dictionary with yaml parameters
    """
    # Load desired variables
    lower_p = ymldict.get("lower_p")
    upper_p = ymldict.get("upper_p")

    check_drum = False
    rest_of_sig = None

    # Compute RMS value for drum stem.
    for name, sig in stems.items():
        if name == "drums":
            check_drum = True
            drum_rms, _, _ = compute_rms(sig.T)
        else:
            if rest_of_sig is None:
                rest_of_sig = np.zeros((len(sig), 2), dtype=np.float32)
                rest_of_sig[:, :] = sig[:, :]

            else:
                rest_of_sig[:, :] += sig[:, :]

    if check_drum is False:
        raise Exception("Stems do not contain any drum tracks!")

    if rest_of_sig is not None:
        ros_rms, _, _ = compute_rms(rest_of_sig.T)

        # rms_check = np.where(rms[0] > threshold, 1, 0)
        rms_check1 = drum_rms[:] > ros_rms[:] / 2
        rms_check2 = drum_rms[:] < ros_rms[:] * 4
        rms_check1 = rms_check1.astype(int)[0]
        rms_check2 = rms_check2.astype(int)[0]
        rms_check = rms_check1[:] * rms_check2[:]
        rms_sum = np.sum(rms_check)
        rms_perc = rms_sum / len(rms_check)

        print("   RMS Pow% : {}".format(rms_perc))

        if lower_p < rms_perc < upper_p:
            return True

        else:
            return False

    else:
        raise Exception("Rest-of-signal is still None.")
