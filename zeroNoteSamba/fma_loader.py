import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import librosa as audio_lib
import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore
import yaml
from spleeter.separator import Separator
from tqdm import tqdm

import zeroNoteSamba.processing.input_rep as input_rep
import zeroNoteSamba.processing.source_separation as source_separation
import zeroNoteSamba.processing.stem_check as stem_check
import zeroNoteSamba.processing.utilities as utils


def gen_clmr(ymldict: Dict[str, Any]) -> None:
    """
    Generate CLMR examples.
    -- ymldict: dictionary with yaml parameters
    """
    # Load desired variables
    fma_dir = ymldict.get("pt_data_dir", "")

    # Iterate through FMA directory and create csv files with good files
    dir_list = os.listdir(fma_dir)

    no_explore = ["README.txt", "checksums"]

    pkl_len = 2048
    pkl_fp = "data/CLMR/clmr_pkl_"

    np_pkl = np.zeros((pkl_len, 2, 96, 313))

    idx = 0
    pkl_idx = 0

    # Iterate through directories
    for sel_dir in dir_list:
        if pkl_idx == 50:
            break

        if sel_dir in no_explore:
            continue

        print("Directory: {}".format(sel_dir))

        # Iterate through wav files
        wav_list = os.listdir(fma_dir + sel_dir + "/")

        for wav in tqdm(wav_list):
            if pkl_idx == 50:
                break

            f = fma_dir + sel_dir + "/" + wav

            try:
                # Create new 16000 Hz file
                yy = utils.convert_to_xxhz(f, 16000)
            except:
                continue

            if len(yy) < 5 * 16000 + 1:
                continue

            vqt = input_rep.generate_XQT(yy, 16000, "vqt")

            ran_idx1 = random.randint(0, vqt.shape[1] - 313)
            ran_idx2 = random.randint(0, vqt.shape[1] - 313)

            np_pkl[idx, 0, :, :] = vqt[:, ran_idx1 : ran_idx1 + 313]
            np_pkl[idx, 1, :, :] = vqt[:, ran_idx2 : ran_idx2 + 313]

            idx += 1

            if idx == pkl_len:
                with open(pkl_fp + str(pkl_idx), "wb") as handle:
                    pickle.dump(np_pkl, handle, pickle.HIGHEST_PROTOCOL)
                    print("-- Saved file {}. --".format(pkl_fp))

                idx = 0
                pkl_idx += 1

    return


def full_fma_stem_check(separator: Separator, ymldict: Dict[str, Any]) -> None:
    """
    Function that writes csv files with good stem files.
    -- separator: Spleeter separator
    -- ymldict: dictionary with yaml parameters
    """
    # Load desired variables
    fma_dir = ymldict.get("pt_data_dir", "")
    sr = ymldict.get("sample_rate", -1)

    # Iterate through FMA directory and create csv files with good files
    dir_list = os.listdir(fma_dir)

    no_explore = ["README.txt", "checksums"]

    up_to = "124238"

    up_to_bool = False

    # Iterate through directories
    for sel_dir in dir_list:
        if sel_dir in no_explore:
            print("{}: no exploration!".format(sel_dir))
            continue

        # Iterate through wav files
        wav_list = os.listdir(fma_dir + sel_dir + "/")

        for wav in wav_list:
            f = fma_dir + sel_dir + "/" + wav
            print("{}".format(f))

            if up_to in f:
                up_to_bool = True
                continue

            if up_to_bool == True:
                try:
                    _, stems, rms_bool = drum_load(f, separator, sr, ymldict)
                    print("   {}".format(rms_bool))
                except:
                    rms_bool = False
                    print("   {}".format("Failed"))

                if rms_bool == True:
                    Path("new_data/" + f.strip()[-10:-4]).mkdir(parents=True, exist_ok=True)

                    for key in stems:
                        stems[key] = utils.convert_to_mono(stems[key])
                        stems[key] = audio_lib.resample(
                            y=stems[key], orig_sr=sr, target_sr=16000, res_type="kaiser_fast"
                        )
                        sf.write(
                            "new_data/" + f.strip()[-10:-4] + "/" + key + ".wav",
                            stems[key],
                            16000,
                        )
                        print("   Saved " + "new_data/" + f.strip()[-10:-4] + "/" + key + ".wav")

    return


def drum_load(
    filename: str, separator: Separator, sr: int, ymldict: Dict[str, Any]
) -> Tuple[npt.NDArray[np.float32], Dict[Any, Any], bool]:
    """
    We run Spleeter on the input file.
    Return RMS status, full signal, and stems.
    -- filename: file to stem check
    -- separator: Spleeter separator
    -- sr: sample rate
    -- ymldict: dictionary with yaml parameters
    """
    spl_mod = ymldict.get("spl_mod", "")

    # Create new 16000 Hz file
    yy = utils.convert_to_xxhz(filename, sr)

    # Run Spleeter
    stems = source_separation.wv_run_spleeter(yy, sr, separator, spl_mod)

    # Check whether drum stem is HQ
    rms_bool = stem_check.check_drum_stem(stems, ymldict)

    return yy, stems, rms_bool


if __name__ == "__main__":
    # Load YAML file configuations
    stream = open("configuration/config.yaml", "r")
    ymldict = yaml.safe_load(stream)

    # Load pretext task
    pt_task = ymldict.get("pt_task")

    if pt_task == "zerons":
        # Load the separation model:
        model = ymldict.get("spl_mod")
        m = "spleeter:{}".format(model)
        separator = Separator(m)

        # Spleet + stem check a directory
        full_fma_stem_check(separator, ymldict)

    elif pt_task == "clmr":
        # Generate CLMR examples
        gen_clmr(ymldict)

    else:
        raise ValueError("Which pretext task are we running?")
