import os
import yaml

import librosa as audio_lib
import soundfile as sf

from spleeter.separator import Separator
from pathlib import Path

# File imports
import processing.utilities as utils
import processing.stem_check as stem_check
import processing.source_separation as source_separation


def full_fma_stem_check(separator, ymldict):
    """
    Function that writes csv files with good stem files.
    -- separator : Spleeter separator
    -- ymldict   : dictionary with yaml parameters
    """
    # Load desired variables
    fma_dir = ymldict.get("pt_data_dir")
    sr = ymldict.get("sample_rate")

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
                    Path("new_data/" + f.strip()[-10:-4]).mkdir(
                        parents=True, exist_ok=True
                    )

                    for key in stems:
                        stems[key] = utils.convert_to_mono(stems[key])
                        stems[key] = audio_lib.resample(
                            stems[key], sr, 16000, res_type="kaiser_fast"
                        )
                        sf.write(
                            "new_data/" + f.strip()[-10:-4] + "/" + key + ".wav",
                            stems[key],
                            16000,
                        )
                        print(
                            "   Saved "
                            + "new_data/"
                            + f.strip()[-10:-4]
                            + "/"
                            + key
                            + ".wav"
                        )

    return


def drum_load(filename, separator, sr, ymldict):
    """
    We run Spleeter on the input file.
    Return RMS status, full signal, and stems.
    -- filename  : file to stem check
    -- separator : Spleeter separator
    -- sr        : sample rate
    -- ymldict   : dictionary with yaml parameters
    """
    spl_mod = ymldict.get("spl_mod")

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

    # Load the separation model:
    model = ymldict.get("spl_mod")
    m = "spleeter:{}".format(model)
    separator = Separator(m)

    # Spleet + stem check a directory
    full_fma_stem_check(separator, ymldict)
