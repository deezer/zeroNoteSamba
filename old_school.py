import mir_eval
import librosa
import numpy as np

from tqdm import tqdm


def dp_ellis(wavs, signals, real_times):
    test_f_measure, test_cmlc, test_cmlt, test_amlc, test_amlt, test_info_gain = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for wav in tqdm(wavs):
        signal = signals[wav]
        times = real_times[wav]

        _, beats = librosa.beat.beat_track(y=signal, sr=16000, hop_length=512)

        estimated_beats = librosa.frames_to_time(beats, sr=16000)
        estimated_beats = mir_eval.beat.trim_beats(estimated_beats)

        reference_beats = np.array(times)
        reference_beats = mir_eval.beat.trim_beats(reference_beats)

        f_measure = mir_eval.beat.f_measure(
            reference_beats, estimated_beats, f_measure_threshold=0.07
        )
        cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(
            reference_beats, estimated_beats
        )
        info_gain = mir_eval.beat.information_gain(reference_beats, estimated_beats)

        test_f_measure.append(f_measure)
        test_cmlc.append(cmlc)
        test_cmlt.append(cmlt)
        test_amlc.append(amlc)
        test_amlt.append(amlt)
        test_info_gain.append(info_gain)

    print("\n-- Full Set --")

    print(
        "Mean F1-score is {:.3f} +- {:.3f}.".format(
            np.mean(test_f_measure), np.std(test_f_measure)
        )
    )
    print(
        "Mean CMLC     is {:.3f} +- {:.3f}.".format(
            np.mean(test_cmlc), np.std(test_cmlc)
        )
    )
    print(
        "Mean CMLT     is {:.3f} +- {:.3f}.".format(
            np.mean(test_cmlt), np.std(test_cmlt)
        )
    )
    print(
        "Mean AMLC     is {:.3f} +- {:.3f}.".format(
            np.mean(test_amlc), np.std(test_amlc)
        )
    )
    print(
        "Mean AMLT     is {:.3f} +- {:.3f}.".format(
            np.mean(test_amlt), np.std(test_amlt)
        )
    )
    print(
        "Mean InfoGain is {:.3f} +- {:.3f}.".format(
            np.mean(test_info_gain), np.std(test_info_gain)
        )
    )

    return
