from typing import Any, Tuple

import librosa as audio_lib
import mir_eval  # type: ignore
import numpy as np
import numpy.typing as npt
from madmom.features import DBNBeatTrackingProcessor  # type: ignore
from torch import Tensor

beat_dbn = DBNBeatTrackingProcessor(min_bpm=55, max_bpm=215, transition_lambda=100, fps=62.5, online=True)


def beat_tracking(
    output: npt.NDArray[np.float32],
    reference: Tensor,
    threshold: bool = False,
    librosa: bool = False,
    thresh_val: float = 0.075,
    fps: float = 62.5,
) -> Tuple[Any, ...]:
    """
    Compute F1-score using standard mir_eval function.
    -- output: pulse output by model
    -- reference: array of beat times
    -- threshold: threshold activation for beat times
    -- librosa: use Ellis DP for beat times
    -- thresh_val: threshold value for beat times
    -- fps: features per second
    """
    reference_beats = np.array(reference)
    reference_beats = mir_eval.beat.trim_beats(reference_beats)

    if threshold == True and librosa == True:
        raise ValueError("\nWhich is it...thresholding or librosa?")

    if threshold == True:
        output = np.where(output > thresh_val, 1, 0)
        tmp_beats = []

        for x in range(len(output)):
            temp = output[x]
            if temp == 1:
                tmp_beats.append(x / fps)

        estimated_beats = np.asarray(tmp_beats)

    elif librosa == True:
        _, beats = audio_lib.beat.beat_track(sr=16000, onset_envelope=output, hop_length=256)
        estimated_beats = audio_lib.frames_to_time(beats, sr=16000, hop_length=256)

    else:
        beat_dbn.reset()
        try:
            estimated_beats = beat_dbn.process_offline(output)
        except:
            beat_dbn.correct = False
            estimated_beats = beat_dbn.process_offline(output)
            beat_dbn.correct = True

    estimated_beats = mir_eval.beat.trim_beats(estimated_beats)

    f_measure = mir_eval.beat.f_measure(reference_beats, estimated_beats, f_measure_threshold=0.07)
    cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(reference_beats, estimated_beats)
    info_gain = mir_eval.beat.information_gain(reference_beats, estimated_beats)

    return f_measure, cmlc, cmlt, amlc, amlt, info_gain
