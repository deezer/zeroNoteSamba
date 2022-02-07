import mir_eval
import numpy as np

from madmom.features import DBNBeatTrackingProcessor

beat_dbn = DBNBeatTrackingProcessor(
    min_bpm=55, max_bpm=215, transition_lambda=100, fps=62.5, online=True
)


def beat_tracking(output, reference_beats, threshold=False, thresh_val=0.075, fps=62.5):
    """
    Compute F1-score using standard mir_eval function.
    -- output          : pulse output by model
    -- reference_beats : array of beat times
    """
    reference_beats = np.array(reference_beats)
    reference_beats = mir_eval.beat.trim_beats(reference_beats)

    if threshold == True:
        output = np.where(output > thresh_val, 1, 0)
        estimated_beats = []

        for x in range(len(output)):
            temp = output[x]
            if temp == 1:
                estimated_beats.append(x / fps)

        estimated_beats = np.asarray(estimated_beats)

    else:
        beat_dbn.reset()
        estimated_beats = beat_dbn.process_offline(output)

    estimated_beats = mir_eval.beat.trim_beats(estimated_beats)

    f_measure = mir_eval.beat.f_measure(
        reference_beats, estimated_beats, f_measure_threshold=0.07
    )
    cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(reference_beats, estimated_beats)
    info_gain = mir_eval.beat.information_gain(reference_beats, estimated_beats)

    return f_measure, cmlc, cmlt, amlc, amlt, info_gain
