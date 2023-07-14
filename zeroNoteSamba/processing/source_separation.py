from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from spleeter.separator import Separator


def wv_run_spleeter(y: npt.NDArray[np.float32], sr: int, separator: Separator, model: str) -> Dict[Any, Any]:
    """
    Run Spleeter on a waveform file. Spleeter stems saved.
    -- y: input as np array
    -- model: Spleeter separation model
    """
    valid_models = {
        "2stems",
        "4stems",
        "5stems",
        "2stems-16kHz",
        "4stems-16kHz",
        "5stems-16kHz",
    }

    if model in valid_models:
        if "16kHz" in model and sr != 16000:
            raise Exception("Model is 16kHz but sound is not!")

        prediction = separator.separate(waveform=y)

        return prediction

    else:
        raise Exception("Model chosen is not one of 2stems, 4stems, and 5stems.")
