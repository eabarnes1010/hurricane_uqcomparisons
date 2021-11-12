import json
import numpy as np

__author__ = "https://github.com/shouldsee"
__version__ = "29 October 2017"


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.

    Example
    -------
    with open("output.json", "w") as outfile:
        json.dump(metadata, outfile, indent="", cls=NumpyEncoder)

    References
    ----------
    * https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689

    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
