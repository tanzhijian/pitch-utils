from typing import Sequence

import numpy as np
import numpy.typing as npt

LocationsTypes = (
    Sequence[int | float]
    | Sequence[Sequence[int | float]]
    | npt.NDArray[np.number]
)
