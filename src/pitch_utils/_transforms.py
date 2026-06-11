from typing import Sequence

import numpy as np
import numpy.typing as npt

from ._coordinates import Locations
from ._pitch import Pitch


def scale(
    locations: Sequence[Sequence[int | float]] | npt.NDArray[np.number],
    from_pitch: Pitch,
    to_pitch: Pitch,
) -> Locations:
    locs = Locations.from_array(locations)
    transformed_arr = locs._arr * (
        to_pitch.bottom.length / from_pitch.bottom.length,
        to_pitch.left.length / from_pitch.left.length,
    )
    return Locations(transformed_arr)
