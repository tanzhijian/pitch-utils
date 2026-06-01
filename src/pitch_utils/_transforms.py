from ._coordinates import Locations
from ._pitch import Pitch
from ._types import LocationsTypes


def scale(
    locations: LocationsTypes,
    from_pitch: Pitch,
    to_pitch: Pitch,
) -> Locations:
    locs = Locations(locations)
    transformed_arr = locs._arr * (
        to_pitch.canvas.width / from_pitch.canvas.width,
        to_pitch.canvas.height / from_pitch.canvas.height,
    )
    return Locations(transformed_arr)
