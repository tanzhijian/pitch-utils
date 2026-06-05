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
        to_pitch.bottom.length / from_pitch.bottom.length,
        to_pitch.left.length / from_pitch.left.length,
    )
    return Locations(transformed_arr)
