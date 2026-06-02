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
        to_pitch.bounds.width / from_pitch.bounds.width,
        to_pitch.bounds.height / from_pitch.bounds.height,
    )
    return Locations(transformed_arr)
