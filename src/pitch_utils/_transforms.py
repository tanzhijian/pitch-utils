from ._models import Locations, Pitch
from ._types import LocationsTypes


def scale(
    locations: LocationsTypes,
    from_pitch: Pitch,
    to_pitch: Pitch,
) -> Locations:
    locs = Locations(locations)
    transformed_arr = locs._arr * (
        to_pitch.coord_sys.x_length / from_pitch.coord_sys.x_length,
        to_pitch.coord_sys.y_length / from_pitch.coord_sys.y_length,
    )
    return Locations(transformed_arr)
