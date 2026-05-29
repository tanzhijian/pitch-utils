from pitch_utils import Pitch, scale


def test_scale() -> None:
    from_pitch = Pitch((0, 100), (0, 60))
    to_pitch = Pitch((0, 50), (0, 30))
    locations = ((60, 40), (20.4, 10.2))
    transformed_locs = scale(locations, from_pitch, to_pitch)
    assert transformed_locs.to_list() == [[30.0, 20.0], [10.2, 5.1]]
