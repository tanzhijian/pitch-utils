from pitch_utils import LineRange, Pitch, Point, transform


def test_transform() -> None:
    point = Point(60, 40)
    from_pitch = Pitch(
        touch_line_range=LineRange(0, 100),
        goal_line_range=LineRange(0, 60),
    )
    to_pitch = Pitch(
        touch_line_range=LineRange(0, 50),
        goal_line_range=LineRange(0, 30),
    )
    transformed_point = transform(
        point, from_pitch=from_pitch, to_pitch=to_pitch
    )
    assert transformed_point.x == 30
    assert transformed_point.y == 20
