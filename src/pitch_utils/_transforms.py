from ._models import Pitch, Point


def transform(point: Point, from_pitch: Pitch, to_pitch: Pitch) -> Point:
    return Point(
        x=point.x
        * (to_pitch.coordinates.x_span / from_pitch.coordinates.x_span),
        y=point.y
        * (to_pitch.coordinates.y_span / from_pitch.coordinates.y_span),
    )
