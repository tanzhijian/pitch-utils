import numpy as np

from ._coordinates import CoordinateSystem, Points
from ._pitch import Pitch


def _source_coord_sys(from_pitch: Pitch, to_pitch: Pitch) -> CoordinateSystem:
    if type(from_pitch) is not type(to_pitch):
        return from_pitch.coord_sys.transposed()
    return from_pitch.coord_sys


def _aligned_coord_sys(from_pitch: Pitch, to_pitch: Pitch) -> CoordinateSystem:
    coord_sys = _source_coord_sys(from_pitch, to_pitch)
    return coord_sys.flipped(
        x=coord_sys.x_dir != to_pitch.coord_sys.x_dir,
        y=coord_sys.y_dir != to_pitch.coord_sys.y_dir,
    )


def transpose(points: Points, from_pitch: Pitch, to_pitch: Pitch) -> Points:
    if type(from_pitch) is type(to_pitch):
        return Points.from_numpy(points.to_numpy())
    return from_pitch.coord_sys.transpose(points)


def flip(points: Points, from_pitch: Pitch, to_pitch: Pitch) -> Points:
    coord_sys = _source_coord_sys(from_pitch, to_pitch)
    coordinates = points.to_numpy()
    if coord_sys.x_dir != to_pitch.coord_sys.x_dir:
        coordinates[:, 0] = (
            coord_sys.x_range.start + coord_sys.x_range.end - coordinates[:, 0]
        )
    if coord_sys.y_dir != to_pitch.coord_sys.y_dir:
        coordinates[:, 1] = (
            coord_sys.y_range.start + coord_sys.y_range.end - coordinates[:, 1]
        )
    return Points.from_numpy(coordinates)


def scale(points: Points, from_pitch: Pitch, to_pitch: Pitch) -> Points:
    from_coord_sys = _aligned_coord_sys(from_pitch, to_pitch)
    to_coord_sys = to_pitch.coord_sys
    return points.scaled(
        to_coord_sys.x_range.length / from_coord_sys.x_range.length,
        to_coord_sys.y_range.length / from_coord_sys.y_range.length,
        from_coord_sys.origin.__class__(
            from_coord_sys.x_range.start,
            from_coord_sys.y_range.start,
        ),
    )


def shift(points: Points, from_pitch: Pitch, to_pitch: Pitch) -> Points:
    from_coord_sys = _aligned_coord_sys(from_pitch, to_pitch)
    to_coord_sys = to_pitch.coord_sys
    offset = np.array(
        [
            to_coord_sys.x_range.start - from_coord_sys.x_range.start,
            to_coord_sys.y_range.start - from_coord_sys.y_range.start,
        ]
    )
    return Points.from_numpy(points.to_numpy() + offset)


def transform(points: Points, from_pitch: Pitch, to_pitch: Pitch) -> Points:
    transposed = transpose(points, from_pitch, to_pitch)
    flipped = flip(transposed, from_pitch, to_pitch)
    scaled = scale(flipped, from_pitch, to_pitch)
    return shift(scaled, from_pitch, to_pitch)


def reflact(points: Points, pitch: Pitch) -> Points:
    return pitch.coord_sys.reflect(points, pitch.halfway_line)
