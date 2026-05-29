from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import numpy.typing as npt

from ._types import LocationsTypes


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def rotate90_around(self, center: "Point") -> "Point":
        raise NotImplementedError


class Line:
    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end


class Circle:
    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius


class Rectangle:
    def __init__(
        self,
        bottom_left: Point,
        width: float,
        height: float,
    ) -> None:
        self.bottom_left = bottom_left
        self.width = width
        self.height = height


class Standard:
    TOUCH_LINE = 105.0
    GOAL_LINE = 68.0
    CENTER_CIRCLE_RADIUS = 9.15
    PENALTY_AREA_LENGTH = 16.5
    PENALTY_MARK_DISTANCE = 11.0
    GOAL_AREA_LENGTH = 5.5
    CORNER_ARC_RADIUS = 1.0
    GOAL_WIDTH = 7.32
    GOAL_HEIGHT = 2.44
    MARK_RADIUS = 0.1


@dataclass
class Markings:
    touch_line: float = Standard.TOUCH_LINE
    goal_line: float = Standard.GOAL_LINE
    center_circle_radius: float = Standard.CENTER_CIRCLE_RADIUS
    penalty_area_length: float = Standard.PENALTY_AREA_LENGTH
    penalty_mark_distance: float = Standard.PENALTY_MARK_DISTANCE
    goal_area_length: float = Standard.GOAL_AREA_LENGTH
    corner_arc_radius: float = Standard.CORNER_ARC_RADIUS
    goal_width: float = Standard.GOAL_WIDTH
    goal_height: float = Standard.GOAL_HEIGHT
    mark_radius: float = Standard.MARK_RADIUS

    def aspect_ratio(self) -> float:
        return self.goal_line / self.touch_line


class CoordinateSystem:
    origin = Point(0, 0)

    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range

    def _validate_axis(self, axis: tuple[float, float]) -> None:
        if len(axis) != 2:
            raise ValueError("Range must be a tuple of length 2")
        start, end = axis
        if not isinstance(start, (int, float)) or not isinstance(
            end, (int, float)
        ):
            raise TypeError("Range values must be numeric")
        if start == end:
            raise ValueError("Range values cannot be the same")

    @property
    def x_dir(self) -> Literal["left", "right"]:
        x0, x1 = self.x_range
        if x1 > x0:
            return "right"
        else:
            return "left"

    @property
    def y_dir(self) -> Literal["up", "down"]:
        y0, y1 = self.y_range
        if y1 > y0:
            return "up"
        else:
            return "down"

    @property
    def center(self) -> Point:
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        return Point(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
        )

    @property
    def x_length(self) -> float:
        x0, x1 = self.x_range
        return abs(x1 - x0)

    @property
    def y_length(self) -> float:
        y0, y1 = self.y_range
        return abs(y1 - y0)


class Pitch:
    def __init__(
        self,
        touch_line_range: tuple[float, float],
        goal_line_range: tuple[float, float],
        vertical: bool = False,
        markings: Markings | None = None,
        coord_sys: CoordinateSystem | None = None,
    ) -> None:
        self._tl_range = touch_line_range
        self._gl_range = goal_line_range
        self.vertical = vertical
        self.markings = (
            markings
            if markings is not None
            else Markings(
                touch_line=max(touch_line_range) - min(touch_line_range),
                goal_line=max(goal_line_range) - min(goal_line_range),
            )
        )
        self.coord_sys = (
            coord_sys
            if coord_sys is not None
            else CoordinateSystem(
                x_range=touch_line_range,
                y_range=goal_line_range,
            )
        )

    def pitch_area(self) -> Rectangle:
        raise NotImplementedError

    def halfway_line(self) -> Line:
        raise NotImplementedError

    def center_circle(self) -> Circle:
        raise NotImplementedError


class Locations:
    def __init__(
        self,
        data: LocationsTypes,
    ) -> None:
        self._arr = self._to_array(data)

    def _to_array(
        self,
        data: LocationsTypes,
    ) -> npt.NDArray[np.float64]:
        arr = np.asarray(data, dtype=np.float64)

        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError
            arr = arr[np.newaxis, :]

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError

        return arr

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return self._arr

    def to_list(self) -> list[list[float]]:
        return self._arr.tolist()

    def iter_tuples(self) -> Iterator[tuple[float, float]]:
        for row in self._arr:
            yield tuple(row)
