from dataclasses import dataclass, fields
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


@dataclass(frozen=True)
class MarkingDimensions:
    center_circle_radius: float = 9.15
    penalty_area_length: float = 16.5
    penalty_mark_distance: float = 11.0
    goal_area_length: float = 5.5
    corner_arc_radius: float = 1.0
    goal_width: float = 7.32
    goal_height: float = 2.44
    mark_radius: float = 0.1

    def scaled(self, factor: float) -> "MarkingDimensions":
        return MarkingDimensions(
            **{f.name: getattr(self, f.name) * factor for f in fields(self)}
        )

    def __mul__(self, factor: float) -> "MarkingDimensions":
        return self.scaled(factor)

    def __rmul__(self, factor: float) -> "MarkingDimensions":
        return self.scaled(factor)

    def __truediv__(self, factor: float) -> "MarkingDimensions":
        return self.scaled(1 / factor)


class Markings:
    _standard_touch_line = 105.0
    _max_touch_line = 110.0
    _min_touch_line = 100.0
    _standard_goal_line = 68.0
    _max_goal_line = 75.0
    _min_goal_line = 64.0

    def __init__(
        self,
        touch_line: float = _standard_touch_line,
        goal_line: float = _standard_goal_line,
        dimensions: Literal["metric", "scaled"]
        | MarkingDimensions
        | None = None,
    ) -> None:
        self._touch_line = touch_line
        self._goal_line = goal_line
        self._mode: Literal["metric", "scaled", "custom"]

        if isinstance(dimensions, MarkingDimensions):
            self._dimensions = dimensions
            self._mode = "custom"
        elif dimensions == "metric":
            self._dimensions = MarkingDimensions()
            self._mode = "metric"
        elif dimensions == "scaled":
            self._dimensions = self._scaled_dimensions()
            self._mode = "scaled"
        elif dimensions is None:
            if not self._is_standard_range():
                self._dimensions = self._scaled_dimensions()
                self._mode = "scaled"
            else:
                self._dimensions = MarkingDimensions()
                self._mode = "metric"
        else:
            raise ValueError("Invalid value for dimensions")

    def _is_standard_range(self) -> bool:
        return (
            self._min_touch_line <= self._touch_line <= self._max_touch_line
            and self._min_goal_line <= self._goal_line <= self._max_goal_line
        )

    def _scaled_dimensions(self) -> MarkingDimensions:
        return MarkingDimensions() / (
            self._touch_line / self._standard_touch_line
        )

    @property
    def mode(self) -> Literal["metric", "scaled", "custom"]:
        return self._mode

    @property
    def touch_line(self) -> float:
        return self._touch_line

    @property
    def goal_line(self) -> float:
        return self._goal_line

    @property
    def center_circle_radius(self) -> float:
        return self._dimensions.center_circle_radius

    @property
    def penalty_area_length(self) -> float:
        return self._dimensions.penalty_area_length

    @property
    def penalty_mark_distance(self) -> float:
        return self._dimensions.penalty_mark_distance

    @property
    def goal_area_length(self) -> float:
        return self._dimensions.goal_area_length

    @property
    def corner_arc_radius(self) -> float:
        return self._dimensions.corner_arc_radius

    @property
    def goal_width(self) -> float:
        return self._dimensions.goal_width

    @property
    def goal_height(self) -> float:
        return self._dimensions.goal_height

    @property
    def mark_radius(self) -> float:
        return self._dimensions.mark_radius

    def aspect_ratio(self) -> float:
        return self._goal_line / self._touch_line


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
