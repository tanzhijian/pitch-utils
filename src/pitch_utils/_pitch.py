from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ._markings import Markings


class Point:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Point):
            return False
        return self.coords == value.coords

    def __repr__(self) -> str:
        return f"Point(x={self._x}, y={self._y})"

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def coords(self) -> tuple[float, float]:
        return (self._x, self._y)

    def reflect(self, pivot: "Line") -> "Point":
        raise NotImplementedError


class Line:
    def __init__(self, start: Point, end: Point) -> None:
        self._start = start
        self._end = end

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Line):
            return False
        return self.coords == value.coords

    def __repr__(self) -> str:
        return f"Line(start={self._start}, end={self._end})"

    @property
    def start(self) -> Point:
        return self._start

    @property
    def end(self) -> Point:
        return self._end

    @property
    def length(self) -> float:
        return (
            (self._end.x - self._start.x) ** 2
            + (self._end.y - self._start.y) ** 2
        ) ** 0.5

    @property
    def center(self) -> Point:
        return Point(
            x=(self._start.x + self._end.x) / 2,
            y=(self._start.y + self._end.y) / 2,
        )

    @property
    def coords(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.start.coords, self.end.coords)

    @property
    def is_vertical(self) -> bool:
        raise NotImplementedError

    def reflect(self, pivot: "Line") -> "Line":
        raise NotImplementedError


class Circle:
    def __init__(self, center: Point, radius: float) -> None:
        self._center = center
        self._radius = radius

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Circle):
            return False
        return self._center == value._center and self._radius == value._radius

    def __repr__(self) -> str:
        return f"Circle(center={self._center}, radius={self._radius})"

    @property
    def center(self) -> Point:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def reflect(self, pivot: "Line") -> "Circle":
        raise NotImplementedError


class Rectangle:
    def __init__(
        self,
        bottom_left: Point,
        width: float,
        height: float,
    ) -> None:
        self._bottom_left = bottom_left
        self._width = width
        self._height = height

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Rectangle):
            return False
        return self.coords == value.coords

    def __repr__(self) -> str:
        return (
            f"Rectangle(bottom_left={self._bottom_left}, "
            f"width={self._width}, height={self._height})"
        )

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def bottom_left(self) -> Point:
        return self._bottom_left

    @property
    def bottom_right(self) -> Point:
        return Point(self._bottom_left.x + self._width, self._bottom_left.y)

    @property
    def top_left(self) -> Point:
        return Point(self._bottom_left.x, self._bottom_left.y + self._height)

    @property
    def top_right(self) -> Point:
        return Point(
            self._bottom_left.x + self._width,
            self._bottom_left.y + self._height,
        )

    @property
    def left(self) -> Line:
        return Line(self._bottom_left, self.top_left)

    @property
    def right(self) -> Line:
        return Line(self.bottom_right, self.top_right)

    @property
    def bottom(self) -> Line:
        return Line(self._bottom_left, self.bottom_right)

    @property
    def top(self) -> Line:
        return Line(self.top_left, self.top_right)

    @property
    def coords(
        self,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        return (
            self._bottom_left.coords,
            self.bottom_right.coords,
            self.top_left.coords,
            self.top_right.coords,
        )

    def reflect(self, pivot: "Line") -> "Rectangle":
        raise NotImplementedError


@dataclass(frozen=True)
class CoordinateSystem:
    origin: tuple[float, float]
    x_dir: Literal["left", "right"]
    y_dir: Literal["up", "down"]


class Pitch(ABC):
    def __init__(
        self,
        touch_line_range: tuple[float, float],
        goal_line_range: tuple[float, float],
        markings: Markings | None = None,
        coord_sys: CoordinateSystem | None = None,
    ) -> None:
        self._touch_line_range = touch_line_range
        self._goal_line_range = goal_line_range

        self._markings = (
            markings
            if markings is not None
            else Markings(
                touch_line=self.touch_line.length,
                goal_line=self.goal_line.length,
            )
        )
        self._coord_sys = (
            coord_sys if coord_sys is not None else self._build_coord_sys()
        )

    @abstractmethod
    def _build_coord_sys(self) -> CoordinateSystem:
        raise NotImplementedError

    def __eq__(self, value: object) -> bool:
        if type(value) is not type(self):
            return False
        return self.bounds == value.bounds and self._markings == value.markings

    @property
    def coord_sys(self) -> CoordinateSystem:
        return self._coord_sys

    @property
    def markings(self) -> Markings:
        return self._markings

    @property
    @abstractmethod
    def touch_line(self) -> Line:
        raise NotImplementedError

    @property
    @abstractmethod
    def goal_line(self) -> Line:
        raise NotImplementedError

    @property
    @abstractmethod
    def bounds(self) -> Rectangle:
        raise NotImplementedError

    @property
    def centre_circle(self) -> Circle:
        return Circle(
            center=self.halfway_line.center,
            radius=self._markings.centre_circle_radius,
        )

    @property
    @abstractmethod
    def halfway_line(self) -> Line:
        raise NotImplementedError

    @property
    def centre_mark(self) -> Circle:
        return Circle(
            center=self.halfway_line.center,
            radius=self._markings.mark_radius,
        )

    @property
    @abstractmethod
    def penalty_arc_1(self) -> Circle:
        raise NotImplementedError

    @property
    @abstractmethod
    def penalty_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    @abstractmethod
    def penalty_mark_1(self) -> Circle:
        raise NotImplementedError

    @property
    @abstractmethod
    def goal_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def goal_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def penalty_arc_2(self) -> Circle:
        return self.penalty_arc_1.reflect(self.halfway_line)

    @property
    def penalty_area_2(self) -> Rectangle:
        return self.penalty_area_1.reflect(self.halfway_line)

    @property
    def penalty_mark_2(self) -> Circle:
        return self.penalty_mark_1.reflect(self.halfway_line)

    @property
    def goal_area_2(self) -> Rectangle:
        return self.goal_area_1.reflect(self.halfway_line)

    @property
    def goal_2(self) -> Rectangle:
        return self.goal_1.reflect(self.halfway_line)


class HorizontalPitch(Pitch):
    def _build_coord_sys(self) -> CoordinateSystem:
        x_dir = (
            "right"
            if self.touch_line.end.x > self.touch_line.start.x
            else "left"
        )
        y_dir = (
            "up" if self.goal_line.end.y > self.goal_line.start.y else "down"
        )
        return CoordinateSystem(origin=(0, 0), x_dir=x_dir, y_dir=y_dir)

    @property
    def touch_line(self) -> Line:
        return Line(
            start=Point(
                x=self._touch_line_range[0], y=self._goal_line_range[0]
            ),
            end=Point(x=self._touch_line_range[1], y=self._goal_line_range[0]),
        )

    @property
    def goal_line(self) -> Line:
        return Line(
            start=Point(
                x=self._touch_line_range[0], y=self._goal_line_range[0]
            ),
            end=Point(x=self._touch_line_range[0], y=self._goal_line_range[1]),
        )

    @property
    def bounds(self) -> Rectangle:
        return Rectangle(
            bottom_left=Point(self.touch_line.start.x, self.goal_line.start.y),
            width=self.touch_line.length,
            height=self.goal_line.length,
        )

    @property
    def halfway_line(self) -> Line:
        return Line(
            start=self.bounds.bottom.center,
            end=self.bounds.top.center,
        )

    @property
    def penalty_arc_1(self) -> Circle:
        raise NotImplementedError

    @property
    def penalty_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def penalty_mark_1(self) -> Circle:
        raise NotImplementedError

    @property
    def goal_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def goal_1(self) -> Rectangle:
        raise NotImplementedError

    def to_vertical(self) -> "VerticalPitch":
        raise NotImplementedError


class VerticalPitch(Pitch):
    def _build_coord_sys(self) -> CoordinateSystem:
        x_dir = (
            "right"
            if self.goal_line.end.x > self.goal_line.start.x
            else "left"
        )
        y_dir = (
            "up" if self.touch_line.end.y > self.touch_line.start.y else "down"
        )
        return CoordinateSystem(origin=(0, 0), x_dir=x_dir, y_dir=y_dir)

    @property
    def touch_line(self) -> Line:
        return Line(
            start=Point(
                x=self._goal_line_range[0], y=self._touch_line_range[0]
            ),
            end=Point(x=self._goal_line_range[0], y=self._touch_line_range[1]),
        )

    @property
    def goal_line(self) -> Line:
        return Line(
            start=Point(
                x=self._goal_line_range[0], y=self._touch_line_range[0]
            ),
            end=Point(x=self._goal_line_range[1], y=self._touch_line_range[0]),
        )

    @property
    def bounds(self) -> Rectangle:
        return Rectangle(
            bottom_left=Point(self.goal_line.start.x, self.touch_line.start.y),
            width=self.goal_line.length,
            height=self.touch_line.length,
        )

    @property
    def halfway_line(self) -> Line:
        return Line(
            start=self.bounds.left.center,
            end=self.bounds.right.center,
        )

    @property
    def penalty_arc_1(self) -> Circle:
        raise NotImplementedError

    @property
    def penalty_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def penalty_mark_1(self) -> Circle:
        raise NotImplementedError

    @property
    def goal_area_1(self) -> Rectangle:
        raise NotImplementedError

    @property
    def goal_1(self) -> Rectangle:
        raise NotImplementedError

    def to_horizontal(self) -> HorizontalPitch:
        raise NotImplementedError
