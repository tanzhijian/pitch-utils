from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ._markings import Markings


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def reflect(self, pivot: "Line") -> "Point":
        raise NotImplementedError


class Line:
    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end

    @property
    def length(self) -> float:
        return (
            (self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2
        ) ** 0.5

    @property
    def center(self) -> Point:
        raise NotImplementedError

    @property
    def is_vertical(self) -> bool:
        raise NotImplementedError

    def reflect(self, pivot: "Line") -> "Line":
        raise NotImplementedError


class Circle:
    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius

    def reflect(self, pivot: "Line") -> "Circle":
        raise NotImplementedError


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

    @property
    def coord_sys(self) -> CoordinateSystem:
        return self._coord_sys

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
    def canvas(self) -> Rectangle:
        raise NotImplementedError

    @property
    def centre_circle(self) -> Circle:
        raise NotImplementedError

    @property
    def centre_mark(self) -> Circle:
        raise NotImplementedError

    @property
    @abstractmethod
    def halfway_line(self) -> Line:
        raise NotImplementedError

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
    def canvas(self) -> Rectangle:
        return Rectangle(
            bottom_left=Point(self.touch_line.start.x, self.goal_line.start.y),
            width=self.touch_line.length,
            height=self.goal_line.length,
        )

    @property
    def halfway_line(self) -> Line:
        raise NotImplementedError

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
    def canvas(self) -> Rectangle:
        return Rectangle(
            bottom_left=Point(self.goal_line.start.x, self.touch_line.start.y),
            width=self.goal_line.length,
            height=self.touch_line.length,
        )

    @property
    def goal_line(self) -> Line:
        return Line(
            start=Point(
                x=self._goal_line_range[0], y=self._touch_line_range[0]
            ),
            end=Point(x=self._goal_line_range[1], y=self._touch_line_range[0]),
        )
