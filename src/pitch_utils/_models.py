from dataclasses import dataclass
from typing import NamedTuple


class LineRange(NamedTuple):
    start: float
    end: float


class Point(NamedTuple):
    x: float
    y: float


class Line(NamedTuple):
    start: Point
    end: Point

    def length(self) -> float:
        return (
            (self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2
        ) ** 0.5


class Circle(NamedTuple):
    center: Point
    radius: float


class Rectangle(NamedTuple):
    bottom_left: Point
    width: float
    height: float


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


class Coordinates:
    def __init__(
        self,
        x_axis: LineRange,
        y_axis: LineRange,
    ) -> None:
        self.x_axis = x_axis
        self.y_axis = y_axis

    @property
    def x_span(self) -> float:
        return max(self.x_axis) - min(self.x_axis)

    @property
    def y_span(self) -> float:
        return max(self.y_axis) - min(self.y_axis)


class Pitch:
    def __init__(
        self,
        touch_line_range: LineRange,
        goal_line_range: LineRange,
        markings: Markings | None = None,
        coordinates: Coordinates | None = None,
    ) -> None:
        self.touch_line_range = touch_line_range
        self.goal_line_range = goal_line_range
        self.markings = (
            markings
            if markings is not None
            else Markings(
                touch_line=max(touch_line_range) - min(touch_line_range),
                goal_line=max(goal_line_range) - min(goal_line_range),
            )
        )
        self.coordinates = (
            coordinates
            if coordinates is not None
            else Coordinates(
                x_axis=touch_line_range,
                y_axis=goal_line_range,
            )
        )
