import math
from dataclasses import dataclass, fields
from typing import Literal


@dataclass(frozen=True)
class MarkingDimensions:
    centre_circle_radius: float = 9.15
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

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, MarkingDimensions):
            return False
        return all(
            math.isclose(getattr(self, f.name), getattr(value, f.name))
            for f in fields(self)
        )


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
        spec: Literal["standard", "scaled"] | MarkingDimensions | None = None,
    ) -> None:
        self._touch_line = touch_line
        self._goal_line = goal_line
        self._mode: Literal["standard", "scaled", "custom"]

        if isinstance(spec, MarkingDimensions):
            self._dims = spec
            self._mode = "custom"
        elif spec == "standard":
            self._dims = MarkingDimensions()
            self._mode = "standard"
        elif spec == "scaled":
            self._dims = self._scaled_dimensions()
            self._mode = "scaled"
        elif spec is None:
            if not self._is_regulation_range():
                self._dims = self._scaled_dimensions()
                self._mode = "scaled"
            else:
                self._dims = MarkingDimensions()
                self._mode = "standard"
        else:
            raise ValueError("Invalid value for spec")

    def _is_regulation_range(self) -> bool:
        return (
            self._min_touch_line <= self._touch_line <= self._max_touch_line
            and self._min_goal_line <= self._goal_line <= self._max_goal_line
        )

    def _scaled_dimensions(self) -> MarkingDimensions:
        return MarkingDimensions() * (
            self._touch_line / self._standard_touch_line
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Markings):
            return False
        return (
            math.isclose(self.touch_line, value.touch_line)
            and math.isclose(self.goal_line, value.goal_line)
            and self.mode == value.mode
            and self.dims == value.dims
        )

    def __repr__(self) -> str:
        return (
            f"Markings(touch_line={self._touch_line}, "
            f"goal_line={self._goal_line}, "
            f"mode='{self._mode}', "
            f"dims={self._dims})"
        )

    @property
    def mode(self) -> Literal["standard", "scaled", "custom"]:
        return self._mode

    @property
    def dims(self) -> MarkingDimensions:
        return self._dims

    @property
    def touch_line(self) -> float:
        return self._touch_line

    @property
    def goal_line(self) -> float:
        return self._goal_line

    @property
    def centre_circle_radius(self) -> float:
        return self._dims.centre_circle_radius

    @property
    def penalty_area_length(self) -> float:
        return self._dims.penalty_area_length

    @property
    def penalty_mark_distance(self) -> float:
        return self._dims.penalty_mark_distance

    @property
    def goal_area_length(self) -> float:
        return self._dims.goal_area_length

    @property
    def corner_arc_radius(self) -> float:
        return self._dims.corner_arc_radius

    @property
    def goal_width(self) -> float:
        return self._dims.goal_width

    @property
    def goal_height(self) -> float:
        return self._dims.goal_height

    @property
    def mark_radius(self) -> float:
        return self._dims.mark_radius

    def aspect_ratio(self) -> float:
        return self._goal_line / self._touch_line
