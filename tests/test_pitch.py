import pytest

from pitch_utils import (
    HorizontalPitch,
    Line,
    MarkingDimensions,
    Markings,
    Point,
    VerticalPitch,
)


class TestHorizontalPitch:
    @pytest.fixture(scope="class")
    def pitch(self) -> HorizontalPitch:
        return HorizontalPitch(
            touch_line_range=(0, 105),
            goal_line_range=(0, 68),
        )

    def test_eq(self, pitch: HorizontalPitch) -> None:
        assert pitch == HorizontalPitch(
            touch_line_range=(0, 105),
            goal_line_range=(0, 68),
        )
        assert pitch != HorizontalPitch(
            touch_line_range=(0, 100),
            goal_line_range=(0, 68),
        )
        assert pitch != HorizontalPitch(
            touch_line_range=(0, 105),
            goal_line_range=(0, 60),
        )
        assert pitch != HorizontalPitch(
            touch_line_range=(0, 105),
            goal_line_range=(0, 68),
            markings=Markings(
                touch_line=105,
                goal_line=68,
                spec=MarkingDimensions(penalty_mark_distance=12),
            ),
        )
        assert pitch != VerticalPitch(
            touch_line_range=(0, 105),
            goal_line_range=(0, 68),
        )

    def test_coord_sys(self, pitch: HorizontalPitch) -> None:
        coord_sys = pitch.coord_sys
        assert coord_sys.x_dir == "right"
        assert coord_sys.y_dir == "up"
        assert coord_sys.origin == (0, 0)

    def test_markings(self, pitch: HorizontalPitch) -> None:
        assert pitch.markings.touch_line == 105
        assert pitch.markings.goal_line == 68
        assert pitch.markings.penalty_mark_distance == 11

    def test_corners(self, pitch: HorizontalPitch) -> None:
        assert pitch.bottom_left == Point(0, 0)
        assert pitch.bottom_right == Point(105, 0)
        assert pitch.top_left == Point(0, 68)
        assert pitch.top_right == Point(105, 68)

    def test_sides(self, pitch: HorizontalPitch) -> None:
        assert pitch.bottom.is_strictly_equal(Line(Point(0, 0), Point(105, 0)))
        assert pitch.top.is_strictly_equal(Line(Point(0, 68), Point(105, 68)))
        assert pitch.left.is_strictly_equal(Line(Point(0, 0), Point(0, 68)))
        assert pitch.right.is_strictly_equal(
            Line(Point(105, 0), Point(105, 68))
        )

    def test_centre_circle(self, pitch: HorizontalPitch) -> None:
        centre_circle = pitch.centre_circle
        assert centre_circle.center == Point(52.5, 34)
        assert centre_circle.radius == 9.15

    def test_halfway_line(self, pitch: HorizontalPitch) -> None:
        halfway_line = pitch.halfway_line
        assert halfway_line.start == Point(52.5, 0)
        assert halfway_line.end == Point(52.5, 68)

    def test_centre_mark(self, pitch: HorizontalPitch) -> None:
        centre_mark = pitch.centre_mark
        assert centre_mark.center == Point(52.5, 34)
        assert centre_mark.radius == 0.1

    def test_left_penalty_arc(self, pitch: HorizontalPitch) -> None:
        arc = pitch.left_penalty_arc
        assert arc.center == Point(11, 34)
        assert arc.radius == 9.15

    def test_left_penalty_mark(self, pitch: HorizontalPitch) -> None:
        mark = pitch.left_penalty_mark
        assert mark.center == Point(11, 34)
        assert mark.radius == 0.1

    def test_left_penalty_area(self, pitch: HorizontalPitch) -> None:
        area = pitch.left_penalty_area
        assert area.coords == (
            (0, 13.84),
            (0, 54.16),
            (16.5, 13.84),
            (16.5, 54.16),
        )

    def test_left_goal_area(self, pitch: HorizontalPitch) -> None:
        area = pitch.left_goal_area
        assert area.coords == (
            (0, 24.84),
            (0, 43.16),
            (5.5, 24.84),
            (5.5, 43.16),
        )

    def test_left_goal(self, pitch: HorizontalPitch) -> None:
        assert pitch.left_goal.coords == (
            (-2.44, 30.34),
            (-2.44, 37.66),
            (0, 30.34),
            (0, 37.66)
        )


class TestVerticalPitch:
    @pytest.fixture(scope="class")
    def pitch(self) -> VerticalPitch:
        return VerticalPitch(
            touch_line_range=(0, 105),
            goal_line_range=(0, 68),
        )

    def test_coord_sys(self, pitch: VerticalPitch) -> None:
        coord_sys = pitch.coord_sys
        assert coord_sys.x_dir == "right"
        assert coord_sys.y_dir == "up"
        assert coord_sys.origin == (0, 0)

    def test_corners(self, pitch: VerticalPitch) -> None:
        assert pitch.bottom_left == Point(0, 0)
        assert pitch.bottom_right == Point(68, 0)
        assert pitch.top_left == Point(0, 105)
        assert pitch.top_right == Point(68, 105)

    def test_halfway_line(self, pitch: VerticalPitch) -> None:
        halfway_line = pitch.halfway_line
        assert halfway_line.start == Point(0, 52.5)
        assert halfway_line.end == Point(68, 52.5)
