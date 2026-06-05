import pytest

from pitch_utils import (
    Circle,
    HorizontalPitch,
    Line,
    MarkingDimensions,
    Markings,
    Point,
    Rectangle,
    VerticalPitch,
)


class TestPoint:
    @pytest.fixture(scope="class")
    def point(self) -> Point:
        return Point(1, 1)

    def test_eq(self, point: Point) -> None:
        assert point == Point(1, 1)
        assert point != Point(2, 2)
        assert point != (1, 1)

    def test_lt(self, point: Point) -> None:
        assert point < Point(2, 1)
        assert point < Point(1, 2)
        with pytest.raises(TypeError):
            assert point < 2

    def test_coords(self, point: Point) -> None:
        assert point.coords == (1, 1)


class TestLine:
    @pytest.fixture(scope="class")
    def line(self) -> Line:
        return Line(Point(0, 0), Point(100, 0))

    def test_eq(self, line: Line) -> None:
        assert line == Line(Point(0, 0), Point(100, 0))
        assert line == Line(Point(100, 0), Point(0, 0))
        assert line != Line(Point(0, 0), Point(50, 0))
        assert line != 2

    def test_strictly_equal(self, line: Line) -> None:
        assert line.is_strictly_equal(Line(Point(0, 0), Point(100, 0)))
        assert not line.is_strictly_equal(Line(Point(100, 0), Point(0, 0)))

    def test_length(self, line: Line) -> None:
        assert line.length == 100

    def test_center(self, line: Line) -> None:
        center = line.center
        assert center.x == 50
        assert center.y == 0

    def test_coords(self, line: Line) -> None:
        assert line.coords == ((0, 0), (100, 0))


class TestCircle:
    @pytest.fixture(scope="class")
    def circle(self) -> Circle:
        return Circle(Point(0, 0), 10)

    def test_eq(self, circle: Circle) -> None:
        assert circle == Circle(Point(0, 0), 10)
        assert circle != Circle(Point(1, 1), 10)
        assert circle != Circle(Point(0, 0), 5)
        assert circle != 2


class TestRectangle:
    @pytest.fixture(scope="class")
    def rectangle(self) -> Rectangle:
        return Rectangle(min_point=Point(0, 0), max_point=Point(100, 60))

    def test_eq(self, rectangle: Rectangle) -> None:
        assert rectangle == Rectangle(Point(0, 0), Point(100, 60))
        assert rectangle != Rectangle(Point(1, 0), Point(100, 60))
        assert rectangle != 2

    def test_coords(self, rectangle: Rectangle) -> None:
        assert rectangle.coords == ((0, 0), (100, 0), (0, 60), (100, 60))


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
