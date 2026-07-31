import numpy as np
import pytest

from pitch_utils import (
    CoordinateSystem,
    HorizontalPitch,
    Point,
    Points,
    VerticalPitch,
)
from pitch_utils._transforms import (
    flip,
    reflact,
    scale,
    shift,
    transform,
    transpose,
)


@pytest.fixture
def horizontal_pitch() -> HorizontalPitch:
    return HorizontalPitch((0, 105), (0, 68))


@pytest.fixture
def reversed_horizontal_pitch() -> HorizontalPitch:
    return HorizontalPitch(
        (0, 105),
        (0, 68),
        coord_sys=CoordinateSystem(Point(0, 0), (105, 0), (68, 0)),
    )


@pytest.fixture
def vertical_pitch() -> VerticalPitch:
    return VerticalPitch((0, 105), (0, 68))


class TestTranspose:
    def test_transpose_between_pitch_orientations(
        self, horizontal_pitch: HorizontalPitch, vertical_pitch: VerticalPitch
    ) -> None:
        points = Points.from_rows([[0, 0], [105, 68]])

        assert transpose(
            points, horizontal_pitch, vertical_pitch
        ).to_list() == [
            [68.0, 0.0],
            [0.0, 105.0],
        ]


class TestFlip:
    def test_flip_aligns_coordinate_directions(
        self,
        horizontal_pitch: HorizontalPitch,
        reversed_horizontal_pitch: HorizontalPitch,
    ) -> None:
        points = Points.from_rows([[10, 20]])

        assert flip(
            points, horizontal_pitch, reversed_horizontal_pitch
        ).to_list() == [[95.0, 48.0]]


class TestScale:
    def test_scale_aligns_pitch_ranges(
        self, horizontal_pitch: HorizontalPitch
    ) -> None:
        target_pitch = HorizontalPitch(
            (100, 300),
            (-50, 250),
            coord_sys=CoordinateSystem(Point(0, 0), (100, 300), (-50, 250)),
        )
        points = Points.from_rows([[0, 0], [105, 68]])

        scaled = scale(points, horizontal_pitch, target_pitch)

        np.testing.assert_allclose(
            scaled.to_numpy(), np.array([[0, 0], [200, 300]])
        )


class TestShift:
    def test_shift_aligns_pitch_ranges(
        self, horizontal_pitch: HorizontalPitch
    ) -> None:
        target_pitch = HorizontalPitch(
            (100, 300),
            (-50, 250),
            coord_sys=CoordinateSystem(Point(0, 0), (100, 300), (-50, 250)),
        )
        points = Points.from_rows([[0, 0], [200, 300]])

        transformed = shift(points, horizontal_pitch, target_pitch)

        np.testing.assert_allclose(
            transformed.to_numpy(), np.array([[100, -50], [300, 250]])
        )


class TestTransform:
    def test_all_steps_align_transposed_reversed_pitch(
        self, horizontal_pitch: HorizontalPitch
    ) -> None:
        target_pitch = VerticalPitch(
            (0, 300),
            (0, 100),
            coord_sys=CoordinateSystem(Point(0, 0), (200, 100), (250, -50)),
        )
        points = Points.from_rows([[0, 0], [105, 68]])

        transformed = transform(points, horizontal_pitch, target_pitch)

        np.testing.assert_allclose(
            transformed.to_numpy(), np.array([[100, 250], [200, -50]])
        )


class TestReflact:
    def test_reflects_across_horizontal_pitch_halfway_line(
        self, horizontal_pitch: HorizontalPitch
    ) -> None:
        points = Points.from_rows([[10, 20], [52.5, 34]])

        assert reflact(points, horizontal_pitch).to_list() == [
            [95.0, 20.0],
            [52.5, 34.0],
        ]

    def test_reflects_across_vertical_pitch_halfway_line(
        self, vertical_pitch: VerticalPitch
    ) -> None:
        points = Points.from_rows([[20, 10], [34, 52.5]])

        assert reflact(points, vertical_pitch).to_list() == [
            [20.0, 95.0],
            [34.0, 52.5],
        ]
