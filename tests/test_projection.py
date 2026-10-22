import numpy as np
import pytest

from segment_lidar.view import PinholeView, TopView


def test_top_view_uses_highest_point_and_preserves_labels():
    points = np.array(
        [
            [0, 0, -5, 1, 0, 0],
            [0, 0, -1, 0, 1, 0],
            [1, 1, 2, 0, 0, 1],
        ]
    )
    view = TopView()
    image = view.cloud_to_image(points, 1)
    assert image.shape == (2, 2, 3)
    np.testing.assert_array_equal(image[1, 0], [0, 255, 0])
    labels = np.array([[0, 7], [9, 0]], dtype=np.int32)
    np.testing.assert_array_equal(view.image_to_cloud(points, labels, 1), [9, 9, 7])
    np.testing.assert_array_equal(view.cloud_to_image(points[::-1], 1), image)


def test_pinhole_depth_and_behind_camera():
    points = np.array(
        [
            [0, 0, 4, 1, 0, 0],
            [0, 0, 2, 0, 1, 0],
            [0, 0, -2, 0, 0, 1],
        ]
    )
    K = np.array([[2, 0, 2], [0, 2, 2], [0, 0, 1]])
    view = PinholeView()
    image, intrinsics, pose = view.cloud_to_image(
        points,
        rotation=np.eye(3),
        translation=np.zeros(3),
        intrinsics=K,
        width=5,
        height=5,
    )
    np.testing.assert_array_equal(image[2, 2], [0, 255, 0])
    reversed_image, _, _ = view.cloud_to_image(
        points[::-1],
        rotation=np.eye(3),
        translation=np.zeros(3),
        intrinsics=K,
        width=5,
        height=5,
    )
    np.testing.assert_array_equal(reversed_image, image)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 12
    np.testing.assert_array_equal(
        view.image_to_cloud(points, labels, intrinsics, pose), [12, 12, -1]
    )


def test_pinhole_distance_filter_keeps_depths_aligned():
    points = np.array(
        [
            [0, 0, 10, 1, 0, 0],
            [0, 0, 2, 0, 1, 0],
        ]
    )
    K = np.array([[2, 0, 2], [0, 2, 2], [0, 0, 1]])
    image, _, _ = PinholeView().cloud_to_image(
        points,
        rotation=np.eye(3),
        translation=np.zeros(3),
        intrinsics=K,
        distance_threshold=3,
        width=5,
        height=5,
    )
    np.testing.assert_array_equal(image[2, 2], [0, 255, 0])
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 3
    _, K_out, pose = PinholeView().cloud_to_image(
        points,
        rotation=np.eye(3),
        translation=np.zeros(3),
        intrinsics=K,
        distance_threshold=3,
        width=5,
        height=5,
    )
    np.testing.assert_array_equal(
        PinholeView().image_to_cloud(
            points,
            labels,
            K_out,
            pose,
            distance_threshold=3,
        ),
        [-1, 3],
    )


@pytest.mark.parametrize("resolution", [0, -1, np.nan])
def test_invalid_resolution(resolution):
    with pytest.raises(ValueError):
        TopView().cloud_to_image(np.array([[0, 0, 0]]), resolution)
