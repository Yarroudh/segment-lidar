"""Project point clouds into images and lift image labels back to points."""

from __future__ import annotations

import numpy as np


def _points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] not in (3, 6) or len(points) == 0:
        raise ValueError("points must be a nonempty array of shape (N, 3) or (N, 6)")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")
    return points


def _colors(points: np.ndarray) -> np.ndarray:
    if points.shape[1] == 3:
        return np.full((len(points), 3), 255, dtype=np.uint8)
    colors = points[:, 3:6]
    if colors.size and colors.max() <= 1:
        colors = colors * 255
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def _labels_at(image: np.ndarray, x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("label image must have shape (H, W) or (H, W, C)")
    valid &= (x >= 0) & (y >= 0) & (x < image.shape[1]) & (y < image.shape[0])
    labels = np.full(len(x), -1, dtype=np.int32)
    if image.ndim == 2:
        values = image[y[valid], x[valid]]
        labels[valid] = np.where(values > 0, values, -1).astype(np.int32)
    else:
        pixels = image[y[valid], x[valid], :3]
        unique, inverse = np.unique(pixels, axis=0, return_inverse=True)
        ids = inverse.astype(np.int32) + 1
        ids[np.all(unique[inverse] == 0, axis=1)] = -1
        labels[valid] = ids
    return labels


class TopView:
    """Orthographic top view, with the highest point visible in each pixel."""

    def _pixels(
        self, points: np.ndarray, resolution: float
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        points = _points(points)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be positive and finite")
        minx, miny = points[:, :2].min(axis=0)
        maxx, maxy = points[:, :2].max(axis=0)
        x = np.floor((points[:, 0] - minx) / resolution + 1e-9).astype(np.int64)
        y = np.floor((maxy - points[:, 1]) / resolution + 1e-9).astype(np.int64)
        width = int(np.floor((maxx - minx) / resolution + 1e-9)) + 1
        height = int(np.floor((maxy - miny) / resolution + 1e-9)) + 1
        return x, y, width, height

    def cloud_to_image(self, points: np.ndarray, resolution: float) -> np.ndarray:
        points = _points(points)
        x, y, width, height = self._pixels(points, resolution)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        flat = y * width + x
        order = np.lexsort((points[:, 2], flat))
        winners = order[np.r_[flat[order][1:] != flat[order][:-1], True]]
        image[y[winners], x[winners]] = _colors(points)[winners]
        return image

    def image_to_cloud(
        self, points: np.ndarray, image: np.ndarray, resolution: float, transform=None
    ) -> np.ndarray:
        if transform is None:
            x, y, _, _ = self._pixels(points, resolution)
        else:
            points = _points(points)
            inverse = ~transform
            x = np.floor(
                inverse.a * points[:, 0] + inverse.b * points[:, 1] + inverse.c + 1e-9
            ).astype(np.int64)
            y = np.floor(
                inverse.d * points[:, 0] + inverse.e * points[:, 1] + inverse.f + 1e-9
            ).astype(np.int64)
        return _labels_at(image, x, y, np.ones(len(x), dtype=bool))


class PinholeView:
    """Pinhole projection using a 4 by 4 world to camera pose and 3 by 3 intrinsics."""

    def __init__(self, interactive: bool = False) -> None:
        self.interactive = interactive

    @staticmethod
    def _camera(rotation, translation, intrinsics):
        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float).reshape(-1)
        intrinsics = np.asarray(intrinsics, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,) or intrinsics.shape != (3, 3):
            raise ValueError(
                "rotation, translation and intrinsics must have shapes (3,3), (3,), (3,3)"
            )
        if not all(np.isfinite(a).all() for a in (rotation, translation, intrinsics)):
            raise ValueError("camera parameters must be finite")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, translation
        return pose, intrinsics

    @staticmethod
    def _project(points, intrinsics, pose):
        camera = points[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        depth = camera[:, 2]
        valid = np.isfinite(depth) & (depth > 0)
        xy = np.zeros((len(points), 2), dtype=float)
        xy[valid] = camera[valid, :2] / depth[valid, None]
        uv = xy @ intrinsics[:2, :2].T + intrinsics[:2, 2]
        return np.floor(uv).astype(np.int64), depth, valid

    def cloud_to_image(
        self,
        points: np.ndarray,
        resolution: float = 0.1,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        distance_threshold: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = _points(points)
        if self.interactive:
            raise NotImplementedError(
                "Interactive camera capture is not supported; provide camera matrices"
            )
        pose, intrinsics = self._camera(rotation, translation, intrinsics)
        width = width or int(np.ceil(2 * intrinsics[0, 2]))
        height = height or int(np.ceil(2 * intrinsics[1, 2]))
        if width <= 0 or height <= 0:
            raise ValueError("image width and height must be positive")
        uv, depth, valid = self._project(points, intrinsics, pose)
        if distance_threshold is not None:
            if distance_threshold <= 0:
                raise ValueError("distance_threshold must be positive")
            valid &= depth <= distance_threshold
        x, y = uv.T
        valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        candidates = np.flatnonzero(valid)
        flat = y[candidates] * width + x[candidates]
        order = np.lexsort((-depth[candidates], flat))
        winners = candidates[order[np.r_[flat[order][1:] != flat[order][:-1], True]]]
        image[y[winners], x[winners]] = _colors(points)[winners]
        return image, intrinsics, pose

    def image_to_cloud(
        self,
        points: np.ndarray,
        image: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        distance_threshold: float | None = None,
    ) -> np.ndarray:
        points = _points(points)
        extrinsics = np.asarray(extrinsics, dtype=float)
        if extrinsics.shape != (4, 4):
            raise ValueError("extrinsics must have shape (4, 4)")
        _, intrinsics = self._camera(extrinsics[:3, :3], extrinsics[:3, 3], intrinsics)
        uv, depth, valid = self._project(points, intrinsics, extrinsics)
        if distance_threshold is not None:
            if distance_threshold <= 0:
                raise ValueError("distance_threshold must be positive")
            valid &= depth <= distance_threshold
        return _labels_at(image, uv[:, 0], uv[:, 1], valid)
