"""LiDAR I/O, image segmentation, and label transfer."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import rasterio

from .view import PinholeView, TopView, _points


def _parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_raster(path: str, *, with_transform: bool = False):
    with rasterio.open(path) as src:
        bands = src.read()
        transform = src.transform
    image = bands[0] if len(bands) == 1 else np.moveaxis(bands[:3], 0, -1)
    return (image, transform) if with_transform else image


def _write_raster(path: str, array: np.ndarray, *, transform=None) -> None:
    _parent(path)
    bands = array[None] if array.ndim == 2 else np.moveaxis(array, -1, 0)
    options = dict(
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=len(bands),
        dtype=array.dtype,
    )
    if transform is not None:
        options["transform"] = transform
    with rasterio.open(path, "w", **options) as dst:
        dst.write(bands)


def _labels_from_masks(masks: np.ndarray, shape: tuple[int, int], scores=None) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.size == 0:
        return np.zeros(shape, dtype=np.int32)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[1:] != shape:
        raise ValueError("model masks must have shape (N, H, W)")
    labels = np.zeros(shape, dtype=np.int32)
    if scores is None:
        scores = np.ones(len(masks))
    scores = np.asarray(scores).reshape(-1)
    if len(scores) != len(masks):
        raise ValueError("model scores and masks have different lengths")
    # Low confidence first; high confidence wins where masks overlap.
    for index in np.argsort(scores, kind="stable"):
        labels[masks[index].astype(bool)] = int(index) + 1
    return labels


def _prompt_inputs(text, boxes, box_labels) -> dict:
    kwargs = {}
    if text is not None:
        if not isinstance(text, str):
            raise ValueError("text prompt must be a string")
        text = text.strip()
    if text:
        kwargs["text"] = text
    if boxes is None:
        if box_labels is not None:
            raise ValueError("box_labels require boxes")
        if not text:
            raise ValueError("SAM 3 needs a text prompt or at least one box")
        return kwargs
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0:
        raise ValueError("boxes must have shape (N, 4)")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("each box must have finite coordinates with positive area")
    if box_labels is None:
        box_labels = np.ones(len(boxes), dtype=int)
    box_labels = np.asarray(box_labels)
    if box_labels.shape != (len(boxes),) or not np.isin(box_labels, [0, 1]).all():
        raise ValueError("box_labels must contain one 0 or 1 for each box")
    if not text and not np.any(box_labels == 1):
        raise ValueError("a box prompt without text needs a positive box")
    kwargs["input_boxes"] = [boxes.tolist()]
    kwargs["input_boxes_labels"] = [box_labels.astype(int).tolist()]
    return kwargs


@dataclass
class TextPrompt:
    text: str | None = None
    score_threshold: float = 0.5
    mask_threshold: float = 0.5


@dataclass
class AutomaticMasks:
    points_per_crop: int = 32
    points_per_batch: int = 16
    pred_iou_thresh: float = 0.65
    stability_score_thresh: float = 0.7


class SamLidar:
    """Segment a point cloud through a top or pinhole raster view.

    Model loading is deferred until :meth:`segment` needs inference. Existing
    label rasters can therefore be projected without installing a model.
    """

    text_prompt = TextPrompt

    def __init__(
        self,
        resolution: float = 0.25,
        height: int = 512,
        width: int = 512,
        distance_threshold: float | None = None,
        device: str | None = None,
        intrinsics: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        hf_token: str | None = None,
        model_id: str = "facebook/sam3",
    ) -> None:
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be positive and finite")
        if height <= 0 or width <= 0:
            raise ValueError("image width and height must be positive")
        self.resolution = resolution
        self.height, self.width = height, width
        self.distance_threshold = distance_threshold
        self.device = device
        self.intrinsics, self.rotation, self.translation = intrinsics, rotation, translation
        self.hf_token = hf_token
        self.model_id = model_id
        self.text_prompt = TextPrompt()
        self.automatic_masks = AutomaticMasks()
        self._model = None
        self._model_mode = None

    def _token(self) -> str:
        token = (self.hf_token or os.environ.get("HF_TOKEN") or "").strip()
        if not token:
            raise ValueError(
                "A Hugging Face token is required for SAM 3. Request access at "
                f"https://huggingface.co/{self.model_id}, then pass hf_token or set HF_TOKEN."
            )
        return token

    def _infer_automatic(self, image: np.ndarray) -> np.ndarray:
        options = self.automatic_masks
        if options.points_per_crop <= 0 or options.points_per_batch <= 0:
            raise ValueError("automatic mask point counts must be positive")
        if not all(
            0 <= value <= 1 for value in (options.pred_iou_thresh, options.stability_score_thresh)
        ):
            raise ValueError("automatic mask thresholds must be between 0 and 1")
        token = self._token() if self._model_mode != "automatic" else None
        try:
            import torch
            from PIL import Image
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Install segment-lidar[sam3] to run SAM 3") from exc
        if self._model_mode != "automatic":
            self._model = None
            gc.collect()
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                generator = pipeline(
                    "mask-generation", model=self.model_id, token=token, device=device
                )
            except Exception as exc:
                message = str(exc).lower()
                if any(term in message for term in ("gated", "401", "403", "access request")):
                    raise PermissionError(
                        "SAM 3 access is pending or denied. Sign in at "
                        f"https://huggingface.co/{self.model_id}, request access, "
                        "accept the terms, and use a token from the approved account."
                    ) from None
                raise
            self._model = generator
            self._model_mode = "automatic"
        result = self._model(
            Image.fromarray(image),
            points_per_crop=options.points_per_crop,
            points_per_batch=options.points_per_batch,
            pred_iou_thresh=options.pred_iou_thresh,
            stability_score_thresh=options.stability_score_thresh,
        )
        masks = []
        for mask in result["masks"]:
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            masks.append(np.asarray(mask))
        scores = result["scores"]
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        return _labels_from_masks(
            np.stack(masks) if masks else np.empty((0,)), image.shape[:2], scores
        )

    def read(self, path: str, classification: int | None = None) -> np.ndarray:
        """Read LAS/LAZ or NPY into XYZ or XYZRGB (RGB normalized to [0, 1])."""
        suffix = Path(path).suffix.lower()
        if suffix == ".npy":
            points = _points(np.load(path))
            if classification is not None:
                raise ValueError("classification filtering requires LAS or LAZ")
            return points
        if suffix not in {".las", ".laz"}:
            raise ValueError("input must be a .las, .laz, or .npy file")
        las = laspy.read(path)
        selected = np.ones(len(las.points), dtype=bool)
        if classification is not None:
            if "classification" not in las.point_format.dimension_names:
                raise ValueError("LAS file has no classification dimension")
            selected = np.asarray(las.classification) == classification
        xyz = np.asarray(las.xyz)[selected]
        dims = set(las.point_format.dimension_names)
        if {"red", "green", "blue"} <= dims:
            rgb = np.column_stack((las.red, las.green, las.blue))[selected]
            scale = 255.0 if rgb.size and rgb.max() <= 255 else 65535.0
            rgb = rgb / scale
            return np.column_stack((xyz, rgb))
        return xyz

    def csf(
        self,
        points: np.ndarray,
        class_threshold: float = 0.5,
        cloth_resolution: float = 0.2,
        iterations: int = 500,
        slope_smooth: bool = False,
        csf_path: str | None = None,
        exists: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return points above ground and their indices, plus ground indices."""
        points = _points(points)
        if exists:
            if csf_path is None:
                raise ValueError("csf_path is required when exists=True")
            las = laspy.read(csf_path)
            if (
                len(las.points) != len(points)
                or "ground" not in las.point_format.extra_dimension_names
            ):
                raise ValueError("saved CSF file does not match the input cloud")
            ground = np.flatnonzero(np.asarray(las.ground) == 1)
            non_ground = np.flatnonzero(np.asarray(las.ground) == 0)
        else:
            try:
                import CSF
            except ImportError as exc:
                raise ImportError("Install segment-lidar[csf] to use ground filtering") from exc
            csf = CSF.CSF()
            csf.params.bSloopSmooth = slope_smooth
            csf.params.cloth_resolution = cloth_resolution
            csf.params.interations = iterations
            csf.params.class_threshold = class_threshold
            csf.setPointCloud(points[:, :3])
            ground_vec, non_ground_vec = CSF.VecInt(), CSF.VecInt()
            csf.do_filtering(ground_vec, non_ground_vec)
            ground = np.asarray(ground_vec, dtype=np.int64)
            non_ground = np.asarray(non_ground_vec, dtype=np.int64)
            if csf_path:
                header = laspy.LasHeader(
                    point_format=3 if points.shape[1] == 6 else 0, version="1.3"
                )
                las = laspy.LasData(header)
                las.xyz = points[:, :3]
                las.add_extra_dim(laspy.ExtraBytesParams(name="ground", type=np.uint8))
                flags = np.zeros(len(points), dtype=np.uint8)
                flags[ground] = 1
                las.ground = flags
                _parent(csf_path)
                las.write(csf_path)
        return points[non_ground], non_ground, ground

    def _infer(
        self,
        image: np.ndarray,
        prompt: str | None = None,
        boxes: np.ndarray | None = None,
        box_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        text = prompt if prompt is not None else self.text_prompt.text
        if (text is None or isinstance(text, str) and not text.strip()) and boxes is None:
            if box_labels is not None:
                raise ValueError("box_labels require boxes")
            return self._infer_automatic(image)
        prompt_inputs = _prompt_inputs(text, boxes, box_labels)
        if "input_boxes" in prompt_inputs:
            box_array = np.asarray(prompt_inputs["input_boxes"][0])
            if (
                np.any(box_array[:, :2] < 0)
                or np.any(box_array[:, 2] > image.shape[1])
                or np.any(box_array[:, 3] > image.shape[0])
            ):
                raise ValueError("boxes must fit inside the image")
        for value in (self.text_prompt.score_threshold, self.text_prompt.mask_threshold):
            if not 0 <= value <= 1:
                raise ValueError("SAM 3 thresholds must be between 0 and 1")
        token = self._token() if self._model_mode != "prompted" else None
        try:
            import torch
            from PIL import Image
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise ImportError("Install segment-lidar[sam3] to run SAM 3") from exc
        if self._model_mode != "prompted":
            self._model = None
            gc.collect()
            try:
                model = Sam3Model.from_pretrained(self.model_id, token=token)
                processor = Sam3Processor.from_pretrained(self.model_id, token=token)
            except Exception as exc:
                message = str(exc).lower()
                if any(term in message for term in ("gated", "401", "403", "access request")):
                    raise PermissionError(
                        "SAM 3 access is pending or denied. Sign in at "
                        f"https://huggingface.co/{self.model_id}, request access, "
                        "accept the terms, and use a token from the approved account."
                    ) from None
                raise
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device).eval()
            self._model = (model, processor)
            self._model_mode = "prompted"
        model, processor = self._model
        kwargs = {"images": Image.fromarray(image), "return_tensors": "pt"}
        kwargs.update(prompt_inputs)
        inputs = processor(**kwargs).to(model.device)
        with torch.inference_mode():
            output = model(**inputs)
        result = processor.post_process_instance_segmentation(
            output,
            threshold=self.text_prompt.score_threshold,
            mask_threshold=self.text_prompt.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        masks = result["masks"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        return _labels_from_masks(masks, image.shape[:2], scores)

    def segment(
        self,
        points: np.ndarray,
        view: TopView | PinholeView | None = None,
        image_path: str = "raster.tif",
        labels_path: str = "labeled.tif",
        image_exists: bool = False,
        label_exists: bool = False,
        prompt: str | None = None,
        boxes: np.ndarray | None = None,
        box_labels: np.ndarray | None = None,
    ):
        """Project points, segment the image, and assign labels to the cloud.

        With no prompt, generate masks across the image. Text and image boxes
        guide SAM 3 toward selected objects. A saved label raster skips model
        loading. Raster value 0 is background. Points on
        background or outside the image receive -1. Pinhole views also return
        the camera intrinsics and pose.
        """
        points = _points(points)
        view = view or TopView()
        raster_transform = None
        if image_exists:
            image, raster_transform = _read_raster(image_path, with_transform=True)
            if raster_transform == rasterio.Affine.identity():
                raster_transform = None
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("raster image must have three RGB bands")
            image = image.astype(np.uint8)
            if isinstance(view, PinholeView):
                pose, K = view._camera(self.rotation, self.translation, self.intrinsics)
        elif isinstance(view, TopView):
            image = view.cloud_to_image(points, self.resolution)
            raster_transform = rasterio.transform.from_origin(
                points[:, 0].min(), points[:, 1].max(), self.resolution, self.resolution
            )
            _write_raster(image_path, image, transform=raster_transform)
        elif isinstance(view, PinholeView):
            image, K, pose = view.cloud_to_image(
                points,
                self.resolution,
                rotation=self.rotation,
                translation=self.translation,
                intrinsics=self.intrinsics,
                distance_threshold=self.distance_threshold,
                width=self.width,
                height=self.height,
            )
            _write_raster(image_path, image)
        else:
            raise TypeError("view must be TopView or PinholeView")
        if label_exists:
            labels = _read_raster(labels_path)
        else:
            labels = self._infer(image, prompt=prompt, boxes=boxes, box_labels=box_labels)
            _write_raster(labels_path, labels.astype(np.int32), transform=raster_transform)
        if labels.shape[:2] != image.shape[:2]:
            raise ValueError("label raster dimensions must match the RGB raster")
        if isinstance(view, TopView):
            ids = view.image_to_cloud(points, labels, self.resolution, transform=raster_transform)
            return ids, labels, image
        ids = view.image_to_cloud(
            points, labels, K, pose, distance_threshold=self.distance_threshold
        )
        return ids, labels, image, K, pose

    def write(
        self,
        points: np.ndarray,
        segment_ids: np.ndarray,
        non_ground: np.ndarray | None = None,
        ground: np.ndarray | None = None,
        save_path: str = "segmented.las",
        ground_path: str | None = None,
    ) -> None:
        """Write point labels to a LAS/LAZ extra dimension named `segment_id`."""
        points = _points(points)
        if Path(save_path).suffix.lower() not in {".las", ".laz"}:
            raise ValueError("output must be a .las or .laz file")
        ids = np.asarray(segment_ids, dtype=np.int32).reshape(-1)
        if ground_path is not None:
            saved = laspy.read(ground_path)
            ground = np.flatnonzero(np.asarray(saved.ground) == 1)
            non_ground = np.flatnonzero(np.asarray(saved.ground) == 0)
        if ground is not None:
            if non_ground is None:
                raise ValueError("non_ground indices are required with ground indices")
            ground = np.asarray(ground, dtype=np.int64)
            non_ground = np.asarray(non_ground, dtype=np.int64)
            if len(ids) != len(non_ground):
                raise ValueError("segment_ids must match the count of points above ground")
            all_ids = np.full(len(points), -1, dtype=np.int32)
            all_ids[non_ground] = ids
            ids = all_ids
        elif len(ids) != len(points):
            raise ValueError("segment_ids must match the point count")
        header = laspy.LasHeader(point_format=3 if points.shape[1] == 6 else 0, version="1.3")
        las = laspy.LasData(header)
        las.xyz = points[:, :3]
        if points.shape[1] == 6:
            colors = np.clip(np.rint(points[:, 3:6] * 65535), 0, 65535).astype(np.uint16)
            las.red, las.green, las.blue = colors.T
        las.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.int32))
        las.segment_id = ids
        _parent(save_path)
        las.write(save_path)
