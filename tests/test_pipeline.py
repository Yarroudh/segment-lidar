import numpy as np
import pytest

from segment_lidar import SamLidar, TopView
from segment_lidar.samlidar import _labels_from_masks, _prompt_inputs


def test_cached_rasters_and_las_round_trip(tmp_path):
    points = np.array([[0, 0, 0, 1, 0, 0], [1, 0, 2, 0, 1, 0]], dtype=float)
    model = SamLidar(resolution=1)
    image_path = tmp_path / "image.tif"
    labels_path = tmp_path / "labels.tif"
    image = TopView().cloud_to_image(points, 1)
    from segment_lidar.samlidar import _write_raster

    _write_raster(str(image_path), image)
    _write_raster(str(labels_path), np.array([[4, 0]], dtype=np.int32))
    ids, labels, loaded = model.segment(
        points,
        image_path=str(image_path),
        labels_path=str(labels_path),
        image_exists=True,
        label_exists=True,
    )
    np.testing.assert_array_equal(ids, [4, -1])
    np.testing.assert_array_equal(labels, [[4, 0]])
    np.testing.assert_array_equal(loaded, image)
    output = tmp_path / "output.las"
    model.write(points, ids, save_path=str(output))
    import laspy

    las = laspy.read(output)
    np.testing.assert_array_equal(las.segment_id, ids)
    np.testing.assert_allclose(model.read(str(output))[:, 3:], points[:, 3:], atol=1 / 65535)


def test_ground_indices_keep_original_point_order(tmp_path):
    model = SamLidar()
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    output = tmp_path / "ground.las"
    model.write(points, [8, 9], non_ground=[0, 2], ground=[1], save_path=str(output))
    import laspy

    np.testing.assert_array_equal(laspy.read(output).segment_id, [8, -1, 9])
    assert model.read(str(output)).shape == (3, 3)


def test_cached_georeferenced_raster_projects_cloud_subset(tmp_path):
    import rasterio

    from segment_lidar.samlidar import _write_raster

    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    image_path, labels_path = tmp_path / "image.tif", tmp_path / "labels.tif"
    transform = rasterio.transform.from_origin(0, 0, 1, 1)
    _write_raster(str(image_path), TopView().cloud_to_image(points, 1), transform=transform)
    _write_raster(str(labels_path), np.array([[4, 5, 6]], dtype=np.int32))
    ids, _, _ = SamLidar(resolution=1).segment(
        points[1:],
        image_path=str(image_path),
        labels_path=str(labels_path),
        image_exists=True,
        label_exists=True,
    )
    np.testing.assert_array_equal(ids, [5, 6])


def test_mask_overlap_prefers_higher_score():
    masks = np.array([[[1, 1]], [[0, 1]]], dtype=bool)
    np.testing.assert_array_equal(_labels_from_masks(masks, (1, 2), [0.9, 0.2]), [[1, 1]])
    np.testing.assert_array_equal(_labels_from_masks(np.empty((0, 1, 2)), (1, 2)), [[0, 0]])


def test_invalid_file_type_raises_instead_of_exiting(tmp_path):
    with pytest.raises(ValueError):
        SamLidar().read(str(tmp_path / "bad.xyz"))


@pytest.mark.parametrize("text", [None, "tree"])
def test_sam3_requires_token_before_model_load(tmp_path, monkeypatch, text):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    model = SamLidar(resolution=1)
    model.text_prompt.text = text
    with pytest.raises(ValueError, match="Hugging Face token"):
        model.segment(
            np.array([[0, 0, 0]], dtype=float),
            image_path=str(tmp_path / "image.tif"),
            labels_path=str(tmp_path / "labels.tif"),
        )


def test_automatic_masks_are_default_and_keep_instance_ids(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    calls = {"loads": 0}
    transformers = ModuleType("transformers")

    class Generator:
        def __call__(self, image, **kwargs):
            calls["image_size"] = image.size
            calls["options"] = kwargs
            return {
                "masks": [
                    np.array([[True, True, False]]),
                    np.array([[False, True, True]]),
                ],
                "scores": np.array([0.8, 0.9]),
            }

    def pipeline(task, model, token, device):
        calls["loads"] += 1
        calls["load_args"] = (task, model, token, device)
        return Generator()

    transformers.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model = SamLidar(resolution=1, hf_token="approved token")
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    ids, labels, image = model.segment(
        points,
        image_path=str(tmp_path / "image.tif"),
        labels_path=str(tmp_path / "labels.tif"),
    )
    np.testing.assert_array_equal(ids, [1, 2, 2])
    np.testing.assert_array_equal(labels, [[1, 2, 2]])
    np.testing.assert_array_equal(model._infer(image), labels)
    assert calls["loads"] == 1
    assert calls["load_args"] == ("mask-generation", "facebook/sam3", "approved token", "cpu")
    assert calls["image_size"] == (3, 1)
    assert calls["options"]["points_per_crop"] == 32


@pytest.mark.parametrize("via_environment", [False, True])
def test_sam3_adapter_uses_hugging_face_token_and_box_prompt(
    tmp_path, monkeypatch, via_environment
):
    import sys
    from types import ModuleType

    import torch

    calls = {}
    transformers = ModuleType("transformers")

    class Batch(dict):
        def to(self, device):
            calls["batch_device"] = device
            return self

    class Sam3Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, model_id, token):
            calls["model_load"] = (model_id, token)
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            calls["model_inputs"] = inputs
            return object()

    class Sam3Processor:
        @classmethod
        def from_pretrained(cls, model_id, token):
            calls["processor_load"] = (model_id, token)
            return cls()

        def __call__(self, **kwargs):
            calls["processor_inputs"] = kwargs
            return Batch({"original_sizes": torch.tensor([[1, 2]])})

        def post_process_instance_segmentation(self, outputs, **kwargs):
            calls["postprocess"] = kwargs
            return (
                {
                    "masks": torch.tensor([[[True, False]]]),
                    "scores": torch.tensor([0.9]),
                },
            )

    transformers.Sam3Model = Sam3Model
    transformers.Sam3Processor = Sam3Processor
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    monkeypatch.setenv("HF_TOKEN", "environment token")
    token = "environment token" if via_environment else "constructor token"
    model = SamLidar(hf_token=None if via_environment else token, resolution=1)
    model.text_prompt.text = "tree"
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    ids, labels, _ = model.segment(
        points,
        image_path=str(tmp_path / "image.tif"),
        labels_path=str(tmp_path / "labels.tif"),
        boxes=np.array([[0, 0, 1, 1]]),
    )
    np.testing.assert_array_equal(ids, [1, -1])
    np.testing.assert_array_equal(labels, [[1, 0]])
    assert calls["model_load"] == ("facebook/sam3", token)
    assert calls["processor_load"] == ("facebook/sam3", token)
    assert calls["processor_inputs"]["text"] == "tree"
    assert calls["processor_inputs"]["images"].size == (2, 1)
    assert calls["processor_inputs"]["input_boxes"] == [[[0.0, 0.0, 1.0, 1.0]]]
    assert calls["processor_inputs"]["input_boxes_labels"] == [[1]]
    assert calls["postprocess"]["target_sizes"] == [[1, 2]]


@pytest.mark.parametrize(
    ("text", "boxes", "labels"),
    [
        (None, None, None),
        ("tree", [[2, 2, 1, 3]], None),
        ("tree", [[0, 0, 1, 1]], [0.5]),
        ("tree", None, [1]),
        (None, [[0, 0, 1, 1]], [0]),
    ],
)
def test_invalid_prompt_inputs(text, boxes, labels):
    with pytest.raises(ValueError):
        _prompt_inputs(text, boxes, labels)


def test_box_prompt_can_be_used_without_text():
    inputs = _prompt_inputs(None, [[0, 0, 4, 6]], None)
    assert inputs == {
        "input_boxes": [[[0.0, 0.0, 4.0, 6.0]]],
        "input_boxes_labels": [[1]],
    }


def test_box_must_fit_image_before_model_load():
    model = SamLidar()
    with pytest.raises(ValueError, match="fit inside"):
        model._infer(np.zeros((4, 4, 3), dtype=np.uint8), boxes=[[0, 0, 5, 2]])


def test_gated_model_error_explains_access(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    transformers = ModuleType("transformers")

    class Sam3Model:
        @classmethod
        def from_pretrained(cls, model_id, token):
            raise OSError("401 gated repo")

    transformers.Sam3Model = Sam3Model
    transformers.Sam3Processor = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model = SamLidar(hf_token="test token")
    with pytest.raises(PermissionError, match="request access") as error:
        model.segment(
            np.array([[0, 0, 0]], dtype=float),
            prompt="tree",
            image_path=str(tmp_path / "image.tif"),
            labels_path=str(tmp_path / "labels.tif"),
        )
    assert "test token" not in str(error.value)
