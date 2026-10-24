API reference
=============

Public imports are ``SamLidar``, ``TopView``, and ``PinholeView`` from
``segment_lidar``. All three can also be imported from their defining
modules. The methods below describe the 0.3 series.

.. automodule:: segment_lidar.samlidar

.. automodule:: segment_lidar.view

SamLidar configuration
----------------------

.. autoclass:: segment_lidar.samlidar.SamLidar

Constructor parameters
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Parameter
     - Default
     - Meaning
   * - ``resolution``
     - ``0.25``
     - Top view world units per pixel. Must be positive and finite.
   * - ``height``, ``width``
     - ``512``, ``512``
     - Pinhole output image dimensions when creating a new image.
   * - ``distance_threshold``
     - ``None``
     - Optional maximum positive pinhole camera depth.
   * - ``device``
     - ``None``
     - PyTorch device; selects CUDA when available, else CPU.
   * - ``intrinsics``
     - ``None``
     - Pinhole 3 by 3 camera matrix.
   * - ``rotation``, ``translation``
     - ``None``
     - World-to-camera 3 by 3 rotation and three element translation.
   * - ``hf_token``
     - ``None``
     - Token for model download; falls back to ``HF_TOKEN``.
   * - ``model_id``
     - ``facebook/sam3``
     - Hugging Face model identifier.

Inference settings
~~~~~~~~~~~~~~~~~~

Each ``SamLidar`` instance exposes ``model.text_prompt`` and
``model.automatic_masks``. Their settings apply to the appropriate SAM 3
mode; model loading is deferred until inference. Switching between automatic
and prompted modes loads the corresponding model path.

.. list-table::
   :header-rows: 1
   :widths: 37 17 46

   * - Setting
     - Default
     - Meaning
   * - ``text_prompt.text``
     - ``None``
     - Default text, overridden by ``segment(prompt=...)``.
   * - ``text_prompt.score_threshold``
     - ``0.5``
     - Minimum prompted instance score, 0–1.
   * - ``text_prompt.mask_threshold``
     - ``0.5``
     - Prompted mask threshold, 0–1.
   * - ``automatic_masks.points_per_crop``
     - ``32``
     - Automatic mask sample grid density; positive integer.
   * - ``automatic_masks.points_per_batch``
     - ``16``
     - Automatic mask sampling batch size; positive integer.
   * - ``automatic_masks.pred_iou_thresh``
     - ``0.65``
     - Automatic predicted IoU threshold, 0–1.
   * - ``automatic_masks.stability_score_thresh``
     - ``0.7``
     - Automatic mask stability threshold, 0–1.

I/O and processing methods
--------------------------

.. automethod:: segment_lidar.samlidar.SamLidar.read

``classification`` filters LAS/LAZ by classification code. The return is an
``(N, 3)`` or ``(N, 6)`` NumPy array. RGB values from LAS/LAZ are normalized
to 0–1. NPY input does not support classification filtering.

.. automethod:: segment_lidar.samlidar.SamLidar.csf

Returns ``(cloud, non_ground, ground)``. ``cloud`` is the subset above
ground, while both index arrays refer to the original input. ``csf_path``
can save a LAS with a ``ground`` flag. With ``exists=True``, it reads those
saved flags and requires a matching point count.

.. automethod:: segment_lidar.samlidar.SamLidar.segment

``view`` defaults to ``TopView()``. If ``image_exists`` is false, a new RGB
GeoTIFF is written to ``image_path``. If ``label_exists`` is false, SAM 3
inference writes an integer label GeoTIFF to ``labels_path``. Both flags true
reuse the files and skip model inference. ``prompt`` is text; ``boxes`` has
shape ``(N, 4)`` in pixel coordinates; ``box_labels`` has one 0 or 1 per box.
With no prompt or boxes, automatic mask generation is used.

Top view returns ``(segment_ids, labels, image)``. Pinhole returns
``(segment_ids, labels, image, intrinsics, pose)``. ``segment_ids`` has one
element per input point. ``labels`` has shape ``(H, W)`` and ``image`` has
shape ``(H, W, 3)``. The returned pose is world-to-camera. Label 0 in the
raster means background; point label -1 means background or out of view.

.. automethod:: segment_lidar.samlidar.SamLidar.write

``segment_ids`` must match ``points`` in length, except when ``ground`` and
``non_ground`` are supplied. In that case, IDs correspond to ``non_ground``;
ground points receive -1 and original order is preserved. ``ground_path``
can read saved ground flags. The output must end in ``.las`` or ``.laz`` and
contains a signed 32 bit ``segment_id`` extra dimension.

Projection views
----------------

.. autoclass:: segment_lidar.view.TopView

.. automethod:: segment_lidar.view.TopView.cloud_to_image

.. automethod:: segment_lidar.view.TopView.image_to_cloud

Top view uses minimum X and maximum Y as its raster origin and chooses the
highest Z point's color at each pixel. New top view images have a GeoTIFF
transform. Passing that transform to ``image_to_cloud`` lets a subset of the
original cloud reuse the raster.

.. autoclass:: segment_lidar.view.PinholeView

.. automethod:: segment_lidar.view.PinholeView.cloud_to_image

.. automethod:: segment_lidar.view.PinholeView.image_to_cloud

The pinhole ``cloud_to_image`` method returns ``(image, K, pose)`` and uses
the nearest positive-depth point for each image pixel. ``image_to_cloud``
projects every input point back through the same camera, so points hidden
behind a visible point may receive its label. See :doc:`pinhole` for camera
conventions and runnable examples. Interactive capture is currently not
implemented.
