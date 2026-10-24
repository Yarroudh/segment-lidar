Top view tutorial
=================

This tutorial covers an aerial LiDAR workflow. See :doc:`pinhole` for a
perspective camera with explicit calibration.

Input and output
----------------

``read`` accepts LAS, LAZ, and NPY. The point array has shape ``(N, 3)`` for
XYZ or ``(N, 6)`` for XYZRGB. LAS color is normalized to 0–1 when present.
NPY arrays must already have one of these shapes. A LAS classification filter
is available with ``model.read("pointcloud.laz", classification=6)``.

For sample data, the `Sao Paulo M3DC archive
<https://registry.opendata.aws/pmsp-lidar/>`_ contains RGB aerial LiDAR
tiles. Check its license and attribution requirements before use. The
repository's `animation notes
<https://github.com/Yarroudh/segment-lidar/blob/main/assets/README.md>`_
identify a specific tile and prompt workflow.

Automatic instance masks
------------------------

With no prompt or boxes, ``segment`` runs SAM 3 automatic mask generation.
The top view maps XY to image pixels at ``resolution`` world units per pixel.
The highest Z point supplies each pixel's color.

.. code-block:: python

   from segment_lidar import SamLidar, TopView

   model = SamLidar(resolution=0.25)
   points = model.read("pointcloud.las")
   ids, labels, image = model.segment(
       points,
       view=TopView(),
       image_path="results/raster.tif",
       labels_path="results/labels.tif",
   )
   model.write(points, ids, save_path="results/segmented.las")

``ids`` contains one label per input point. ``labels`` is a 2D integer image;
``image`` is a 3 band RGB image. ``segment`` writes the image and label
rasters to the chosen paths. The output LAS/LAZ has a signed 32 bit
``segment_id`` extra dimension. Open it in CloudCompare or another LAS viewer
that displays extra dimensions.

For automatic generation, tune ``model.automatic_masks.points_per_crop`` and
``points_per_batch`` (both positive integers) and ``pred_iou_thresh`` and
``stability_score_thresh`` (both 0–1). The defaults are 32, 16, 0.65, and
0.7 respectively. More sampled points can take longer.

Text and box prompts
--------------------

A text prompt finds visible instances of a concept:

.. code-block:: python

   ids, labels, image = model.segment(
       points, view=TopView(), prompt="building",
       image_path="results/building_rgb.tif",
       labels_path="results/building_labels.tif",
   )

``model.text_prompt.text`` can hold a default prompt. The per-call ``prompt``
argument overrides it. ``score_threshold`` and ``mask_threshold`` default to
0.5 and must be in 0–1:

.. code-block:: python

   model.text_prompt.score_threshold = 0.6
   model.text_prompt.mask_threshold = 0.5

Boxes use image pixel coordinates ``[left, top, right, bottom]``. Positive
labels are 1 and negative labels are 0. Every box must have positive area and
fit inside the projected image. A positive box can be used without text; a
negative-only box set requires text.

.. code-block:: python

   import numpy as np

   ids, labels, image = model.segment(
       points,
       prompt="tree",
       boxes=np.array([[80, 40, 180, 220], [210, 30, 300, 160]]),
       box_labels=np.array([1, 0]),
       image_path="results/tree_rgb.tif",
       labels_path="results/tree_labels.tif",
   )

Inspect the projected image before choosing box coordinates. A useful
token-free step is to call ``TopView().cloud_to_image(points, 0.25)`` and
display or save the returned array. Box coordinates refer to *that* image,
not to world XY coordinates.

Ground filtering
----------------

Install ``segment-lidar[csf]`` first. Filtering can keep ground points from
obscuring aerial objects, and the writer can restore their original order:

.. code-block:: python

   from segment_lidar import SamLidar, TopView

   model = SamLidar(resolution=0.25)
   points = model.read("pointcloud.las")
   cloud, non_ground, ground = model.csf(
       points, class_threshold=0.5, csf_path="results/ground_flags.las"
   )
   ids, labels, image = model.segment(
       cloud, view=TopView(),
       image_path="results/filtered_rgb.tif",
       labels_path="results/filtered_labels.tif",
   )
   model.write(
       points, ids, non_ground=non_ground, ground=ground,
       save_path="results/segmented.las",
   )

``non_ground`` and ``ground`` are indices into the original ``points`` array.
Ground points receive ``-1`` in the output. To reuse saved flags with the
same point order, call
``model.csf(points, csf_path="results/ground_flags.las", exists=True)``.

Saved raster and label workflows
--------------------------------

You can transfer edited or previously computed labels without SAM 3:

.. code-block:: python

   from segment_lidar import SamLidar, TopView

   model = SamLidar(resolution=0.25)
   points = model.read("pointcloud.las")
   ids, labels, image = model.segment(
       points,
       view=TopView(),
       image_path="results/raster.tif",
       labels_path="results/labels.tif",
       image_exists=True,
       label_exists=True,
   )
   model.write(points, ids, save_path="results/from_saved_labels.las")

The RGB image must have three bands and the integer label raster one band,
with matching width and height. Label 0 is background. New top view rasters
include a spatial transform, so a subset of the original point cloud can
reuse them. Older rasters without that transform need the same XY origin and
``resolution`` as the cloud used to create them. See :doc:`pinhole` for the
camera parameters required to reuse a perspective raster.
