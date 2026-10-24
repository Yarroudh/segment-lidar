segment-lidar documentation
===========================

**segment-lidar** projects a LiDAR point cloud to a 2D image, finds object
instances with Meta's SAM 3, and assigns the image labels back to the original
3D points. It supports top and pinhole views, automatic masks, text and box
prompts, optional ground filtering, cached rasters, and LAS/LAZ output.

The current API is the **0.3 series**. SAM 3 replaces the SAM 1 checkpoint
workflow described in documentation published before 0.3. See :doc:`migration`
when updating an older script.

Start here
----------

1. :doc:`installation` — install the package, request SAM 3 model access, and
   set up a token.
2. :doc:`tutorial` — complete top view workflows, prompts, filtering, and saved
   rasters.
3. :doc:`pinhole` — camera geometry and worked projection and segmentation
   examples.
4. :doc:`module` — constructor, method, and view reference.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   tutorial
   pinhole
   module
   migration
   troubleshooting
   citation
   license

How the pipeline works
----------------------

``read`` loads XYZ or XYZRGB points. An optional ``csf`` call removes ground
points. ``segment`` projects the remaining points with :class:`~segment_lidar.view.TopView`
or :class:`~segment_lidar.view.PinholeView`, runs SAM 3 or reads saved labels,
and transfers the labels to points. ``write`` stores a signed 32 bit
``segment_id`` extra dimension in LAS/LAZ.

The image label **0** is background. Point label **-1** means background or
outside the image; positive values identify instances. When several points
project to one pixel, they can receive the same instance label.

.. code-block:: python

   from segment_lidar import SamLidar, TopView

   model = SamLidar(resolution=0.25)
   points = model.read("pointcloud.las")
   ids, labels, image = model.segment(points, view=TopView(), prompt="tree")
   model.write(points, ids, save_path="segmented.las")

The repository includes a `sample animation and data provenance
<https://github.com/Yarroudh/segment-lidar/tree/main/assets>`_ and a
`GitHub issue tracker <https://github.com/Yarroudh/segment-lidar/issues>`_.
