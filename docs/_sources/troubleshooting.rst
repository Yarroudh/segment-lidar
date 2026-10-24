Troubleshooting
===============

**Model access error or HTTP 401/403.** Request access to
`facebook/sam3 <https://huggingface.co/facebook/sam3>`_ and use a token from
the approved account. A valid token from a different account is insufficient.
For existing labels, set both ``image_exists=True`` and ``label_exists=True``
to avoid model loading.

**SAM 3 import error.** Install ``segment-lidar[sam3]`` in the same Python
environment used to run the script. For ground filtering, install
``segment-lidar[csf]``.

**The output is mostly -1.** Inspect the RGB raster first. For a pinhole
camera, verify world-to-camera ``R`` and ``t``, positive camera Z, image
size, focal length, and ``distance_threshold``. For top view, confirm the
image and label raster share dimensions and that cached rasters cover the
points. Unlabeled raster pixels also yield -1.

**Pinhole image is empty or cropped.** Set the camera center and rotation so
the cloud is in front of the camera. Lower ``fx``/``fy`` for a wider field of
view, raise ``width``/``height``, or move the camera back. Check the numerical
example in :doc:`pinhole` before using survey coordinates.

**The pinhole labels reach points behind the visible surface.** The raster
uses the nearest point for color, but label transfer assigns a pixel's label
to every point that projects there. Inspect extra views or apply your own
visibility check if this affects the application.

**A cached raster gives different labels.** Reuse the same camera matrices
and depth threshold for pinhole views. For top views, prefer a saved raster
with its GeoTIFF spatial transform. Older unreferenced top view rasters need
the original XY origin and resolution.

**Interactive camera error.** ``PinholeView(interactive=True)`` is not
implemented in the 0.3 series. Supply explicit matrices as in :doc:`pinhole`.
