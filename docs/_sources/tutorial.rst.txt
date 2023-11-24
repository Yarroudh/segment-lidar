Basic tutorial
==============


In this tutorial, we will learn how to use the `segment_lidar` module for
automatic unsupervised instance segmentation of LiDAR data.

Prerequisites
-------------

Before getting started, make sure you have the following:

1. Python installed on your system.
2. The `segment_lidar` module installed. You can install it using pip:

.. code-block:: bash

    pip install segment-lidar

For more information on how to install the module, please refer to the :doc:`installation` page.


Sample data
------------

For testing purposes, you can download a sample data here: `pointcloud.las <https://drive.google.com/file/d/16EF2aRSvo8u0pXvwtaQ6sjhP5h0sWw3o/view?usp=sharing>`__.
This data was retrieved from **AHN-4**. For more data, please visit `AHN-Viewer <https://ahn.arcgisonline.nl/ahnviewer>`__.


Model checkpoints
-----------------

Click the links below to download the checkpoint for the corresponding Segment-Anything model (SAM) type.

- `default` or `vit_h`: `ViT-H SAM model <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth>`__.
- `vit_l`: `ViT-L SAM model <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth>`__.
- `vit_b`: `ViT-B SAM model <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>`__.


Basic usage
-----------

1. Import the necessary modules:

.. code-block:: python

    from segment_lidar import samlidar, view

2. Define the viewpoint using the **view** module. You can choose between the following:

- `TopView`: Top view of the point cloud.
- `PinholeView`: Pinhole camera view of the point cloud, defined by its intrinsic and extrinsic parameters.

For example, to define a top view, you can do the following:

.. code-block:: python

    viewpoint = view.TopView()

The pinhole view can be defined either by providing the intrinsic and extrinsic parameters:

.. code-block:: python

    viewpoint = view.PinholeView(intrinsic=K, rotation=R, translation=T)

or by using the interactive mode:

.. code-block:: python

    viewpoint = view.PinholeView(interactive=True)

3. Create an instance of the SamLidar class and specify the path to the checkpoint
file **ckpt_path** when instantiating the class:

.. code-block:: python

    model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")

4. Read the point cloud data from a **.las/.laz** file using the read method of the
SamLidar instance. Provide the path to the point cloud file `pointcloud.las` as an argument:

.. code-block:: python

    points = model.read("pointcloud.las")

5. Apply the Cloth Simulation Filter (CSF) algorithm for ground filtering using the **csf**
method of the SamLidar instance. This method returns the filtered point cloud `cloud`,
the non-ground `non_ground` and the ground `ground` indices:

.. code-block:: python

    cloud, non_ground, ground = model.csf(points, class_threshold=0.1)

6. Perform segmentation using the **segment** method of the SamLidar instance. This
method requires the filtered point cloud `cloud` as input, and you can optionally provide
an image path `image_path` and labels path `labels_path` to save the segmentation
results as an image and labels, respectively. The segment method returns the segmentation
labels `labels`:

.. code-block:: python

    labels, *_ = model.segment(points=cloud, image_path="raster.tif", labels_path="labeled.tif")

7. Save results to **.las/.laz** file using the **write** method of the SamLidar instance:

.. code-block:: python

    model.write(points=points, non_ground=non_ground, ground=ground, segment_ids=labels, save_path="segmented.las")

Now, the entire code should look like this:

.. code-block:: python

    from segment_lidar import samlidar, view

    # Define viewpoint
    view = view.TopView()

    # Create SamLidar instance
    model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")

    # Load point cloud
    points = model.read("pointcloud.las")

    # Apply CSF
    cloud, non_ground, ground = model.csf(points)

    # Segment the point cloud
    labels, *_ = instance.segment(points=cloud, image_path="raster.tif", labels_path="labeled.tif")

    # Save results
    model.write(points=points, non_ground=non_ground, ground=ground, segment_ids=labels, save_path="segmented.las")

8. The resulted point cloud contains a new scalar field called `segment_id`.
For visualization and further processing, we recommand using `CloudCompare <https://www.danielgm.net/cc>`__.

The following figure shows the results of the segmentation on the sample data form AHN-4:

.. image:: _static/results.gif
   :width: 100%
   :align: center
   :alt: Segmented point cloud

Interactive mode
----------------

The interactive mode allows you to interactively define the viewpoint using GUI.

.. code-block:: python

    viewpoint = view.PinholeView(interactive=True)

.. image:: _static/interactive.png
   :width: 100%
   :align: center
   :alt: Interactive mode

You can rotate, move and zoom the camera using the mouse (please refer to `Open3D documentation <http://www.open3d.org/docs/release/tutorial/visualization/visualization.html>`_ for more details).

Once you are done, press **p** to save the image and the camera parameters, than **esc** to quit the interactive mode.

Example:

.. code-block:: python

    import os
    from segment_lidar import samlidar, view

    view = view.PinholeView(interactive=True)

    model = samlidar.SamLidar(ckpt_path='sam_vit_h_4b8939.pth',
                            device='cuda:0',
                            algorithm='segment-anything')

    model.mask.min_mask_region_area = 200
    model.mask.points_per_side = 5
    model.mask.pred_iou_thresh = 0.60
    model.mask.stability_score_thresh = 0.85

    points = model.read('laundry.las')

    os.makedirs("results/", exist_ok=True)

    labels, *_ = model.segment(points=points,
                            view=view,
                            image_path="results/raster.tif",
                            labels_path="results/labeled.tif")

    model.write(points=points, segment_ids=labels, save_path="results/segmented.las")


Configuration
-------------

The `segment_lidar` module provides a set of parameters that can be used to configure
the segmentation process. These parameters are passed to the `SamLidar` class as arguments
when instantiating the class. The following table shows the parameters and their default values:

+------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter        | Default value        | Description                                                                                                                                                                                                                                                                                                                                         |
+==================+======================+=====================================================================================================================================================================================================================================================================================================================================================+
| algorithm        | "segment-geospatial" | Algorithm to use for segmentation. Possible values are: "segment-geospatial", "segment-anything".                                                                                                                                                                                                                                                   |
+------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ckpt_path        | None                 | Path to the checkpoint file.                                                                                                                                                                                                                                                                                                                        |
+------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| device           | "cuda:0"             | Device to use for inference.                                                                                                                                                                                                                                                                                                                        |
+------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| model_type       | "vit_h"              | Type of the SAM model. Possible values are: "vit_h", "vit_l", "vit_b".                                                                                                                                                                                                                                                                              |
+------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| resolution       | 0.25                 | The resolution value of the created image raster.                                                                                                                                                                                                                                                                                                   |
+------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| sam_kwargs       | False                | Whether to use the SAM kwargs when using "segment-geospatial" as algorithm                                                                                                                                                                                                                                                                          |
+------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Here is an example of how to configure the parameters:

.. code-block:: python

    model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth",
                              algorithm="segment-geo-spatial",
                              model_type="vit_h",
                              resolution=0.5,
                              sam_kwargs=True)

Additionally, the parameters of `segment-anything` can be configured as follows:

.. code-block:: python

    model.mask.crop_n_layers = 1
    model.mask.crop_n_points_downscale_factor = 2
    model.mask.min_mask_region_area = 500
    model.mask.points_per_side = 10
    model.mask.pred_iou_thresh = 0.90
    model.mask.stability_score_thresh = 0.92

Please, refer to the `segment-anything <https://github.com/facebookresearch/segment-anything>`__ repository for more details about these parameters.
See the complete arguments list of the `SamLidar` class :doc:`here <module>`.