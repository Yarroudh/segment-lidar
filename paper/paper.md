---
title: 'SamLidar: Automatic Unsupervised LiDAR-Segmentation using Segment-Anything Model (SAM)'
tags:
  - Python
  - Segment-Anything Model
  - LiDAR
  - Machine Learning
  - Unsupervised segmentation
authors:
  - name: Anass Yarroudh
    orcid: 0000-0003-1387-8288
    corresponding: true
    affiliation: 1
  - name: Abderrazzaq Kharroubi
    orcid: 0000-0001-7712-6208
    affiliation: 1
  - name: Roland Billen
    orcid: 0000-0002-3101-8057
    affiliation: 1
affiliations:
 - name: Geomatics Unit, University of Liège, Allée du six Août 19, 4000 Liège, Belgium
   index: 1
date: 31 August 2023
bibliography: paper.bib
---

# Summary

`SamLidar` is a Python package for automatic unsupervised segmentation of aerial LiDAR data. It proposes an image-based approach for segmenting aerial point clouds using `Segment-Anthing Model (SAM)` package from [Meta AI](https://github.com/facebookresearch). (to complete by justifying why do we need unsupervised segmentation and how SAM is powerful)

The API for `segment-lidar` provides functions and classes to define the segmentation model and its parameters, and also to handle transformation of 3D point clouds into images. `segment-lidar` also relies on other packages that make use of `SAM` for instance segmentation of images. This includes the `segment-geospatial` package from [Open Geospatial Solutions](https://github.com/opengeos) for segmenting geospatial data and the `Grounded-SAM` package from [The International Digital Economy Academy Research (IDEA-Research)](https://github.com/IDEA-Research) that combines `SAM` with `GroundingDINO` to detect and segment anything with text prompts. The `GroundingDINO` package was introduced by IDEA-Research as an implementation of the paper "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection".

For optimization purposes, `segment-lidar` enables using `Fast Segment Anything Model (FastSAM)` as an alternative to `SAM` original API. The `FastSAM` is a Convolutional Neural Network (CNN) `SAM` that was trained using only 2% of the `SA-1B` dataset published by `SAM` authors. It achieves comparable performance at 50x higher run-time speed.

# Statement of need

Image-based segmentation is generally considered easier than 3D point cloud segmentation.

# Overview of the method

The main aim of using `SAM` is to automatically identify and separate different instances in 3D LiDAR data through automated image segmentation. The process can be divided into four main steps:

## Step 1: Ground filtering using Cloth Simulation Filter [Optional]

The ground filtering is optional but preferred for aerial LiDAR data with top viewpoint. It serves two primary purposes that significantly enhance the accuracy and reliability of object detection and segmentation. First, ground filtering helps improve the detection of objects within the image by eliminating the interference of ground points. This is especially vital for identifying objects such as buildings, vehicles, and infrastructure, as it allows for a clearer focus on target objects against a clutter-free background. Second, ground filtering prevents the projection of segmentation results onto ground points, especially for tall structures like trees and poles.

Our package uses the Cloth Simulation Filter (CSF) to separate the ground points from non-ground points (Figure 1). The algorithm was proposed by @zhang:2016 as an implementation of the Cloth Simulation algorithm used in 3D computer graphics to simulate fabric attached to an object.

## Step 2: Projection of the 3D point cloud into a two-dimensional image

This projection can be based on various views, including cubic views (top, bottom, left, right, front, back) and panoramic views (360°).

1. **Cubic Projection**:

In a cubic projection, each face of the cube represents a different view (\autoref{fig:cubicview}). The 3D coordinates (X, Y, Z) are projected onto the 2D coordinates (u, v) on the image plane.

As shown in Figure 1, for the top face:

`u` represents the horizontal axis in the image.
`v` represents the vertical axis in the image.

The projection equations for the top face are then:

$$u = X$$
$$v = Y$$

Similarly, these equations are adapted for other faces of the cube, adjusting the coordinates based on the view.

![Different viewpoints of the Cubic View.\label{fig:cubicview}](figures/cubicview.png)

2. **Panoramic View**:

Panoramic projections capture a full 360-degree view around a point. A common panoramic projection is the equirectangular projection. In this projection, the 3D spherical coordinates ($\Theta$, $\Phi$, $\rho$) are mapped onto 2D coordinates (u, v) on the image plane.

$\Theta$ represents the azimuthal angle (longitude).
$\Phi$ represents the polar angle (latitude).
$\rho$ represents the radial distance from the center.

The equirectangular projection equations are:

$$u = w\times \frac{\Theta - \Theta_{min}}{\Theta_{max} - \Theta_{min}}$$
$$v = h\times \frac{\Phi - \Phi_{min}}{\Phi_{max} - \Phi_{min}}$$

$\Theta_{min}$ and $\Theta_{max}$ are the minimum and maximum azimuthal angles, and $\Phi_{min}$ and $\Phi_{min}$ are the minimum and maximum polar angles. `w` and `h` are the dimensions of the image.

## Step 3: Inference on the generated image

The Segment-Anything Model (SAM) was used to generate masks for all objects in the resulting image [@kirillov:2023]. Additionally, segment-geospatial [@wu:2023] is implemented to leverage SAM for geospatial analysis by enabling users to achieve results with minimal parameters tuning. The results for sample data are illustrated in \autoref{fig:inference}.

![Inference results using SAM and SamGeo.\label{fig:inference}](figures/inference.png)

## Step 4: Reprojection of results on the 3D point cloud

In the final step of our methodology, we seamlessly reproject the instance segmentation results onto the original point cloud (\autoref{fig:results}). This associates each point in the cloud with its corresponding segment label obtained from the 2D image segmentation. Mathematically, this process involves identifying the 2D image coordinates for each point in the point cloud, which can be achieved through reverse projection of the cubic or panoramic projection. Once the corresponding 2D image coordinates are identified, we assign the segment label from the segmentation map to the corresponding point in the cloud.

![Segmented point cloud.\label{fig:results}](figures/results.png)

# Use of the package

## Installation

The package is available as a Python library and can be installed directly from [PyPI](https://pypi.org/project/segment-lidar/). We recommend using `Python>=3.9`. It is also required to install [PyTorch](https://pytorch.org/) before installing `segment-lidar`.

The easiest way to install the package in a Python environment is tu run the following command:

```bash
pip install segment-lidar
```

It is also possible to install it from source by running these commands:

```bash
git clone https://github.com/Yarroudh/segment-lidar
cd segment-lidar
python setup.py install
```

## Basic usage

1. Import the necessary modules:

```python
from segment_lidar import samlidar, view
```

2. Create an instance of the SamLidar class and specify the path to the checkpoint file **ckpt_path** when instantiating the class:

```python
model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")
```

3. Define the view by choosing either a **CubicView** or **PanromaicView**. The Cubic View offers a default view point, which is the **top** view, among its six available viewpoints:

```python
view = view.CubicView(viewpoint='top')
```

4. Read the point cloud data from a **.las/.laz** file using the read method of the SamLidar instance. Provide the path to the point cloud file pointcloud.las as an argument:

```python
points = model.read("pointcloud.las")
```

5. Apply the Cloth Simulation Filter (CSF) algorithm for ground filtering using the **csf** method of the SamLidar instance. This method returns the filtered point cloud cloud, the non-ground non_ground and the ground ground indices:

```python
cloud, non_ground, ground = model.csf(points)
```

6. Perform segmentation using the **segment** method of the SamLidar instance. This method requires the filtered (or not) point cloud as input, and you can optionally provide an **image_path** and **labels_path// to save the projection and segmentation results:

```python
labels, *_ = model.segment(points=cloud, image_path="raster.tif", labels_path="labeled.tif")
```

7. Save results to **.las/.laz** file using the **write** method of the SamLidar instance:

```python
model.write(points=points, non_ground=non_ground, ground=ground, segment_ids=labels, save_path="segmented.las")
```

As shown in Figue, the resulted point cloud contains a new scalar field called **segment_id**.

It's also possible to use **segment-lidar** without ground filtering as follow:

```python
from segment_lidar import samlidar

model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")
points = model.read("pointcloud.las")
labels, *_ = model.segment(points=points, image_path="raster.tif", labels_path="labeled.tif")
model.write(points=points, segment_ids=labels, save_path="segmented.las")
```

# References