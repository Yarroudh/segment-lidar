---
title: 'Automatic Unsupervised LiDAR-Segmentation using Segment-Anything Model'
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

To be filled

# Statement of need

`segment-lidar` is a Python package for automatic unsupervised segmentation of aerial LiDAR data. It proposes an image-based approach for segmenting aerial point clouds using `Segment-Anthing Model (SAM)` package from [Meta AI](https://github.com/facebookresearch). (to complete by justifying why do we need unsupervised segmentation and how SAM is powerful)

The API for `segment-lidar` provides functions and classes to define the segmentation model and its paramaters, and also to handle transformation of 3D point clouds into images. `segment-lidar` also relies on other packages that make use of `SAM` for instance segmentation of images. This includes the `segment-geospatial` package from [Open Geospatial Solutions](https://github.com/opengeos) for segmenting geospatial data and the `Grounded-SAM` package from [The International Digital Economy Academy Research (IDEA-Research)](https://github.com/IDEA-Research) that combines `SAM` with `GroundingDINO` to detect and segment anything with text prompts. The `GroundingDINO` package was introduced by IDEA-Research as an implementation of the paper "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection".

For optimization purposes, `segment-lidar` enables using `Fast Segment Anything Model (FastSAM)` as an alternative to `SAM` original API. The `FastSAM` is a Convolutional Neural Network (CNN) `SAM` that was trained using only 2% of the `SA-1B` dataset published by `SAM` authors. It achieves comparable performance at 50x higher run-time speed.


# Overview of the method

The main idea behind using `SAM` is to make use of unsupervised image-segmentation to automatically find and separate different objects in LiDAR point clouds. The process can be divided into three main steps:

## Step 1: Projection of the 3D point cloud into a two-dimensional image

This projection can be based on various views, including cubic views (top, bottom, left, right, front, back) and panoramic views (360°).

1. Cubic Projection:

In a cubic projection, each face of the cube represents a different view. The 3D coordinates (X, Y, Z) are projected onto the 2D coordinates (u, v) on the image plane.

As shown in Figure 1, for the top face:

`u` represents the horizontal axis in the image.
`v` represents the vertical axis in the image.

The projection equations for the top face are then:

$$u = X$$
$$v = Y$$

Similarly, these equations are adapted for other faces of the cube, adjusting the coordinates based on the view.

2. Panoramic View:

Panoramic projections capture a full 360-degree view around a point. A common panoramic projection is the equirectangular projection. In this projection, the 3D spherical coordinates ($\Theta$, $\Phi$, $\rho$) are mapped onto 2D coordinates (u, v) on the image plane.

$\Theta$ represents the azimuthal angle (longitude).
$\Phi$ represents the polar angle (latitude).
$\rho$ represents the radial distance from the center.

The equirectangular projection equations are:

$$u = (\Theta - \Theta_{min})\times \frac{w}{(\Theta_{max} - \Theta_{min})}$$
$$v = (\Phi - \Phi_{min})\times \frac{h}{(\Phi_{max} - \Phi_{min})}$$

$\Theta_{min}$ and $\Theta_{max}$ are the minimum and maximum azimuthal angles, and $\Phi_{min}$ and $\Phi_{min}$ are the minimum and maximum polar angles. `w` and `h` are the dimensions of the image.

## Step 2: Inference on the generated image



## Step 3: Reprojection of results on the 3D point cloud

In the final step of our methodology, we seamlessly reprojet the instance segmentation results onto the original point cloud. This associates each point in the cloud with its corresponding segment label obtained from the 2D image segmentation. Mathematically, this process involves identifying the 2D image coordinates for each point in the point cloud, which can be achieved through reverse projection of the cubic or panoramic projection. Once the corresponding 2D image coordinates are identified, we assign the segment label from the segmentation map to the corresponding point in the cloud.

# Use of the package


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References