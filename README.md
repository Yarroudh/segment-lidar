<img src="https://user-images.githubusercontent.com/72500344/210864557-4078754f-86c1-4e7c-b291-73223bdf4e4d.png" alt="logo" width="200"/>

# segment-lidar
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Yarroudh/ZRect3D/blob/main/LICENSE)
[![Geomatics Unit of ULiege - Development](https://img.shields.io/badge/Geomatics_Unit_of_ULiege-Development-2ea44f)](http://geomatics.ulg.ac.be/)

*Python package for segmenting LiDAR data using Segment-Anything Model (SAM) from Meta Research.*

This package is specifically designed for **unsupervised instance segmentation** of **aerial LiDAR data**. It brings together the power of the **Segment-Anything Model (SAM)** developed by [Meta Research](https://github.com/facebookresearch) and the **segment-geospatial** package from [Open Geospatial Solutions](https://github.com/opengeos). Whether you're a researcher, developer, or a geospatial enthusiast, segment-lidar opens up new possibilities for automatic processing of aerial LiDAR data and enables further applications. We encourage you to explore our code, contribute to its development and leverage its capabilities for your segmentation tasks.

![results](https://github.com/Yarroudh/segment-lidar/assets/72500344/089a603b-697e-4483-af1e-3687a79adcc1)

## Installation

We recommand using `Python 3.9`. First, you need to install `PyTorch`. Please follow the instructions [here](https://pytorch.org/).

Then, you can easily install `segment-lidar` from [PyPI](https://pypi.org/project/segment-lidar/):

```bash
pip install segment-lidar
```

Or, you can install it from source by running the following commands:

```bash
git clone https://github.com/Yarroudh/segment-lidar
cd segment-lidar
python setup.py install
```

Please, note that the actual version is a pre-release and it's under tests. If you find any issues or bugs, please report them in [issues](https://github.com/Yarroudh/segment-lidar/issues) section. The second version will have more advanced features and fonctionalities.

## Documentation

If you are using `segment-lidar`, we highly recommend that you take the time to read the [documentation](segment-lidar.readthedocs.io). The documentation is an essential resource that will help you understand the features and functionality of our software, as well as provide guidance on how to use it effectively.

## Basic tutorial

A basic tutorial is available [here]().
You can also refer to API for more information about different parameters.

```python
from segment_lidar import samlidar

model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")
points = model.read("pointcloud.las")
cloud, non_ground, ground = model.csf(points)
labels, *_ = model.segment(points=cloud, image_path="raster.tif", labels_path="labeled.tif")
model.write(points=points, non_ground=non_ground, ground=ground, segment_ids=labels, save_path="segmented.las")
```

## Sample data

For testing purposes, you can download a sample here: [pointcloud.las](https://drive.google.com/file/d/16EF2aRSvo8u0pXvwtaQ6sjhP5h0sWw3o/view?usp=sharing).

This data was retrieved from **AHN-4**. For more data, please visit [AHN-Viewer](https://ahn.arcgisonline.nl/ahnviewer/).

## Related repositories

We would like to express our acknowledgments to the creators of:

- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [segment-geospatial](https://github.com/opengeos/segment-geospatial)

Please, visit these repositories for more information about image raster automatic segmentation using SAM from Meta AI.

## License

This software is under the BSD 3-Clause "New" or "Revised" license which is a permissive license that allows you almost unlimited freedom with the software so long as you include the BSD copyright and license notice in it. Please refer to the [LICENSE](https://github.com/Yarroudh/segment-lidar/blob/main/LICENSE) file for more detailed information.

## Citation

The use of open-source software repositories has become increasingly prevalent in scientific research. If you use this repository for your research, please make sure to cite it appropriately in your work. The recommended citation format for this repository is provided in the accompanying [BibTeX citation](https://github.com/Yarroudh/segment-lidar/blob/main/CITATION.bib). Additionally, please make sure to comply with any licensing terms and conditions associated with the use of this repository.

```bib
@misc{yarroudh:2023:samlidar,
  author = {Yarroudh, Anass},
  title = {LiDAR Automatic Unsupervised Segmentation using Segment-Anything Model (SAM) from Meta AI},
  year = {2023},
  howpublished = {GitHub Repository},
  url = {https://github.com/Yarroudh/segment-lidar}
}
```

Yarroudh, A. (2023). *LiDAR Automatic Unsupervised Segmentation using Segment-Anything Model (SAM) from Meta AI* [GitHub repository]. Retrieved from https://github.com/Yarroudh/segment-lidar

## Author

This software was developped by [Anass Yarroudh](https://www.linkedin.com/in/anass-yarroudh/), a Research Engineer in the [Geomatics Unit of the University of Liege](http://geomatics.ulg.ac.be/fr/home.php).
For more detailed information please contact us via <ayarroudh@uliege.be>, we are pleased to send you the necessary information.

-----

Copyright © 2023, [Geomatics Unit of ULiège](http://geomatics.ulg.ac.be/fr/home.php). Released under [BSD-3 Clause License](https://github.com/Yarroudh/segment-lidar/blob/main/LICENSE).
