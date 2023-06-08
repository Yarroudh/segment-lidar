<img src="https://user-images.githubusercontent.com/72500344/210864557-4078754f-86c1-4e7c-b291-73223bdf4e4d.png" alt="logo" width="200"/>

# segment-lidar
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Geomatics Unit of ULiege - Development](https://img.shields.io/badge/Geomatics_Unit_of_ULiege-Development-2ea44f)](http://geomatics.ulg.ac.be/)

*A package for segmenting LiDAR data using Segment-Anything Model (SAM) from Meta Research.*

This package is specifically designed for **unsupervised instance segmentation** of **aerial LiDAR data**. It brings together the power of the **Segment-Anything Model (SAM)** developed by [Meta Research](https://github.com/facebookresearch) and the **segment-geospatial** package from [Open Geospatial Solutions](https://github.com/opengeos). Whether you're a researcher, developer, or a geospatial enthusiast, segment-lidar opens up new possibilities for automatic processing of aerial LiDAR data and enables further applications. We encourage you to explore our code, contribute to its development and leverage its capabilities for your segmentation tasks.

![ezgif-2-e94819e04d](https://github.com/Yarroudh/segment-lidar/assets/72500344/634ef333-a7a0-421e-868f-c78f2e18e7ee)

## Installation

To install `segment-lidar` from source, you need to run the following commands:

```bash
git clone https://github.com/Yarroudh/segment-lidar
cd segment-lidar
python setup.py install
```

Verify that the application was correctly installed by running this command:

```bash
segment-lidar --help
```

If the command shows you the following message, the application is correctly installed in your environment:

```bash
Usage: segment-lidar [OPTIONS] COMMAND [ARGS]...

  A package for segmenting LiDAR data using Segment-Anything from Meta AI
  Research.

Options:
  --help  Show this message and exit.

Commands:
  create-config  Create a configuration YAML file.
  segment        Segment LiDAR data.
```
