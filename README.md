<img src="https://user-images.githubusercontent.com/72500344/210864557-4078754f-86c1-4e7c-b291-73223bdf4e4d.png" alt="logo" width="200"/>

# segment-lidar
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Yarroudh/ZRect3D/blob/main/LICENSE)
[![Geomatics Unit of ULiege - Development](https://img.shields.io/badge/Geomatics_Unit_of_ULiege-Development-2ea44f)](http://geomatics.ulg.ac.be/)

*Python CLI for segmenting LiDAR data using Segment-Anything Model (SAM) from Meta Research.*

This package is specifically designed for **unsupervised instance segmentation** of **aerial LiDAR data**. It brings together the power of the **Segment-Anything Model (SAM)** developed by [Meta Research](https://github.com/facebookresearch) and the **segment-geospatial** package from [Open Geospatial Solutions](https://github.com/opengeos). Whether you're a researcher, developer, or a geospatial enthusiast, segment-lidar opens up new possibilities for automatic processing of aerial LiDAR data and enables further applications. We encourage you to explore our code, contribute to its development and leverage its capabilities for your segmentation tasks.

![results](https://github.com/Yarroudh/segment-lidar/assets/72500344/1ed9c405-a305-46ff-827b-a70275733371)


## Installation

We recommand using `python==3.9`. First, you need to install `PyTorch`. Please follow the instructions [here](https://pytorch.org/).
Make sure to also install **Cloth Simulation Filter** for **ground filtering** option by running:

```bash
pip install git+https://github.com/jianboqi/CSF.git
```

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

## Usage of the package

After installation, you have a small program called <code>segment-lidar</code>. Use <code>segment-lidar --help</code> to see the detailed help:

```
Usage: segment-lidar [OPTIONS] COMMAND [ARGS]...

  A package for segmenting LiDAR data using Segment-Anything from Meta AI
  Research.

Options:
  --help  Show this message and exit.

Commands:
  create-config  Create a configuration YAML file.
  segment        Segment LiDAR data.
```

The package reads `.las` or `.laz` file, then perform instance segmentation using [segment-geospatial](https://github.com/opengeos/segment-geospatial) or/and [segment-anything](https://github.com/facebookresearch/segment-anything) algorithms. The user should first create the **configuration** file by running:

```bash
segment-lidar create-config -o config.yaml
```

This will create a configuration file as follow:

```yaml
algorithm: segment-geospatial
classification: null
csf_filter: true
csf_params:
  class_threshold: 0.5
  cloth_resolution: 0.2
  interations: 500
  slope_smooth: false
device: cuda:0
image_path: raster.tif
input_path: pointcloud.las
model_path: sam_vit_h_4b8939.pth
model_type: vit_h
output_path: classified.las
resolution: 0.15
sam_geo:
  automatic: true
  box_threshold: 0.24
  erosion_kernel_size: 3
  sam_kwargs: true
  text_prompt: null
  text_threshold: 0.3
sam_kwargs:
  crop_n_layers: 1
  crop_n_points_downscale_factor: 1
  min_mask_region_area: 1000
  points_per_side: 32
  pred_iou_thresh: 0.95
  stability_score_thresh: 0.92
```

The second step is to run the segmentation:

```bash
segment-lidar segment --config config.yaml
```

The resulted point cloud contains a new scalar field called `segment_id`. For visualization and further processing, we recommand using [CloudCompare](https://www.danielgm.net/cc/).

## Testing data

For testing purposes, you can download a sample here: [pointcloud.las](https://drive.google.com/file/d/16EF2aRSvo8u0pXvwtaQ6sjhP5h0sWw3o/view?usp=sharing).

This data was retrieved from **AHN-4**. For more data, please visit [AHN-Viewer](https://ahn.arcgisonline.nl/ahnviewer/)

## Configuration

- `algorithm`: algorithm to use for instance segmentation [segment-geospatial/segment-anything].
- `model_path`: path of the model checkpoints. See **segment-anything** repository for models.
- `model_type`: SAM model version [**vit_h/vit_l/vit_b**].
- `device`: if **cpu** the prediction will use the CPU, if you have cuda, use **cuda:0** instead for GPU.
- `input_path`: path to your input **.las/.laz** file.
- `output_path`: path to your output .las/.laz file. The results will be saved in this file.
- `image_path`: path to the resulted image. The segmentation results of SAM or segment-geospatial will be saved in this file.
- `classification`: specify the class number you want to segment. This will limit instance segmentation to specified **class**.
- `csf_filter`: apply ground filtering using cloth simulation filter.
- `csf_params`: refer to [Cloth Simulation Filer](http://ramm.bnu.edu.cn/projects/CSF) for additional information.
- `resolution`: resolution of the image created from the point cloud.
- `sam_kwargs`: refer to **segment-anything** for additional information.
- `sam_geo`: refer to **segment-geospatial** for additional information.

Please note that the actual version is a pre-release and it's under tests. If you find any issues or bugs, please report them in [issues](https://github.com/Yarroudh/segment-lidar/issues) section. The second version will have more advanced features and fonctionalities.

## Related repositories

We would like to express our acknowledgments to the creators of:

- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [segment-geospatial](https://github.com/opengeos/segment-geospatial)

## Documentation

Coming soon.

## License

This software is under the BSD 3-Clause "New" or "Revised" license which is a permissive license that allows you almost unlimited freedom with the software so long as you include the BSD copyright and license notice in it. Please refer to the [LICENSE](https://github.com/Yarroudh/segment-lidar/blob/main/LICENSE) file for more detailed information.

## Author

This software was developped by [Anass Yarroudh](https://www.linkedin.com/in/anass-yarroudh/), a Research Engineer in the [Geomatics Unit of the University of Liege](http://geomatics.ulg.ac.be/fr/home.php).
For more detailed information please contact us via <ayarroudh@uliege.be>, we are pleased to send you the necessary information.
