# Copyright (c) 2023 - University of Liège
# Author : Anass Yarroudh (ayarroudh@uliege.be), Geomatics Unit of ULiege
# This file is distributed under the BSD-3 licence. See LICENSE file for complete text of the license.

import os

try:
    import CSF
except ImportError:
    os.system('pip install git+https://github.com/jianboqi/CSF.git')

import yaml
import click
import time
import torch
import cv2
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from samgeo import SamGeo
from samgeo.text_sam import LangSAM
import rasterio
import laspy


def cloud_to_image(points, minx, maxx, miny, maxy, resolution):
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1

    image = np.zeros((height, width, 3), dtype=np.uint8)
    for point in points:
        if points.shape[1] == 3:
            x, y, *_ = point
            r, g, b = (255, 255, 255)
            pixel_x = int((x - minx) / resolution)
            pixel_y = int((maxy - y) / resolution)
            image[pixel_y, pixel_x] = [r, g, b]
        else:
            x, y, *_ = point
            r, g, b = point[-3:]
            pixel_x = int((x - minx) / resolution)
            pixel_y = int((maxy - y) / resolution)
            image[pixel_y, pixel_x] = [r, g, b]

    return image


def image_to_cloud(points, minx, maxy, image, resolution):
    segment_ids = []
    unique_values = {}
    image = np.asarray(image)

    for point in points:
        x, y, *_ = point
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)
        if not (0 <= pixel_x < image.shape[1]) or not (0 <= pixel_y < image.shape[0]):
            segment_ids.append(-1)
            continue
        rgb = image[pixel_y, pixel_x]

        if rgb not in unique_values:
            unique_values[rgb] = len(unique_values)

        id = unique_values[rgb]
        segment_ids.append(id)

    return segment_ids


@click.group(help="A package for segmenting LiDAR data using Segment-Anything from Meta AI Research.")
def cli():
    pass


@click.command()
@click.option('--output', '-o', type=click.Path(exists=False), required=False, help='Output path')

def create_config(output):
    '''
    Create a configuration YAML file.
    '''
    configuration = {
        "algorithm": "segment-geospatial",
        "classification": None,
        "csf_filter": True,
        "csf_params": {
            "slope_smooth": False,
            "cloth_resolution": 0.2,
            "class_threshold": 0.5,
            "iterations": 500
        },
        "device": "cuda:0",
        "image_path": "raster.tif",
        "input_path": "pointcloud.las",
        "model_path": "sam_vit_h_4b8939.pth",
        "model_type": "vit_h",
        "output_path": "classified.las",
        "resolution": 0.15,
        "sam_geo": {
            "automatic": True,
            "box_threshold": 0.24,
            "erosion_kernel_size": 3,
            "sam_kwargs": True,
            "text_prompt": None,
            "text_threshold": 0.3
        },
        "sam_kwargs": {
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 1000,
            "points_per_side": 32,
            "pred_iou_thresh": 0.95,
            "stability_score_thresh": 0.92
        }
    }

    yaml_data = yaml.dump(configuration)
    output = output if output else 'config.yaml'

    with open(output, 'w') as file:
        file.write(yaml_data)

    print(f'Configuration file created at {output}.')


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=False, help='Input LiDAR data')
@click.option('--output', '-o', type=click.Path(exists=False), required=False, help='Output file')
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')

def segment(input, output, config):
    '''
    Segment LiDAR data.
    '''
    start = time.time()

    # Read configuration file
    cnfg = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
    device = cnfg['device']
    model_type = cnfg['model_type']
    model_path = cnfg['model_path']
    mask = cnfg['sam_kwargs']

    # Set environment variables and device
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    DEVICE = torch.device('cuda:0') if device == 'cuda:0' and torch.cuda.is_available() else torch.device('cpu')

    # Read input LiDAR data
    input_path = input if input else cnfg['input_path']
    lidar = laspy.read(input_path)


    # 3D point cloud to 2D image
    image_path = cnfg['image_path']
    resolution = cnfg['resolution']
    classification = cnfg['classification']
    extension = os.path.splitext(input_path)[1]

    try:
        if extension not in ['.laz', '.las']:
            raise ValueError(f'The input file format {extension} is not supported.\nThe file format should be [.las/.laz].')
    except ValueError as error:
        error_message = str(error)
        error_lines = error_message.split('\n')
        print(error_lines[-2])
        print(error_lines[-1])
        exit()

    extension = '.las' if extension == '.laz' else extension

    # Filter points based on classification value
    if classification == None:
        pcd = lidar.points
    else:
        pcd = lidar.points[lidar.classification == classification]

    # Store points in a numpy array
    if hasattr(lidar, 'red') and hasattr(lidar, 'green') and hasattr(lidar, 'blue'):
        points = np.vstack((pcd.x, pcd.y, pcd.z, pcd.red / 255.0, pcd.green / 255.0, pcd.blue / 255.0)).transpose()
    else:
        points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()

    # Filter ground points if required
    if cnfg['csf_filter']:
        csf = CSF.CSF()
        csf.params.bSloopSmooth = cnfg['csf_params']['slope_smooth']
        csf.params.cloth_resolution = cnfg['csf_params']['cloth_resolution']
        csf.params.interations = cnfg['csf_params']['iterations']
        csf.params.class_threshold = cnfg['csf_params']['class_threshold']
        csf.setPointCloud(points[:, :3])

        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground)
        points = points[non_ground, :]

    # Convert points to image
    minx = np.min(points[:, 0])
    maxx = np.max(points[:, 0])
    miny = np.min(points[:, 1])
    maxy = np.max(points[:, 1])
    image = cloud_to_image(points, minx, maxx, miny, maxy, resolution)
    image = np.asarray(image).astype(np.uint8)

    with rasterio.open(
        image_path,
        'w',
        driver='GTiff',
        width=image.shape[1],
        height=image.shape[0],
        count=3,
        dtype=image.dtype
    ) as dst:
        for i in range(3):
            dst.write(image[:, :, i], i + 1)

    # Instance segmentation
    with rasterio.open(image_path, 'r') as src:
        image_rgb = src.read()

    output_path = output if output else cnfg['output_path']
    if cnfg['algorithm'] == 'segment-anything':
        # Load model
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=DEVICE)

        # Set mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=mask['points_per_side'],
            pred_iou_thresh=mask['pred_iou_thresh'],
            stability_score_thresh=mask['stability_score_thresh'],
            crop_n_layers=mask['crop_n_layers'],
            crop_n_points_downscale_factor=mask['crop_n_points_downscale_factor'],
            min_mask_region_area=mask['min_mask_region_area']
        )

        image_rgb = image_rgb.reshape((image_rgb.shape[1], image_rgb.shape[2], image_rgb.shape[0]))
        result = mask_generator.generate(image_rgb)
        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(result)
        annotated_image = mask_annotator.annotate(image_rgb, detections)

        # Save annotated image
        cv2.imwrite(f"{os.path.dirname(output_path)}/labeled.tif", annotated_image)

    elif cnfg['algorithm'] == 'segment-geospatial':
        # Parameters
        sam_kwargs = mask if cnfg['sam_geo']['sam_kwargs'] else None
        automatic = cnfg['sam_geo']['automatic']
        text_prompt = cnfg['sam_geo']['text_prompt']
        box_threshold = cnfg['sam_geo']['box_threshold']
        text_threshold = cnfg['sam_geo']['text_threshold']
        erosion_kernel = (cnfg['sam_geo']['erosion_kernel_size'], cnfg['sam_geo']['erosion_kernel_size']) if cnfg['sam_geo']['erosion_kernel_size'] else None

        # Segmentation
        if text_prompt is not None:
            sam = LangSAM()
            sam.predict(image=image_path, text_prompt=text_prompt, box_threshold=box_threshold, text_threshold=text_threshold, output=f"{os.path.dirname(output_path)}/labeled.tif",)

        else:
            sam = SamGeo(
                model_type=model_type,
                checkpoint=model_path,
                automatic=automatic,
                sam_kwargs=sam_kwargs
            )
            sam.generate(source=image_path, output=f"{os.path.dirname(output_path)}/labeled.tif", erosion_kernel=erosion_kernel, foreground=True, unique=True)

    # Project results on point cloud
    extension = os.path.splitext(output_path)[1]
    try:
        if extension not in ['.laz', '.las']:
            raise ValueError(f'The output file format {extension} is not supported.\nThe file format should be [.las/.laz].')
    except ValueError as error:
        error_message = str(error)
        error_lines = error_message.split('\n')
        print(error_lines[-2])
        print(error_lines[-1])
        exit()

    with rasterio.open(os.path.dirname(output_path) + "/labeled.tif", 'r') as src:
        segmented_image = src.read()
        segmented_image = np.squeeze(segmented_image)

    segment_ids = image_to_cloud(points, minx, maxy, segmented_image, resolution)

    # Save point cloud
    if cnfg['csf_filter']:
        points = np.concatenate((non_ground, ground))
        lidar.points = pcd[points]
        segment_ids = np.append(segment_ids, np.full(len(ground), -1))
        lidar.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.int32))
        lidar.segment_id = segment_ids
    else:
        lidar.points = pcd
        lidar.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.int32))
        lidar.segment_id = segment_ids

    lidar.write(output_path)


    end = time.time()
    print(f'Point cloud segmentation completed in {np.round(end - start, 2)} seconds.')


cli.add_command(create_config)
cli.add_command(segment)

if __name__ == '__main__':
    cli(prog_name='segment-lidar')
