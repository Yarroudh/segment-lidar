import os
import yaml
import click
import time
import json
import torch
import cv2
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pdal
from samgeo import SamGeo, tms_to_geotiff
from osgeo import gdal
import laspy
from PIL import Image
gdal.UseExceptions()

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

    for point in points:
        x, y, *_ = point
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)
        rgb = tuple(image[pixel_y, pixel_x])

        if rgb not in unique_values:
            unique_values[rgb] = len(unique_values)

        id = unique_values[rgb]
        segment_ids.append(id)

    return segment_ids


@click.group(help="A package for segmenting LiDAR data using Segment-Anything from Meta AI Research.")
def cli():
    pass

@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=False, help='Input LiDAR data')
@click.option('--output', '-o', type=click.Path(exists=False), required=False, help='Output file')
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')

def segment(input, output, config):
    '''
    Segment LiDAR data using SAM.
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

    # 3D point cloud to 2D image
    image_path = cnfg['image_path']
    resolution = cnfg['resolution']
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

    lidar = laspy.read(input_path)
    if hasattr(lidar, 'red') and hasattr(lidar, 'green') and hasattr(lidar, 'blue'):
        points = np.vstack((lidar.x, lidar.y, lidar.z, lidar.red / 255.0, lidar.green / 255.0, lidar.blue / 255.0)).transpose()
    else:
        points = np.vstack((lidar.x, lidar.y, lidar.z)).transpose()

    minx = np.min(points[:, 0])
    maxx = np.max(points[:, 0])
    miny = np.min(points[:, 1])
    maxy = np.max(points[:, 1])
    image = cloud_to_image(points, minx, maxx, miny, maxy, resolution)

    image = np.asarray(image).astype(np.uint8)
    geotransform = (minx, resolution, 0, miny, 0, -resolution)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(image_path, image.shape[1], image.shape[0], 3, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geotransform)
    for i in range(3):
        band = out_ds.GetRasterBand(i + 1)
        band.WriteArray(image[:, :, i])
    out_ds = None

    # Instance segmentation
    gdal.AllRegister()
    src = gdal.Open(image_path, 1)
    band = src.GetRasterBand(1)
    image_rgb = src.ReadAsArray()

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

        result = mask_generator.generate(image_rgb)
        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(result)
        annotated_image = mask_annotator.annotate(image_rgb, detections)

        # Save annotated image
        cv2.imwrite(f"{os.path.dirname(output_path)}/labeled.tif", annotated_image)

    elif cnfg['algorithm'] == 'segment-geospatial':
        sam = SamGeo(
            model_type=model_type,
            checkpoint=model_path,
            sam_kwargs=mask
        )

        sam.generate(source=image_path, output=f"{os.path.dirname(output_path)}/labeled.tif", foreground=True, unique=True)

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

    segment_ids = image_to_cloud(points, f"{os.path.dirname(output_path)}/labeled.tif", minx, maxy, resolution)
    lidar.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.int32))
    lidar.segment_id = segment_ids

    lidar.write(output_path)

    end = time.time()
    print(f'Point cloud segmentation completed in {np.round(end - start, 2)} seconds.')

cli.add_command(segment)

if __name__ == '__main__':
    cli()
