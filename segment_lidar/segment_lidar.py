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
from osgeo import gdal
gdal.UseExceptions()


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
    mask = cnfg['mask_generator']

    # Set environment variables and device
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    DEVICE = torch.device('cuda:0') if device == 'cuda:0' and torch.cuda.is_available() else torch.device('cpu')

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

    # Read input LiDAR data
    input_path = input if input else cnfg['input_path']

    # 3D point cloud to 2D image
    image_path = cnfg['image_path']
    resolution = cnfg['resolution']
    extension = os.path.splitext(input_path)[1]

    try:
        if extension not in ['.bpf', '.laz', '.las', '.buffer', '.copc', '.draco', '.ept', '.e57', '.faux', '.hdf', '.matlab', '.nitf', '.numpy', '.obj', '.ply', '.pts', '.text', '.pcd', '.pgpointcloud']:
            raise ValueError(f'The input file format {extension} is not supported.\nSee PDAL readers for more information.')
    except ValueError as error:
        error_message = str(error)
        error_lines = error_message.split('\n')
        print(error_lines[-2])
        print(error_lines[-1])
        exit()

    extension = '.las' if extension == '.laz' else extension

    steps = []
    steps.append({
        "type":f"readers{extension}",
        "filename":f"{input_path}"
    })

    classification = cnfg['classification']
    if classification is not None:
        steps.append({
            "type":"filters.range",
            "limits":f"Classification[{classification[0]}:{classification[-1]}]"
        })

    steps.append({
        "filename":f"{image_path}",
        "gdaldriver":"GTiff",
        "output_type":"max",
        "resolution":f"{resolution}",
        "data_type":"uint8",
        "type": "writers.gdal"
    })

    data = {
        "pipeline": steps
    }

    pipeline = pdal.Pipeline(json.dumps(data))
    pipeline.execute()

    # Instance segmentation with SAM
    gdal.AllRegister()
    src = gdal.Open(f"{image_path}", 1)
    image = src.ReadAsArray()

    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = mask_generator.generate(image_bgr)

    mask_annotator = sv.MaskAnnotator()
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(image_bgr, detections)

    # Save annotated image
    output_path = output if output else cnfg['output_path']
    cv2.imwrite(f"{os.path.join(os.path.dirname(output_path), 'labeled.tif')}", annotated_image)

    # Project results on point cloud
    extension = os.path.splitext(output_path)[1]
    try:
        if extension not in ['.bpf', '.laz', '.las', '.buffer', '.copc', '.draco', '.ept', '.e57', '.faux', '.hdf', '.matlab', '.nitf', '.numpy', '.obj', '.ply', '.pts', '.text', '.pcd', '.pgpointcloud']:
            raise ValueError(f'The output file format {extension} is not supported.\nSee PDAL writers for more information.')
    except ValueError as error:
        error_message = str(error)
        error_lines = error_message.split('\n')
        print(error_lines[-2])
        print(error_lines[-1])
        exit()

    extension = '.las' if extension == '.laz' else extension

    data = {
        "pipeline": [
            {
                "type":f"readers{extension}",
                "filename":f"{input_path}"
            },
            {
                "type": "filters.colorization",
                "dimensions":"segment_id",
                "raster": f"{image_path}"
            },
            {
                "type": f"writers{extension}",
                "filename":f"{output_path}",
                "extra_dims":"segment_id=int8"
            }
        ]
    }
    pipeline = pdal.Pipeline(json.dumps(data))
    pipeline.execute()

    end = time.time()
    print(f'Point cloud segmentation completed in {np.round(end - start, 2)} seconds.')

cli.add_command(segment)

if __name__ == '__main__':
    cli()
