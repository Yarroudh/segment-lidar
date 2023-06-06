import os
import yaml
import click
import time
import torch
import cv2
import numpy as np
from osgeo import gdal
import supervision as sv
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


@click.group()
def cli():
    pass

@cli.command()
@click.option('--input', '-i', required=True, help='Input LiDAR data')
@click.option('--output', '-o', required=True, help='Output file')
@click.option('--config', '-m', required=True, help='Configuration file')
def segment(input, output, config):

    # Read configuration file
    cnfg = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
    environment = cnfg['environment']
    device = cnfg['device']
    model_type = cnfg['model_type']
    ckp_path = cnfg['ckp_path']
    mask = cnfg['mask_generator']

    # Set environment variables
    if environment['KMP_DUPLICATE_LIB_OK'] == 'TRUE':
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # Set device
    if device == 'cuda:0':
        if torch.cuda.is_available():
            DEVICE = torch.device(device)
        else:
            DEVICE = torch.device('cpu')
    else:
        DEVICE = torch.device('cpu')

    # Load model
    sam = sam_model_registry[model_type](checkpoint=ckp_path)
    sam.to(device=DEVICE)

    # Set mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=mask['model'],
        points_per_side=mask['points_per_side'],
        pred_iou_thresh=mask['pred_iou_thresh'],
        stability_score_thresh=mask['stability_score_thresh'],
        crop_n_layers=mask['crop_n_layers'],
        crop_n_points_downscale_factor=mask['crop_n_points_downscale_factor'],
        min_mask_region_area=mask['min_mask_region_area']
    )

    # Read input LiDAR data