# Copyright (c) 2023 - University of Liège
# Author : Anass Yarroudh (ayarroudh@uliege.be), Geomatics Unit of ULiege
# This file is distributed under the BSD-3 licence. See LICENSE file for complete text of the license.

import os
import time

import CSF
import torch
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from samgeo import SamGeo
from samgeo.text_sam import LangSAM
from typing import List, Tuple, Union
import rasterio
import laspy
from segment_lidar.view import TopView, PinholeView

class SamLidar:
    class mask:
        def __init__(self, crop_n_layers: int = 1, crop_n_points_downscale_factor: int = 1, min_mask_region_area: int = 200, points_per_side: int = 5, pred_iou_thresh: float = 0.90, stability_score_thresh: float = 0.92):
            """
            Initializes an instance of the mask class.

            :param crop_n_layers: The number of layers to crop from the top of the image, defaults to 1.
            :type crop_n_layers: int
            :param crop_n_points_downscale_factor: The downscale factor for the number of points in the cropped image, defaults to 1.
            :type crop_n_points_downscale_factor: int
            :param min_mask_region_area: The minimum area of a mask region, defaults to 1000.
            :type min_mask_region_area: int
            :param points_per_side: The number of points per side of the mask region, defaults to 32.
            :type points_per_side: int
            :param pred_iou_thresh: The IoU threshold for the predicted mask region, defaults to 0.90.
            :type pred_iou_thresh: float
            :param stability_score_thresh: The stability score threshold for the predicted mask region, defaults to 0.92.
            :type stability_score_thresh: float
            """
            self.crop_n_layers = crop_n_layers
            self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
            self.min_mask_region_area = min_mask_region_area
            self.points_per_side = points_per_side
            self.pred_iou_thresh = pred_iou_thresh
            self.stability_score_thresh = stability_score_thresh

    class text_prompt:
        def __init__(self, text: str = None, box_threshold: float = 0.24, text_threshold: float = 0.15):
            """
            Initializes an instance of the text_prompt class.

            :param text: The text to search for, defaults to None.
            :type text: str
            :param box_threshold: The box threshold, defaults to 0.24.
            :type box_threshold: float
            :param text_threshold: The text threshold, defaults to 0.15.
            :type text_threshold: float
            """
            self.text = text
            self.box_threshold = box_threshold
            self.text_threshold = text_threshold

    def __init__(self, ckpt_path: str, algorithm: str = 'segment-geospatial', model_type: str = 'vit_h', resolution: float = 0.25, height: int = 512, width: int = 512, distance_threshold: float = None, device: str = 'cuda:0', sam_kwargs: bool = False, intrinsics: np.ndarray = None, rotation: np.ndarray = None, translation: np.ndarray = None, interactive: bool = False) -> None:
        """
        Initializes an instance of the SamLidar class.

        :param algorithm: The algorithm to use, defaults to 'segment-geospatial'.
        :type algorithm: str
        :param model_type: The type of the model, defaults to 'vit_h'.
        :type model_type: str
        :param ckpt_path: The path to the model checkpoint.
        :type ckpt_path: str
        :param resolution: The resolution value, defaults to 0.25.
        :type resolution: float
        :param device: The device to use, defaults to 'cuda:0'.
        :type device: str
        :param sam_kwargs: Whether to use the SAM kwargs when using 'segment-geospatial' as algorithm, defaults to False.
        :type sam_kwargs: bool
        :param intrinsics: The intrinsics matrix, defaults to None.
        :type intrinsics: np.ndarray
        :param rotation: The rotation matrix, defaults to None.
        :type rotation: np.ndarray
        :param translation: The translation matrix, defaults to None.
        :type translation: np.ndarray
        :param interactive: A boolean indicating whether to use the interactive mode, defaults to False.
        :type interactive: bool
        """
        self.algorithm = algorithm
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.resolution = resolution
        self.height = height
        self.width = width
        self.distance_threshold = distance_threshold
        self.device = torch.device('cuda:0') if device == 'cuda:0' and torch.cuda.is_available() else torch.device('cpu')
        self.mask = SamLidar.mask()
        self.text_prompt = SamLidar.text_prompt()
        self.intrinsics = intrinsics
        self.rotation = rotation
        self.translation = translation
        self.interactive = interactive

        if algorithm == 'segment-anything':
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam_model_registry[model_type](checkpoint=ckpt_path).to(device=self.device),
                crop_n_layers=self.mask.crop_n_layers,
                crop_n_points_downscale_factor=self.mask.crop_n_points_downscale_factor,
                min_mask_region_area=self.mask.min_mask_region_area,
                points_per_side=self.mask.points_per_side,
                pred_iou_thresh=self.mask.pred_iou_thresh,
                stability_score_thresh=self.mask.stability_score_thresh
            )

        if sam_kwargs:
            self.sam_kwargs = {
                'crop_n_layers': self.mask.crop_n_layers,
                'crop_n_points_downscale_factor': self.mask.crop_n_points_downscale_factor,
                'min_mask_region_area': self.mask.min_mask_region_area,
                'points_per_side': self.mask.points_per_side,
                'pred_iou_thresh': self.mask.pred_iou_thresh,
                'stability_score_thresh': self.mask.stability_score_thresh
            }
        else:
            self.sam_kwargs = None

        self.sam_geo = SamGeo(
            model_type=self.model_type,
            checkpoint=self.ckpt_path,
            sam_kwargs=self.sam_kwargs
        )

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


    def read(self, path: str, classification: int = None) -> np.ndarray:
        """
        Reads a point cloud from a file and returns it as a NumPy array.

        :param path: The path to the input file.
        :type path: str
        :param classification: The optional classification value to filter the point cloud, defaults to None.
        :type classification: int, optional
        :return: The point cloud as a NumPy array.
        :rtype: np.ndarray
        :raises ValueError: If the input file format is not supported.
        """
        start = time.time()
        extension = os.path.splitext(path)[1]
        try:
            if extension not in ['.laz', '.las']:
                raise ValueError(f'The input file format {extension} is not supported.\nThe file format should be [.las/.laz].')
        except ValueError as error:
            message = str(error)
            lines = message.split('\n')
            print(lines[-2])
            print(lines[-1])
            exit()

        print(f'Reading {path}...')

        if extension == '.npy':
            points = np.load(path)
        elif extension in ['.laz', '.las']:
            las = laspy.read(path)
            if classification == None:
                print(f'- Classification value is not provided. Reading all points...')
                pcd = las.points
            else:
                try:
                    if hasattr(las, 'classification'):
                        print(f'- Reading points with classification value {classification}...')
                        pcd = las.points[las.raw_classification == classification]
                    else:
                        raise ValueError(f'The input file does not contain classification values.')
                except ValueError as error:
                    message = str(error)
                    lines = message.split('\n')
                    print(lines[-2])
                    print(lines[-1])
                    exit()

            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                print(f'- Reading RGB values...')
                points = np.vstack((pcd.x, pcd.y, pcd.z, pcd.red / 255.0, pcd.green / 255.0, pcd.blue / 255.0)).transpose()
            else:
                print(f'- RGB values are not provided. Reading only XYZ values...')
                points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()

        end = time.time()
        print(f'File reading is completed in {end - start:.2f} seconds. The point cloud contains {points.shape[0]} points.\n')
        return points


    def csf(self, points: np.ndarray, class_threshold: float = 0.5, cloth_resolution: float = 0.2, iterations: int = 500, slope_smooth: bool = False, csf_path: str = None, exists: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies the CSF (Cloth Simulation Filter) algorithm to filter ground points in a point cloud.

        :param points: The input point cloud as a NumPy array, where each row represents a point with x, y, z coordinates.
        :type points: np.ndarray
        :param class_threshold: The threshold value for classifying points as ground/non-ground, defaults to 0.5.
        :type class_threshold: float, optional
        :param cloth_resolution: The resolution value for cloth simulation, defaults to 0.2.
        :type cloth_resolution: float, optional
        :param iterations: The number of iterations for the CSF algorithm, defaults to 500.
        :type iterations: int, optional
        :param slope_smooth: A boolean indicating whether to enable slope smoothing, defaults to False.
        :type slope_smooth: bool, optional
        :param csf_path: The path to save the results, defaults to None.
        :type csf_path: str, optional
        :param exists: A boolean indicating whether the results already exist, defaults to False.
        :type exists: bool, optional
        :return: A tuple containing three arrays: the filtered point cloud, non-ground (filtered) points indinces and ground points indices.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        if not exists:
            start = time.time()
            print(f'Applying CSF algorithm...')
            csf = CSF.CSF()
            csf.params.bSloopSmooth = slope_smooth
            csf.params.cloth_resolution = cloth_resolution
            csf.params.interations = iterations
            csf.params.class_threshold = class_threshold
            csf.setPointCloud(points[:, :3])
            ground = CSF.VecInt()
            non_ground = CSF.VecInt()
            csf.do_filtering(ground, non_ground)

            cloud = points[non_ground, :]
            os.remove('cloth_nodes.txt')

            if csf_path is not None:
                print(f'Saving the filtered point cloud to {csf_path}...')
                header = laspy.LasHeader(point_format=3, version="1.3")
                lidar = laspy.LasData(header=header)
                lidar.xyz = points[:, :3]
                lidar.red = points[:, 3] * 255
                lidar.green = points[:, 4] * 255
                lidar.blue = points[:, 5] * 255
                classification = np.full(points.shape[0], 0)
                classification[ground] = 1
                lidar.add_extra_dim(laspy.ExtraBytesParams(name="ground", type=np.int8))
                lidar.ground = classification
                lidar.write(csf_path)

            end = time.time()
            print(f'CSF algorithm is completed in {end - start:.2f} seconds. The filtered non-ground cloud contains {points.shape[0]} points.\n')

        else:
            print(f'Reading {csf_path}...')
            las = laspy.read(csf_path)
            ground = las[las.ground == 1]
            ground = np.vstack((ground.x, ground.y, ground.z, ground.red / 255.0, ground.green / 255.0, ground.blue / 255.0)).transpose()
            non_ground = las[las.ground == 0]
            non_ground = np.vstack((non_ground.x, non_ground.y, non_ground.z, non_ground.red / 255.0, non_ground.green / 255.0, non_ground.blue / 255.0)).transpose()
            las = las[las.ground == 0]
            cloud = np.vstack((las.x, las.y, las.z, las.red / 255.0, las.green / 255.0, las.blue / 255.0)).transpose()
            print(f'File reading is completed. The filtered non-ground cloud contains {non_ground.shape[0]} points.\n')

        return cloud, np.asarray(non_ground), np.asarray(ground)


    def segment(self, points: np.ndarray, view: Union[TopView, PinholeView] = TopView(), image_path: str = 'raster.tif', labels_path: str = 'labeled.tif', image_exists: bool = False, label_exists: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segments a point cloud based on the provided parameters and returns the segment IDs, original image, and segmented image.

        :param points: The point cloud data as a NumPy array.
        :type points: np.ndarray
        :param view: The viewpoint to use for segmenting the point cloud, defaults to TopView().
        :type view: Union[TopView, PinholeView]
        :param image_path: Path to the input raster image, defaults to 'raster.tif'.
        :type image_path: str
        :param labels_path: Path to save the labeled output image, defaults to 'labeled.tif'.
        :type labels_path: str
        :param image_exists: A boolean indicating whether the raster image already exists, defaults to False.
        :type image_exists: bool
        :param label_exists: A boolean indicating whether the labeled image already exists, defaults to False.
        :type label_exists: bool
        :return: A tuple containing the segment IDs, segmented image, and RGB image.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        start = time.time()

        directory = os.path.dirname(labels_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if image_exists:
            print(f'- Reading raster image...')
        else:
            print(f'- Generating raster image...')

        if label_exists:
            print(f'- Reading segmented image...')

        if view.__class__.__name__ == 'TopView':
            image = view.cloud_to_image(points=points, resolution=self.resolution)
        elif view.__class__.__name__ == 'PinholeView':
            if self.interactive:
                image, K, pose = view.cloud_to_image(points=points, resolution=self.resolution, distance_threshold=self.distance_threshold)
            else:
                image, K, pose = view.cloud_to_image(points=points, resolution=self.resolution, distance_threshold=self.distance_threshold, intrinsics=self.intrinsics, rotation=self.rotation, translation=self.translation)

        image = np.asarray(image).astype(np.uint8)

        if not image_exists:
            print(f'- Saving raster image...')
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

        with rasterio.open(image_path, 'r') as src:
            image_rgb = src.read()

        if not label_exists:
            print(f'- Applying {self.algorithm} to raster image...')
            if self.algorithm == 'segment-anything':
                sam = sam_model_registry[self.model_type](checkpoint=self.ckpt_path)
                sam.to(self.device)

                image_rgb = image_rgb.transpose(1, 2, 0)
                result = self.mask_generator.generate(image_rgb)
                detections = sv.Detections.from_sam(result)
                num_masks, height, width = detections.mask.shape
                segmented_image = np.zeros((height, width), dtype=np.uint8)
                for i in range(num_masks):
                    mask = detections.mask[i]
                    segmented_image[mask] = i

                print(f'- Saving segmented image...')
                with rasterio.open(
                    labels_path,
                    'w',
                    driver='GTiff',
                    width=segmented_image.shape[1],
                    height=segmented_image.shape[0],
                    count=1,
                    dtype=segmented_image.dtype
                ) as dst:
                    dst.write(segmented_image, 1)

                print(f'- Saving segmented image...')

            elif self.algorithm == 'segment-geospatial':
                if self.text_prompt.text is not None:
                    print(f'- Generating labels using text prompt...')
                    sam = LangSAM()
                    sam.predict(image=image_path, text_prompt=self.text_prompt.text, box_threshold=self.text_prompt.box_threshold, text_threshold=self.text_prompt.text_threshold, output=labels_path)
                    print(f'- Saving segmented image...')
                else:
                    sam = self.sam_geo
                    sam.generate(source=image_path, output=labels_path, erosion_kernel=(3, 3), foreground=True, unique=True)
                    print(f'- Saving segmented image...')

        with rasterio.open(labels_path, 'r') as src:
            segmented_image = src.read()
            segmented_image = np.squeeze(segmented_image)

        print(f'- Generating segment IDs...')
        if view.__class__.__name__ == 'TopView':
            segment_ids = view.image_to_cloud(points=points, image=segmented_image, resolution=self.resolution)
        elif view.__class__.__name__ == 'PinholeView':
            segment_ids = view.image_to_cloud(points=points, image=segmented_image, intrinsics=K, extrinsics=pose)
        end = time.time()

        print(f'Segmentation is completed in {end - start:.2f} seconds. Number of instances: {np.max(segmented_image)}\n')

        if view.__class__.__name__ == 'TopView':
            return segment_ids, segmented_image, image_rgb
        elif view.__class__.__name__ == 'PinholeView':
            return segment_ids, segmented_image, image_rgb, K, pose


    def write(self, points: np.ndarray, segment_ids: np.ndarray, non_ground: np.ndarray = None, ground: np.ndarray = None, save_path: str = 'segmented.las', ground_path: str = None) -> None:
        """
        Writes the segmented point cloud data to a LAS/LAZ file.

        :param points: The input point cloud data as a NumPy array, where each row represents a point with x, y, z coordinates.
        :type points: np.ndarray
        :param segment_ids: The segment IDs corresponding to each point in the point cloud.
        :type segment_ids: np.ndarray
        :param non_ground: Optional array of indices for non-ground points in the original point cloud (default: None).
        :type non_ground: np.ndarray, optional
        :param ground: Optional array of indices for ground points in the original point cloud (default: None).
        :type ground: np.ndarray, optional
        :param save_path: The path to save the segmented LAS/LAZ file (default: 'segmented.las').
        :type save_path: str, optional
        :return: None
        """
        start = time.time()

        directory = os.path.dirname(save_path)
        if not os.path.exists(directory) and directory != '':
            os.makedirs(directory)

        if ground_path is not None:
            las = laspy.read(ground_path)
            ground = las[las.classification == 1]
            ground = np.vstack((ground.x, ground.y, ground.z, ground.red / 255.0, ground.green / 255.0, ground.blue / 255.0)).transpose()
            non_ground = las[las.classification == 0]
            non_ground = np.vstack((non_ground.x, non_ground.y, non_ground.z, non_ground.red / 255.0, non_ground.green / 255.0, non_ground.blue / 255.0)).transpose()

        extension = os.path.splitext(save_path)[1]
        try:
            if extension not in ['.laz', '.las']:
                raise ValueError(f'The input file format {extension} is not supported.\nThe file format should be [.las/.laz].')
        except ValueError as error:
            message = str(error)
            lines = message.split('\n')
            print(lines[-2])
            print(lines[-1])
            exit()

        print(f'Writing the segmented point cloud to {save_path}...')

        header = laspy.LasHeader(point_format=3, version="1.3")
        lidar = laspy.LasData(header=header)

        if ground is not None:
            cloud = np.concatenate((ground, non_ground))

            if cloud.ndim == 1:
                indices = np.concatenate((non_ground, ground))
                lidar.xyz = points[indices]
                colors = points[indices, 3:]
                if points.shape[1] > 3:
                    lidar.red = colors[:, 0] * 255
                    lidar.green = colors[:, 1] * 255
                    lidar.blue = colors[:, 2] * 255
            else:
                lidar.xyz = cloud[:, :3]
                if cloud.shape[1] > 3:
                    lidar.red = cloud[:, 3] * 255
                    lidar.green = cloud[:, 4] * 255
                    lidar.blue = cloud[:, 5] * 255

            segment_ids = np.append(segment_ids, np.full(len(ground), -1))
            lidar.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.int32))
            lidar.segment_id = segment_ids
        else:
            lidar.xyz = points[:, :3]
            if points.shape[1] > 3:
                lidar.red = points[:, 3] * 255
                lidar.green = points[:, 4] * 255
                lidar.blue = points[:, 5] * 255
            lidar.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.int32))
            lidar.segment_id = segment_ids

        lidar.write(save_path)

        end = time.time()
        print(f'Writing is completed in {end - start:.2f} seconds.\n')
        return None