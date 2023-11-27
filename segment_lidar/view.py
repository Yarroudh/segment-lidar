# Copyright (c) 2023 - University of Liège
# Author : Anass Yarroudh (ayarroudh@uliege.be), Geomatics Unit of ULiege
# This file is distributed under the BSD-3 licence. See LICENSE file for complete text of the license.

import numpy as np
from typing import Tuple
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import json
import os
import time


class TopView:
    """
    The TopView class converts a point cloud to a top view image and vice versa.
    """
    def __init__(self) -> None:
        """
        Initializes a new instance of the CubicView class.
        """
        pass

    def cloud_to_image(self, points: np.ndarray, resolution: float) -> np.ndarray:
        """
        Converts a point cloud to a planar image.

        :param points: An array of points in the cloud, where each row represents a point.
                    The array shape can be (N, 3) or (N, 6).
                    If the shape is (N, 3), each point is assumed to have white color (255, 255, 255).
                    If the shape is (N, 6), the last three columns represent the RGB color values for each point.
        :type points: ndarray
        :param minx: The minimum x-coordinate value of the cloud bounding box.
        :type minx: float
        :param maxx: The maximum x-coordinate value of the cloud bounding box.
        :type maxx: float
        :param miny: The minimum y-coordinate value of the cloud bounding box.
        :type miny: float
        :param maxy: The maximum y-coordinate value of the cloud bounding box.
        :type maxy: float
        :param resolution: The resolution of the image in units per pixel.
        :type resolution: float
        :return: An image array representing the point cloud, where each pixel contains the RGB color values
                of the corresponding point in the cloud.
        :rtype: ndarray
        :raises ValueError: If the shape of the points array is not valid or if any parameter is invalid.
        """
        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])
        minz, maxz = np.min(points[:, 2]), np.max(points[:, 2])

        width = int((maxx - minx) / resolution) + 1
        height = int((maxy - miny) / resolution) + 1

        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i, point in enumerate(points):
            if points.shape[1] == 3:
                x, y, z, *_ = point
                r, g, b = np.array([255, 255, 255])
            else:
                x, y, z, r, g, b = point

            pixel_x = int((x - minx) / resolution)
            pixel_y = int((maxy - y) / resolution)
            criterion = z

            closest_criterion = np.zeros((height, width), dtype=np.float32)
            closest_criterion[pixel_y, pixel_x] = criterion

            if criterion >= closest_criterion[pixel_y, pixel_x]:
                image[pixel_y, pixel_x] = np.array([r, g, b])

        return image


    def image_to_cloud(self, points: np.ndarray, image: np.ndarray, resolution: float) -> np.ndarray:
        """
        Converts an image to a point cloud.

        :param points: An array of points in the cloud, where each row represents a point.
                    The array shape can be (N, 3) or (N, 6).
                    If the shape is (N, 3), each point is assumed to have white color (255, 255, 255).
                    If the shape is (N, 6), the last three columns represent the RGB color values for each point.
        :type points: ndarray
        :param image: An image array representing the point cloud, where each pixel contains the RGB color values of the corresponding point in the cloud.
        :type image: ndarray
        :param resolution: The resolution of the image in units per pixel.
        :type resolution: float
        :return: An array of segments' IDs in the cloud, where each row represents the segment's ID of a point.
        :rtype: ndarray
        :raises ValueError: If the shape of the points array is not valid or if any parameter is invalid.
        """
        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])
        minz, maxz = np.min(points[:, 2]), np.max(points[:, 2])

        segment_ids = []
        unique_values = {}
        image = np.asarray(image)

        for i, point in enumerate(points):
            x, y, z, *_ = point

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


class PinholeView:
    """
    The PinholeView class converts a point cloud to a pinhole camera view image and vice versa.
    """
    def __init__(self, interactive: bool = True) -> None:
        """
        Initializes a new instance of the CustomCameraView class.
        """
        self.interactive = interactive
        pass

    def cloud_to_image(self, points: np.ndarray, resolution: float = 0.1, rotation: np.ndarray = None, translation: np.ndarray = None, intrinsics: np.ndarray = None, distance_threshold: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts a point cloud to an image.

        :param extrinsics: The extrinsics matrix of the camera.
        :type extrinsics: ndarray (4x4)
        :param intrinsics: The intrinsics matrix of the camera.
        :type intrinsics: ndarray (width, height, fx, fy, cx, cy) (6x1)
        :param points: An array of points in the cloud, where each row represents a point.
                    The array shape can be (N, 3) or (N, 6).
                    If the shape is (N, 3), each point is assumed to have white color (255, 255, 255).
                    If the shape is (N, 6), the last three columns represent the RGB color values for each point.
        :type points: ndarray
        :param resolution: The resolution of the image in units per pixel.
        :type resolution: float
        :param distance_threshold: An optional distance threshold. Points with distances greater than this threshold are ignored.
        :type distance_threshold: float
        :return: A tuple containing:
            - An image array representing the point cloud, where each pixel contains the RGB color values of the corresponding point in the cloud.
            - An array of pixel x-coordinates in the image.
            - An array of pixel y-coordinates in the image.
        :rtype: tuple of ndarrays
        :raises ValueError: If the shape of the points array is not valid or if any parameter is invalid.
        """

        # Calculate the width and height of the image
        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])

        width = int((maxx - minx) / resolution) + 1
        height = int((maxy - miny) / resolution) + 1

        if not self.interactive:
            # Create 4x4 extrinsics matrix
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = rotation
            extrinsics[:3, 3] = translation

            # Separate the points into 3D coordinates and color values
            coords = points[:, :3]
            colors = points[:, 3:6]

            # Camera center
            C = -np.dot(np.linalg.inv(rotation), translation)

            # Calculate points distance from the camera center
            distances = np.linalg.norm(coords - C, axis=1)

            # Filter points based on the distance threshold
            if distance_threshold is not None:
                coords = coords[distances <= distance_threshold]
                colors = colors[distances <= distance_threshold]

            # Project 3D points to 2D image using projectPoints
            if coords.shape[0] > 0:
                points_2d, _ = cv2.projectPoints(coords, rotation, translation, intrinsics, None)
            else:
                image = np.zeros((height, width, 3), dtype=np.uint8)
                print("WARNING: No points were projected to the image.")
                print("This can happen if the distance threshold is too small.")
                return image, intrinsics, extrinsics

            # Create an empty image
            image = np.zeros((height, width, 3), dtype=np.uint8)

            # Create an empty depth map
            depth_map = np.full((height, width), np.inf, dtype=np.float32)

            # Fill the image with the color values and make sure that the points are within the image boundaries
            # Also take the point with the smallest distance to the camera center
            for i, point in enumerate(points_2d):
                x, y = point[0]
                if 0 <= x < width and 0 <= y < height:
                    if image[int(y), int(x)].any():
                        existing_dist = depth_map[int(y), int(x)]
                        curr_dist = distances[i]
                        if curr_dist < existing_dist:
                            image[int(y), int(x)] = colors[i]
                            depth_map[int(y), int(x)] = curr_dist
                    else:
                        image[int(y), int(x)] = colors[i]
                        depth_map[int(y), int(x)] = distances[i]

            return image, intrinsics, extrinsics

        else:
            # Define the point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255)

            # Visualize the point cloud and save the image
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(visible=True, width=1000, height=700)
            vis.add_geometry(pcd)
            render = vis.get_render_option()
            render.point_size = 1
            render.background_color = np.asarray([0, 0, 0])
            vis.run()
            vis.destroy_window()

            # Wait 5 seconds for the image to be saved
            time.sleep(5)

            # Get the image and camera parameters (find files that starts with Screen)
            dir_path = os.getcwd()
            files = os.listdir(dir_path)
            files = [file for file in files if file.startswith('Screen')]
            image_path = [file for file in files if file.endswith('.png')]
            image_path = sorted(image_path, reverse=True)[0]
            camera_path = [file for file in files if file.endswith('.json')]
            camera_path = sorted(camera_path, reverse=True)[0]

            # Load the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load the camera parameters
            with open(camera_path) as file:
                camera = json.load(file)

            # Convert open3d.geometry.Image to numpy array
            image = np.asarray(image)
            intrinsics = np.asarray(camera['intrinsic']['intrinsic_matrix']).reshape((3, 3)).T
            extrinsics = np.asarray(camera['extrinsic']).reshape((4, 4)).T

            # Delete the files
            for file in files:
                os.remove(file)

            return image, intrinsics, extrinsics


    def image_to_cloud(self, points: np.ndarray, image: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """
        Converts an image to a point cloud.

        :param points: An array of points in the cloud, where each row represents a point.
        :type points: ndarray
        :param image: An image array representing the point cloud, where each pixel contains the RGB color values of the corresponding point in the cloud.
        :type image: ndarray
        :param intrinsics: The intrinsics matrix of the camera.
        :type intrinsics: ndarray (width, height, fx, fy, cx, cy) (6x1)
        :param extrinsics: The extrinsics matrix of the camera.
        :type extrinsics: ndarray (4x4)
        :return: An array of segments' IDs in the cloud, where each row represents the segment's ID of a point.
        :rtype: ndarray
        """

        # Create 4x4 extrinsics matrix
        rotation = extrinsics[:3, :3]
        translation = extrinsics[:3, 3]

        # Separate the points into 3D coordinates and color values
        coords = points[:, :3]
        colors = points[:, 3:6]

        # Project 3D points to 2D image using projectPoints
        points_2d, _ = cv2.projectPoints(coords, rotation, translation, intrinsics, None)

        # Extract segment IDs from the image
        # Give -1 as segment ID if the point is outside the image boundaries
        segment_ids = []
        unique_values = {}
        image = np.asarray(image)

        for i, point in enumerate(points_2d):
            x, y = point[0]
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                rgb = image[int(y), int(x)]
            else:
                rgb = (0, 0, 0)

            if rgb not in unique_values:
                unique_values[rgb] = len(unique_values)

            id = unique_values[rgb]
            segment_ids.append(id)

        return segment_ids