# API


### _class_ segment_lidar.samlidar.SamLidar(ckpt_path: str, algorithm: str = 'segment-geospatial', model_type: str = 'vit_h', resolution: float = 0.25, device: str = 'cuda:0', sam_kwargs: bool = False)
Bases: `object`


#### csf(points: ndarray, class_threshold: float = 0.5, cloth_resolution: float = 0.2, iterations: int = 500, slope_smooth: bool = False)
Applies the CSF (Cloth Simulation Filter) algorithm to filter ground points in a point cloud.


* **Parameters**

    
    * **points** (*np.ndarray*) – The input point cloud as a NumPy array, where each row represents a point with x, y, z coordinates.


    * **class_threshold** (*float**, **optional*) – The threshold value for classifying points as ground/non-ground, defaults to 0.5.


    * **cloth_resolution** (*float**, **optional*) – The resolution value for cloth simulation, defaults to 0.2.


    * **iterations** (*int**, **optional*) – The number of iterations for the CSF algorithm, defaults to 500.


    * **slope_smooth** (*bool**, **optional*) – A boolean indicating whether to enable slope smoothing, defaults to False.



* **Returns**

    A tuple containing three arrays: the filtered point cloud, non-ground (filtered) points indinces and ground points indices.



* **Return type**

    tuple[np.ndarray, np.ndarray, np.ndarray]



#### read(path: str, classification: int | None = None)
Reads a point cloud from a file and returns it as a NumPy array.


* **Parameters**

    
    * **path** (*str*) – The path to the input file.


    * **classification** (*int**, **optional*) – The optional classification value to filter the point cloud, defaults to None.



* **Returns**

    The point cloud as a NumPy array.



* **Return type**

    np.ndarray



* **Raises**

    **ValueError** – If the input file format is not supported.



#### segment(points: ndarray, text_prompt: str | None = None, image_path: str = 'raster.tif', labels_path: str = 'labeled.tif')
Segments a point cloud based on the provided parameters and returns the segment IDs, original image, and segmented image.


* **Parameters**

    
    * **points** (*np.ndarray*) – The point cloud data as a NumPy array.


    * **text_prompt** (*str*) – Optional text prompt for segment generation, defaults to None.


    * **image_path** (*str*) – Path to the input raster image, defaults to ‘raster.tif’.


    * **labels_path** (*str*) – Path to save the labeled output image, defaults to ‘labeled.tif’.



* **Returns**

    A tuple containing the segment IDs, segmented image, and RGB image.



* **Return type**

    tuple[np.ndarray, np.ndarray, np.ndarray]



#### write(points: ndarray, segment_ids: ndarray, non_ground: ndarray | None = None, ground: ndarray | None = None, save_path: str = 'segmented.las')
Writes the segmented point cloud data to a LAS/LAZ file.


* **Parameters**

    
    * **points** (*np.ndarray*) – The input point cloud data as a NumPy array, where each row represents a point with x, y, z coordinates.


    * **segment_ids** (*np.ndarray*) – The segment IDs corresponding to each point in the point cloud.


    * **non_ground** (*np.ndarray**, **optional*) – Optional array of indices for non-ground points in the original point cloud (default: None).


    * **ground** (*np.ndarray**, **optional*) – Optional array of indices for ground points in the original point cloud (default: None).


    * **save_path** (*str**, **optional*) – The path to save the segmented LAS/LAZ file (default: ‘segmented.las’).



* **Returns**

    None



### segment_lidar.samlidar.cloud_to_image(points: ndarray, minx: float, maxx: float, miny: float, maxy: float, resolution: float)
Converts a point cloud to an image.


* **Parameters**

    
    * **points** (*ndarray*) – An array of points in the cloud, where each row represents a point.
    The array shape can be (N, 3) or (N, 6).
    If the shape is (N, 3), each point is assumed to have white color (255, 255, 255).
    If the shape is (N, 6), the last three columns represent the RGB color values for each point.


    * **minx** (*float*) – The minimum x-coordinate value of the cloud bounding box.


    * **maxx** (*float*) – The maximum x-coordinate value of the cloud bounding box.


    * **miny** (*float*) – The minimum y-coordinate value of the cloud bounding box.


    * **maxy** (*float*) – The maximum y-coordinate value of the cloud bounding box.


    * **resolution** (*float*) – The resolution of the image in units per pixel.



* **Returns**

    An image array representing the point cloud, where each pixel contains the RGB color values
    of the corresponding point in the cloud.



* **Return type**

    ndarray



* **Raises**

    **ValueError** – If the shape of the points array is not valid or if any parameter is invalid.



### segment_lidar.samlidar.image_to_cloud(points: ndarray, minx: float, maxy: float, image: ndarray, resolution: float)
Converts an image to a point cloud with segment IDs.


* **Parameters**

    
    * **points** – An array of points representing the cloud, where each row represents a point.
    The array shape is (N, 3) where each point contains x, y, and z coordinates.


    * **minx** – The minimum x-coordinate value of the cloud bounding box.


    * **maxy** – The maximum y-coordinate value of the cloud bounding box.


    * **image** – The image array representing the input image.


    * **resolution** – The resolution of the image in units per pixel.



* **Returns**

    A list of segment IDs for each point in the cloud. The segment ID represents the color segment
    in the input image that the corresponding point belongs to.



* **Return type**

    List[int]



### _class_ segment_lidar.samlidar.mask(crop_n_layers: int = 1, crop_n_points_downscale_factor: int = 1, min_mask_region_area: int = 1000, points_per_side: int = 32, pred_iou_thresh: float = 0.9, stability_score_thresh: float = 0.92)
Bases: `object`
