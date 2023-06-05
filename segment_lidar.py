import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

# Assuming your point cloud data is stored in a NumPy array called "point_cloud"
point_cloud = np.array([[x1, y1, z1], [x2, y2, z2], ...])

cloud = pv.PolyData(point_cloud)

bounds = cloud.bounds
x_min, x_max, y_min, y_max, z_min, z_max = bounds
width = int(x_max - x_min)
height = int(y_max - y_min)

# Define the resolution (pixel size) of the raster image
pixel_size = 1.0  # Adjust as desired
image_resolution = (int(width / pixel_size), int(height / pixel_size))

# Create a plotter
plotter = pv.Plotter(off_screen=True)
plotter.add_points(cloud, color="white")  # Add the points to the plotter

# Set the camera to an orthogonal view
plotter.camera_position = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_max + 1]  # Adjust the z position if needed
plotter.camera.SetParallelProjection(True)

# Set the aspect ratio to match the resolution
plotter.set_aspect_ratio(1.0)

# Set the window size
plotter.window_size = image_resolution

# Generate the raster image
plotter.show(auto_close=False)
raster_image = plotter.screenshot()
