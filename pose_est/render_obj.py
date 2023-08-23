import open3d as o3d
import json
import numpy as np

# Load camera intrinsic parameters from JSON file
with open("/home/glsrujan/Documents/personal/raise_robotics/Perception Take Home-20230819T194332Z-001/Perception Take Home/camera.json", "r") as json_file:
    camera_intrinsics = json.load(json_file)

# Create a PinholeCameraIntrinsic object
cam_para = o3d.camera.PinholeCameraParameters()
intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.set_intrinsics(
    camera_intrinsics["width"],
    camera_intrinsics["height"],
    camera_intrinsics["fx"],
    camera_intrinsics["fy"],
    camera_intrinsics["cx"],
    camera_intrinsics["cy"]
)
extrinsics = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,20],[0,0,0,1]])
cam_para.intrinsic = intrinsics
cam_para.extrinsic = extrinsics
print(cam_para.intrinsic.intrinsic_matrix)
# quit()
# Load a mesh from .obj file
mesh = o3d.io.read_triangle_mesh("/home/glsrujan/Documents/personal/raise_robotics/Perception Take Home-20230819T194332Z-001/Perception Take Home/scissors.obj")



# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window(width=camera_intrinsics["width"], height=camera_intrinsics["height"])

# Set camera intrinsic parameters
ctr = vis.get_view_control()
ctr.convert_from_pinhole_camera_parameters(cam_para)

# Add the mesh to the scene
vis.add_geometry(mesh)
vis.run()
# Get a snapshot of the rendered scene
image = vis.capture_screen_float_buffer()

# Clean up visualization
vis.destroy_window()

# Save the image
o3d.io.write_image("rendered_image.png", image)

print("Image rendered and saved.")