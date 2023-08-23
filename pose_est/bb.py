import open3d as o3d
import numpy as np

if __name__ == "__main__":
    # sample_ply_data = o3d.data.PLYPointCloud()
    # pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    mesh = o3d.io.read_triangle_mesh("/home/glsrujan/Documents/personal/raise_robotics/Perception Take Home-20230819T194332Z-001/Perception Take Home/scissors.obj")
    axis_aligned_bounding_box = mesh.get_axis_aligned_bounding_box()
    print(np.asarray(axis_aligned_bounding_box.get_box_points()))
    print("Center Coordinates: ", axis_aligned_bounding_box.get_center())
    sphere = o3d.geometry.TriangleMesh.create_sphere(.002)
    sphere.compute_vertex_normals()
    sphere.translate(axis_aligned_bounding_box.get_center())
    # Flip it, otherwise the pointcloud will be upside down.
    # mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(mesh)
    # axis_aligned_bounding_box = mesh.get_axis_aligned_bounding_box()
    # axis_aligned_bounding_box.color = (1, 0, 0)
    # oriented_bounding_box = mesh.get_oriented_bounding_box()
    # oriented_bounding_box.color = (0, 1, 0)
    print(
        "Displaying axis_aligned_bounding_box in red and oriented bounding box in green ..."
    )
    
    print(np.asarray(axis_aligned_bounding_box.get_box_points()))
    o3d.visualization.draw(
        [mesh, sphere, axis_aligned_bounding_box])