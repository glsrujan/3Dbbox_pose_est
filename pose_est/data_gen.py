import open3d as o3d
import json
import open3d.visualization.rendering as rendering
import numpy as np
import cv2

def perspective_proj(c_int, c_ext, w_p):
    
    P_0 = np.zeros((3,4))
    P_0[0,0],P_0[1,1],P_0[2,2] = 1,1,1 
    img_p = c_int@P_0@c_ext@w_p
    
    return(img_p)


def rot_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0,0],[np.sin(angle), np.cos(angle),0,0],[0,0,1,0],[0,0,0,1]],dtype=np.float64)

object = o3d.io.read_triangle_mesh("/home/glsrujan/Documents/personal/raise_robotics/Perception Take Home-20230819T194332Z-001/Perception Take Home/scissors.obj")
object.compute_vertex_normals()
# object.translate([0,0,1.5])
oriented_bounding_box = object.get_oriented_bounding_box(robust=True)
oriented_bounding_box.color = (0, 1, 0)
bbox_cord = np.asarray(oriented_bounding_box.get_box_points())
print("init_bb",bbox_cord)

R = object.get_rotation_matrix_from_xyz((0, 0, 0))
# object.rotate(R, center = (0,0,1.5))
obj_cent = object.get_center()
R_new = R
T = np.array([np.append(R_new[0],obj_cent[0]),np.append(R_new[1],obj_cent[1]),np.append(R_new[2],obj_cent[2]+1.5),[0,0,0,1]])
object.transform(T)
bbox_cord_h = np.c_[bbox_cord,np.ones(8)]
bbox_cord_rotated = np.linalg.inv(T)@ bbox_cord_h.T
bbox_cord_rotated = bbox_cord_rotated[:3].T

print("bb_rotated_est",bbox_cord_rotated )

axis_aligned_bounding_box = object.get_axis_aligned_bounding_box()
# print(np.asarray(axis_aligned_bounding_box.get_box_points()))
print("Center Coordinates: ", axis_aligned_bounding_box.get_center())

oriented_bounding_box = object.get_oriented_bounding_box(robust=True)
oriented_bounding_box.color = (0, 1, 0)
bbox_cord = np.asarray(oriented_bounding_box.get_box_points())
print("bbox_rotated",bbox_cord)
print("oriented_bounding_box Center Coordinates: ", oriented_bounding_box.get_center())
print("object Center Coordinates: ", object.get_center())
# print(type(object))
sphere = o3d.geometry.TriangleMesh.create_sphere(.01)
sphere.compute_vertex_normals()
sphere.translate(object.get_center())

sphere_2 = o3d.geometry.TriangleMesh.create_sphere(.01)
sphere_2.compute_vertex_normals()
sphere_2.translate(bbox_cord_rotated[0])

cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size  = 0.1,origin = np.array([0,0,0], dtype=np.float64))

with open("/home/glsrujan/Documents/personal/raise_robotics/Perception Take Home-20230819T194332Z-001/Perception Take Home/camera.json", "r") as json_file:
    camera_intrinsics = json.load(json_file)
intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.set_intrinsics(
    camera_intrinsics["width"],
    camera_intrinsics["height"],
    camera_intrinsics["fx"]/camera_intrinsics["depth_scale"],
    camera_intrinsics["fy"]/camera_intrinsics["depth_scale"],
    camera_intrinsics["cx"],
    camera_intrinsics["cy"]
)

# print("Focal Length", intrinsics.get_focal_length())
extrinsics = rot_z(np.pi)
# print("ext:", extrinsics)
# print("inv_ext:", np.linalg.inv(extrinsics))
render = rendering.OffscreenRenderer(720, 540)

yellow = rendering.MaterialRecord()
yellow.base_color = [1.0, 0.75, 0.0, 1.0]
yellow.shader = "defaultLit"

green = rendering.MaterialRecord()
green.base_color = [0.0, 0.5, 0.0, 1.0]
green.shader = "defaultLit"

red = rendering.MaterialRecord()
red.base_color = [0.5, 0.0, 0.0, 1.0]
# red.shader = "defaultLit"

render.scene.add_geometry("object", object, yellow)
# render.scene.add_geometry("ax_bb", axis_aligned_bounding_box, green)
render.scene.add_geometry("or_bb", oriented_bounding_box, red)
render.scene.add_geometry("object_cent", sphere, green)
render.scene.add_geometry("bbox_1", sphere_2, green)
render.setup_camera(intrinsics,extrinsics)
# render.setup_camera(60.0, [0, 0, 0], [-10, 0, 0], [0, 0, 1])
# render.setup_camera(intrinsics.intrinsic_matrix,extrinsics,camera_intrinsics["width"], camera_intrinsics["height"])
# render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],75000)
# render.scene.scene.enable_sun_light(True)
# render.scene.show_axes(True)

img = render.render_to_image()
# print("type of img", type(img))
# quit()
img_raw = np.array(img)
for i in range(8):
    img_new = img_raw.copy()
    img_point = perspective_proj(intrinsics.intrinsic_matrix, extrinsics,np.append(bbox_cord_rotated[i],1))
    img_point = (img_point/img_point[2])
    print("img_point_{}".format(i),img_point)
    print("Saving image at test.png")
    # cv2.circle(img_new, (int(img_point[0]),int(img_point[1])), 5, (255,0,0), thickness=5)
    img_new = cv2.cvtColor(img_new,cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_{}.png".format(i),img_new)
    # o3d.io.write_image("test.png", img, 9)

# o3d.visualization.draw(
#         [object, sphere, oriented_bounding_box,cf])
img = cv2.imread("test.png")
print(img.shape)

