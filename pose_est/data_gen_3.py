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

def transform_obj(dx,dy,dz,phi,th,psi,obj):
    oriented_bounding_box = obj.get_oriented_bounding_box(robust=True)
    oriented_bounding_box.color = (0, 1, 0)

    R = obj.get_rotation_matrix_from_xyz((phi, th, psi))

    obj_cent = obj.get_center()
    R_new = R.copy()
    obj_t = np.array([obj_cent[0]+dx,obj_cent[1]+dy,1.5+dz],dtype=np.float64)
    T = np.array([np.append(R_new[0],obj_t[0]),np.append(R_new[1],obj_t[1]),np.append(R_new[2],obj_t[2]),[0,0,0,1]])
    # obj.transform(T)
    
    oriented_bounding_box.translate([obj_t[0],obj_t[1],obj_t[2]])
    oriented_bounding_box.rotate(R,center = [obj_t[0],obj_t[1],obj_t[2]])
    bbox_cord = np.asarray(oriented_bounding_box.get_box_points())
    
    return obj, bbox_cord , oriented_bounding_box,T

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

yellow = rendering.MaterialRecord()
yellow.base_color = [1.0, 0.75, 0.0, 1.0]
yellow.shader = "defaultLit"

green = rendering.MaterialRecord()
green.base_color = [0.0, 0.5, 0.0, 1.0]
green.shader = "defaultLit"

red = rendering.MaterialRecord()
red.base_color = [0.5, 0.0, 0.0, 1.0]

object = o3d.io.read_triangle_mesh("/home/glsrujan/Documents/personal/raise_robotics/Perception Take Home-20230819T194332Z-001/Perception Take Home/scissors.obj")
object.compute_vertex_normals()


# print(oriented_bounding_box.get_center())
render = rendering.OffscreenRenderer(720, 540)

obj,bbox_cord, oriented_bounding_box,T = transform_obj(0.25,0,0,0,np.pi/4,np.pi/8, object)
render.scene.add_geometry("object", obj, yellow)
render.scene.add_geometry("or_bb", oriented_bounding_box, red)
render.scene.set_geometry_transform("object",T)
render.setup_camera(intrinsics,extrinsics)
# render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],75000)
# render.scene.scene.enable_sun_light(True)
# render.scene.show_axes(True)

img = render.render_to_image()

img_raw = np.array(img)
img_proj = perspective_proj(intrinsics.intrinsic_matrix, extrinsics,np.c_[bbox_cord,np.ones(8)].T)
img_proj_norm = (img_proj/img_proj[2])[:2]
img_proj_norm = img_proj_norm.astype(np.int16)
tup_img_proj = list(zip(img_proj_norm[0], img_proj_norm[1]))
print(tup_img_proj)
for point in tup_img_proj:
    cv2.circle(img_raw, point, 3, (255,0,0), thickness=2)
img_new = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
cv2.imshow("image",img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()

obj,bbox_cord, oriented_bounding_box,T = transform_obj(0,0,0,0,np.pi/4,np.pi/8, object)
render.scene.set_geometry_transform("object",T)

print(render.scene.has_geometry("object"))
render.setup_camera(intrinsics,extrinsics)
# render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],75000)
# render.scene.scene.enable_sun_light(True)
# render.scene.show_axes(True)

img = render.render_to_image()

img_raw = np.array(img)
img_proj = perspective_proj(intrinsics.intrinsic_matrix, extrinsics,np.c_[bbox_cord,np.ones(8)].T)
img_proj_norm = (img_proj/img_proj[2])[:2]
img_proj_norm = img_proj_norm.astype(np.int16)
tup_img_proj = list(zip(img_proj_norm[0], img_proj_norm[1]))
print(tup_img_proj)
for point in tup_img_proj:
    cv2.circle(img_raw, point, 3, (255,0,0), thickness=2)
img_new = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
cv2.imshow("image",img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
# for i in range(8):
#     img_new = img_raw.copy()
#     img_point = perspective_proj(intrinsics.intrinsic_matrix, extrinsics,np.append(bbox_cord[i],1))
#     img_point = (img_point/img_point[2])
#     print("img_point_{}".format(i),img_point)
#     print("Saving image at test.png")
#     cv2.circle(img_new, (int(img_point[0]),int(img_point[1])), 5, (255,0,0), thickness=5)
#     img_new = cv2.cvtColor(img_new,cv2.COLOR_BGR2RGB)
#     cv2.imwrite("test_{}.png".format(i),img_new)

obj,bbox_cord, oriented_bounding_box,T = transform_obj(0,0.25,0,0,np.pi/4,np.pi/8, object)
render.scene.set_geometry_transform("object",T)

print(render.scene.has_geometry("object"))
render.setup_camera(intrinsics,extrinsics)
# render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],75000)
# render.scene.scene.enable_sun_light(True)
# render.scene.show_axes(True)

img = render.render_to_image()

img_raw = np.array(img)
img_proj = perspective_proj(intrinsics.intrinsic_matrix, extrinsics,np.c_[bbox_cord,np.ones(8)].T)
img_proj_norm = (img_proj/img_proj[2])[:2]
img_proj_norm = img_proj_norm.astype(np.int16)
tup_img_proj = list(zip(img_proj_norm[0], img_proj_norm[1]))
print(tup_img_proj)
for point in tup_img_proj:
    cv2.circle(img_raw, point, 3, (255,0,0), thickness=2)
img_new = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
cv2.imshow("image",img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
# for i in range(8):
#     img_new = img_raw.copy()
#     img_point = perspective_proj(intrinsics.intrinsic_matrix, extrinsics,np.append(bbox_cord[i],1))
#     img_point = (img_point/img_point[2])
#     print("img_point_{}".format(i),img_point)
#     print("Saving image at test.png")
#     cv2.circle(img_new, (int(img_point[0]),int(img_point[1])), 5, (255,0,0), thickness=5)
#     img_new = cv2.cvtColor(img_new,cv2.COLOR_BGR2RGB)
#     cv2.imwrite("test_{}.png".format(i),img_new)