import open3d as o3d
import json
import open3d.visualization.rendering as rendering
import numpy as np
import cv2
import os
import tensorflow as tf

IMGENC_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_DATA_DIR = os.path.join(IMGENC_DIR,'train_data')
IMGS_DIR = os.path.join(TRAIN_DATA_DIR,'images')
LABELS_DIR = os.path.join(TRAIN_DATA_DIR,'labels')

def perspective_proj(c_int, c_ext, w_p):
    P_0 = np.zeros((3,4))
    P_0[0,0],P_0[1,1],P_0[2,2] = 1,1,1 
    img_p = c_int@P_0@c_ext@w_p
    return(img_p)


def rot_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0,0],[np.sin(angle), np.cos(angle),0,0],[0,0,1,1],[0,0,0,1]],dtype=np.float64)

def transform_obj(dx,dy,dz,phi,th,psi,obj):
    oriented_bounding_box = obj.get_oriented_bounding_box(robust=True)
    # oriented_bounding_box.color = (0, 1, 0)

    R = obj.get_rotation_matrix_from_xyz((phi, th, psi))

    obj_cent = obj.get_center()
    R_new = R.copy()
    obj_t = np.array([obj_cent[0]+dx,obj_cent[1]+dy,dz],dtype=np.float64)
    T = np.array([np.append(R_new[0],obj_t[0]),np.append(R_new[1],obj_t[1]),np.append(R_new[2],obj_t[2]),[0,0,0,1]])
    # obj.transform(T)
    oriented_bounding_box.translate([obj_t[0],obj_t[1],obj_t[2]])
    oriented_bounding_box.rotate(R,center = [obj_t[0],obj_t[1],obj_t[2]])
    bbox_cord = np.asarray(oriented_bounding_box.get_box_points())
    
    return T, bbox_cord

def gen_imgs_o3d():
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
    yellow.base_color = [1, 0.64, 0.2, 1.0]
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
    object.translate([0,0,0])
    render.setup_camera(intrinsics,extrinsics)
    render.scene.add_geometry("object", object, yellow)
    render.scene.set_background(color = [0.15,0.15,0.2,1])
    render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],200000)
    render.scene.scene.enable_sun_light(True)

    labels_path = 'labels_'+ "2" + '.csv'
    csv_path = os.path.join(LABELS_DIR,labels_path)

    with open(csv_path,"a") as file:
        for i in range(0,100000):
            
            img_name = "frame_{}.jpg".format(i)
            img_savepath =os.path.join(IMGS_DIR,img_name)
            dx = np.random.randint(-500,501)*0.0002
            dy = np.random.randint(-500,501)*0.0002
            dz = np.random.randint(-500,501)*0.0002
            # dx,dy,dz = 0,0,0
            phi = np.random.randint(0,360)*np.pi/180.0
            th = np.random.randint(0,360)*np.pi/180.0
            psi = np.random.randint(0,360)*np.pi/180.0
            # phi, psi= 0,np.pi/8
            # th = i*4*np.pi/180
            T , bbox_cord= transform_obj(dx,dy,dz,phi,th,psi, object)
            # render.scene.add_geometry("or_bb", oriented_bounding_box, red)
            render.scene.set_geometry_transform("object",T)
            img = render.render_to_image()

            img_raw = np.array(img)
            img_proj = perspective_proj(intrinsics.intrinsic_matrix, extrinsics,np.c_[bbox_cord,np.ones(8)].T)
            img_proj_norm = (img_proj/img_proj[2])[:2]
            img_proj_norm = img_proj_norm.astype(np.int16)
            tup_img_proj = list(zip(img_proj_norm[0], img_proj_norm[1]))
            # print(tup_img_proj)
            # for point in tup_img_proj:
            #     cv2.circle(img_raw, point, 3, (255,50,50), thickness=2)
            img_new = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
            img_cords = img_proj_norm.T.ravel()
            
            data = np.concatenate([[img_name],img_cords])
            # np.savetxt(file, [data],delimiter=',',fmt="%s")
            # cv2.imwrite(img_savepath , img_new)
            if i%100==0:
                print((i/100000)*100)
            
            cv2.imshow("image",img_new)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

# for i in range(0,200):
#     dx = np.random.randint(-500,501)*0.0002
#     dy = np.random.randint(-500,501)*0.0002
#     dz = np.random.randint(-500,501)*0.0002
#     phi = np.random.randint(0,360)*np.pi/180.0
#     th = np.random.randint(0,360)*np.pi/180.0
#     psi = np.random.randint(0,360)*np.pi/180.0
#     T , bbox_cord= transform_obj(dx,dy,dz,phi,th,psi, object)
#     # render.scene.add_geometry("or_bb", oriented_bounding_box, red)
#     render.scene.set_geometry_transform("object",T)
#     img = render.render_to_image()

#     img_raw = np.array(img)
#     img_proj = perspective_proj(intrinsics.intrinsic_matrix, extrinsics,np.c_[bbox_cord,np.ones(8)].T)
#     img_proj_norm = (img_proj/img_proj[2])[:2]
#     img_proj_norm = img_proj_norm.astype(np.int16)
#     tup_img_proj = list(zip(img_proj_norm[0], img_proj_norm[1]))
#     print(tup_img_proj)
#     for point in tup_img_proj:
#         cv2.circle(img_raw, point, 3, (255,0,0), thickness=2)
#     img_new = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
#     cv2.imshow("image",img_new)
#     cv2.waitKey(5)
# cv2.destroyAllWindows()

def test_pose_model():
    
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
    yellow.base_color = [1, 0.64, 0.2, 1.0]
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
    object.translate([0,0,0])
    render.setup_camera(intrinsics,extrinsics)
    render.scene.add_geometry("object", object, yellow)
    render.scene.set_background(color = [0.15,0.15,0.2,1])
    render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],200000)
    render.scene.scene.enable_sun_light(True)
    
    model = tf.keras.models.load_model("/home/glsrujan/Documents/personal/raise_robotics/pose_est/cnn_models/3d_pose_90")
    
    for i in range(50):
        # dx = np.random.randint(-500,501)*0.0002
        # dy = np.random.randint(-500,501)*0.0002
        # dz = np.random.randint(-500,501)*0.0002
        # # dx,dy,dz = 0,0,0
        # phi = np.random.randint(0,360)*np.pi/180.0
        # th = np.random.randint(0,360)*np.pi/180.0
        # psi = np.random.randint(0,360)*np.pi/180.0
        # # phi, psi= 0,np.pi/8
        # # th = i*4*np.pi/180
        # T , bbox_cord= transform_obj(dx,dy,dz,phi,th,psi, object)
        # # render.scene.add_geometry("or_bb", oriented_bounding_box, red)
        # render.scene.set_geometry_transform("object",T)
        # img = render.render_to_image()

        # img_raw = np.array(img)
        # disp_img = img_raw.copy()
        
        data = np.loadtxt("/home/glsrujan/Documents/personal/raise_robotics/pose_est/train_data/labels/labels_1.csv",dtype=str,delimiter=",")
        labels = np.array(data[:,1:17],dtype=np.float64)
        idx = np.random.randint(0,len(data))
        bbox_vert_true = labels[idx].reshape((8,2)).astype(np.int16)
        
        img_raw = cv2.imread("/home/glsrujan/Documents/personal/raise_robotics/pose_est/train_data/images/"+data[idx,0])
        disp_img = img_raw.copy()
        
        # img_raw = cv2.cvtColor(img_raw,cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.resize(img_gray,(180,135))
        img_gray = img_gray/255.0
        img_gray = img_gray.astype(np.float32)
        img_gray = np.expand_dims(img_gray,axis=2)
        img_gray = np.expand_dims(img_gray,axis=0)
        
        pred = model.predict(img_gray)
        # pred[:,[0,2,4,6,8,10,12,14]] = pred[:,[0,2,4,6,8,10,12,14]] * 720
        # pred[:,[1,3,5,7,9,11,13,15]] = pred[:,[1,3,5,7,9,11,13,15]] * 540
        
        bbox_vert = pred.reshape((1,2)).astype(np.int16)
        print(bbox_vert)
        
        tup_img_proj = list(zip(bbox_vert[:,0], bbox_vert[:,1]))
        print(tup_img_proj)
        num = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        for point in tup_img_proj:
            cv2.circle(disp_img, point, 3, (255,50,50), thickness=2)
            # cv2.putText(disp_img, "{}".format(num), point, font, 1, (255,50,50), 1, cv2.LINE_AA)
            num+=1
        for point_tru in bbox_vert_true:
            cv2.circle(disp_img, point_tru, 3, (50,255,50), thickness=2)
        # cv2.line(disp_img,tup_img_proj[0],tup_img_proj[1],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[0],tup_img_proj[2],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[0],tup_img_proj[3],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[2],tup_img_proj[7],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[2],tup_img_proj[5],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[4],tup_img_proj[7],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[4],tup_img_proj[6],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[4],tup_img_proj[5],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[6],tup_img_proj[1],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[6],tup_img_proj[3],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[7],tup_img_proj[1],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[5],tup_img_proj[3],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[3],tup_img_proj[4],(50,255,50),thickness=1)
        # cv2.line(disp_img,tup_img_proj[5],tup_img_proj[6],(50,255,50),thickness=1)

        # disp_img = cv2.cvtColor(disp_img,cv2.COLOR_BGR2RGB)
        cv2.imshow("image",disp_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    test_pose_model()

