"""
@author: Srujan Gowdru Lingaraju
@email: gowdr002@umn.edu
@date: 04/28/2023

Summary:
function main, depending on the cofigs from img_enc_config.yaml generates images and trains CNN.
"""

import hydra
import os
from pathlib import Path
import tensorflow as tf
import cv2
import numpy as np
from img_encoder import image_encoder
# from datagen_img import gen_images , LABELS_DIR, IMGS_DIR
from data_gen_4 import LABELS_DIR, IMGS_DIR

TB_DIR_PATH = os.path.join(os.path.dirname(__file__), 'tensorboard_logs')
CNN_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'cnn_models')

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

@hydra.main(version_base=None, config_path=".", config_name="img_enc_config")
def main(cfg):
    print(cfg)
    # Generate training Images for CNN
    # if cfg.gen_train_imgs:
    #     gen_images(cfg)
    
    # Train CNN
    if cfg.train_cnn:
        assert os.path.isdir(IMGS_DIR) and os.path.isdir(LABELS_DIR), "Training Images not found"
        train_imgenc(cfg)


def pre_proc(img_path , rgb=False):
    # print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(160,120))
    img = img/255.0
    img = img.astype(np.float32)
    if not rgb:
        img = np.expand_dims(img,axis=2)
    return img

def train_imgenc(cfg):
    batch_size = cfg.batch_size_cnn
    # if cfg.get_latest_data:
    #     latestdir_cnt = 0
    #     train_data_name = cfg.task +"_"+ str(cfg.train_eps)+"_"+ str(cfg.seed)
    #     while os.path.isdir(os.path.join(IMGS_DIR,train_data_name+f'_{latestdir_cnt}')):
    #     # If subdirectory_name already exists, increment the counter and update the subdirectory_name      
    #         latestdir_cnt += 1
    #     train_data_name +=f'_{latestdir_cnt-1}'
    # else:
    #     train_data_name = cfg.task +"_"+ str(cfg.train_eps)+"_"+ str(cfg.seed)+f'_{cfg.folder_number}'
        
        
    # img_dir = os.path.join(IMGS_DIR,train_data_name+'/')
    img_dir = IMGS_DIR+"/"
    filename=os.path.join(LABELS_DIR,"labels_"+"1"+'.csv')
    data = np.loadtxt(filename,dtype=str,delimiter=",")
    labels = np.array(data[:,1:17],dtype=np.float64)
    labels[:,[0,2,4,6,8,10,12,14]] = labels[:,[0,2,4,6,8,10,12,14]] / 720
    labels[:,[1,3,5,7,9,11,13,15]] = labels[:,[1,3,5,7,9,11,13,15]] / 540

    # ori = np.array(data[:,10],dtype=np.float64)
    # labels = np.c_[obj_pos,ori]
    # print(labels.shape)
    # quit()
    img_name = data[:,0]

    idx = np.random.permutation(len(img_name))
    img_name_shuf = img_name[idx]
    labels_shuf = labels[idx]

    img_name_bat = np.array_split(img_name_shuf,len(img_name_shuf)//batch_size+1)
    labels_bat = np.array_split(labels_shuf,len(labels_shuf)//batch_size+1,axis=0)
    
    # Tensorboard Configs
    Path(os.path.join(TB_DIR_PATH,"1")).mkdir(parents=True, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(TB_DIR_PATH,"1"),update_freq='epoch')

    #!! Script to test the model performance
    # model = tf.keras.models.load_model("/home/glsrujan/Documents/CHOICE_Lab/openai_gym/dVRK_gym/dex/res_models/enc_mod_8")

    # for i in range(100):
    #     img = pre_proc(img_dir+img_name_bat[i][0])
    #     img = np.expand_dims(img,axis=0)
    #     pred = model.predict(img)

    #     print(img_name_bat[i][0])
    #     print(pred)
    #     print(labels_bat[i][0])
    #     print("Error:",np.linalg.norm(pred[0]-labels_bat[i][0]))
    # print(len(img_name_bat))
    # #|| Load Images and Train model
    model = image_encoder(output_layer=16,rgb=False)
    for j in range(len(img_name_bat)):

        print("Training Batch {} Progress {}%".format(j,(j/len(img_name_bat))*100))
        img_bat = [pre_proc(img_dir+i,rgb=True) for i in img_name_bat[j]]
        # x_val_img = [cv2.imread(val_img_dir+k) for k in val_img_name[0:100]]
        img_bat = np.stack(img_bat)
        # print(img_bat.shape)
        model.fit(img_bat,labels_bat[j],epochs = cfg.cnn_epochs,verbose=1,callbacks=[tensorboard_callback])
    Path(os.path.join(CNN_MODEL_DIR,"3d_pose")).mkdir(parents=True, exist_ok=True)
    model.save(os.path.join(CNN_MODEL_DIR,"3d_pose"))

if __name__ == "__main__":
    main()