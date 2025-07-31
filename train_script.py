# Script to train Embedseg model. Use within directory in which EmbedSeg is installed.
import numpy as np
import os
from train_multipledatasets import begin_training
from EmbedSeg.utils.create_dicts import create_dataset_dict, create_model_dict, create_loss_dict, create_configs
import torch
from matplotlib.colors import ListedColormap
import json

center = 'medoid' # 'centroid', 'medoid'

data_dir_list = ["/mnt/home/dkim1/ceph/StyleID/ContentD6_StyleD6/crops/", 
                 "/mnt/home/dkim1/ceph/StyleID/ContentC1_StyleD6/crops/"]
project_name_list = ["ContentD6_StyleD6", "ContentC1_StyleD6"]

data_dir_list_val = ["/mnt/home/dkim1/ceph/StyleID/ContentMO_StyleD6/crops/"] 
project_name_list_val = ["ContentMO_StyleD6"]


if os.path.isfile('/mnt/home/dkim1/StyleID/data_properties.json'):
    with open('/mnt/home/dkim1/StyleID/data_properties.json') as json_file:
        data = json.load(json_file)
        data_type, foreground_weight, n_z, n_y, n_x, pixel_size_z_microns, pixel_size_x_microns = data['data_type'], float(data['foreground_weight']), int(data['n_z']), int(data['n_y']), int(data['n_x']), float(data['pixel_size_z_microns']), float(data['pixel_size_x_microns'])

train_batch_size = 4 

train_size_list = [len(os.listdir(os.path.join(data_dir_list[i], project_name_list[i], 'train', 'images'))) for i in range(len(data_dir_list))]

train_dataset_dict_list = []

for i in range(len(data_dir_list)):

    train_dataset_dict = create_dataset_dict(data_dir = data_dir_list[i], 
                                             project_name = project_name_list[i],  
                                             center = center, 
                                             size = train_size_list[i], 
                                             batch_size = train_batch_size, 
                                             type = 'train',
                                             name = '3d')
    train_dataset_dict_list.append(train_dataset_dict)

val_batch_size = 16
val_size_list = [len(os.listdir(os.path.join(data_dir_list_val[i], project_name_list_val[i], 'val', 'images'))) for i in range(len(data_dir_list_val))]

val_dataset_dict_list = []

for i in range(len(data_dir_list_val)):

    val_dataset_dict = create_dataset_dict(data_dir = data_dir_list_val[i], 
                                             project_name = project_name_list_val[i],  
                                             center = center, 
                                             size = val_size_list[i], 
                                             batch_size = val_batch_size, 
                                             type = 'val',
                                             name = '3d')
    val_dataset_dict_list.append(val_dataset_dict)

input_channels = 1
num_classes = [6, 1] 

model_dict = create_model_dict(input_channels = input_channels,
                              num_classes = num_classes,
                              name = '3d')

loss_dict = create_loss_dict(n_sigma = 3, foreground_weight = foreground_weight)

n_epochs = 400
save_dir = os.path.join('experiment', "demo")
resume_path = None

configs = create_configs(n_epochs = n_epochs,
                         resume_path = resume_path, 
                         save_dir = save_dir, 
                         n_z = n_z,
                         n_y = n_y, 
                         n_x = n_x,
                         anisotropy_factor = pixel_size_z_microns/pixel_size_x_microns, 
                         save_checkpoint_frequency = 5
                         )

begin_training(train_dataset_dict_list, val_dataset_dict_list, model_dict, loss_dict, configs)