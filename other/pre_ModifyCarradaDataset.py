import os
import json
import time
import pickle
import shutil
from turtle import down

import numpy as np

from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
# start_time = time.time()
# with open('/workspace/RAD_temp.pickle','rb') as f:
#     rad_temp = pickle.load(f)
# end_time = time.time()-start_time

path = '/data/Carrada'

flag_DB = True
downsampling = 2
flag_save = True

with open(os.path.join(path,'data_seq_ref.json'), 'r') as fp:
    data_seq_ref = json.load(fp)
with open(os.path.join(path,'light_dataset_frame_oriented.json'), 'r') as fp:
    annotations = json.load(fp)

for sequence in annotations.keys():
    save_path_radmat = os.path.join('/data/datasets_master/Carrada_RAD/',sequence,'mod_RAD_numpy')
    save_path_annot = os.path.join(path,sequence,'mod_annotations','dense')
    os.makedirs(save_path_radmat,exist_ok=True)
    os.makedirs(save_path_annot,exist_ok=True)
    # if os.path.exists(save_path_radmat):
    #     shutil.rmtree(save_path_radmat)
    # if os.path.exists(save_path_annot):
    #     shutil.rmtree(save_path_annot)
    print(sequence)
    for template in annotations[sequence]:
        print(template)
        rd_mask = np.load(os.path.join(path,sequence,'annotations','dense',template,'range_doppler.npy'))
        ra_mask = np.load(os.path.join(path,sequence,'annotations','dense',template,'range_angle.npy'))
        rad_mat = np.load(os.path.join('/data/datasets_master/Carrada_RAD/',sequence,'RAD_numpy',template+'.npy'))
        
        # Preprocess
        if flag_DB==True:                       # linear -> DB
            rad_mat = np.log10(rad_mat**2) 
        if downsampling>0:             # downsample or interpolation
            # RAD
            x = np.linspace(0,rad_mat.shape[0]-1,rad_mat.shape[0])
            y = np.linspace(0,rad_mat.shape[1]-1,rad_mat.shape[1])
            z = np.linspace(0,rad_mat.shape[2]-1,rad_mat.shape[2])
            f_interp3d = RegularGridInterpolator((x,y,z), rad_mat, method='linear')
            xi = np.linspace(0,rad_mat.shape[0]-1,rad_mat.shape[0]//downsampling)
            yi = np.linspace(0,rad_mat.shape[1]-1,rad_mat.shape[1]//downsampling)
            zi = np.linspace(0,rad_mat.shape[2]-1,rad_mat.shape[2]//downsampling)
            X,Y,Z = np.meshgrid(xi,yi,zi)
            rad_mat_resize = f_interp3d((Y,X,Z))    # 이상하게 이렇게 해야 원래랑 같아짐..
            # Mask (RD, RA)
            rd_mask_resize = rd_mask[:,::downsampling,::downsampling]
            ra_mask_resize = ra_mask[:,::downsampling,::downsampling]
        

        # Save as pickle type
        if flag_save:
            os.makedirs(os.path.join(save_path_annot,template),exist_ok=True)
            np.save(os.path.join(save_path_annot,template,'range_doppler.npy'), rd_mask_resize)
            np.save(os.path.join(save_path_annot,template,'range_angle.npy'), ra_mask_resize)
            np.save(os.path.join(save_path_radmat,template+'.npy'), rad_mat_resize)
            
        

a = 1

