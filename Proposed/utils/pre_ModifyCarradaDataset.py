import os
import json
import time
import pickle
import shutil

import numpy as np

# start_time = time.time()
# with open('/workspace/RAD_temp.pickle','rb') as f:
#     rad_temp = pickle.load(f)
# end_time = time.time()-start_time

path = '/data/Carrada'

flag_DB = True

with open(os.path.join(path,'data_seq_ref.json'), 'r') as fp:
    data_seq_ref = json.load(fp)
with open(os.path.join(path,'light_dataset_frame_oriented.json'), 'r') as fp:
    annotations = json.load(fp)

for sequence in annotations.keys():
    save_path_radmat = os.path.join('/data/datasets_master/Carrada_RAD/',sequence,'mod_RAD_numpy')
    save_path_annot = os.path.join(path,sequence,'mod_annotations','dense')
    os.makedirs(save_path_radmat,exist_ok=True)
    os.makedirs(save_path_annot,exist_ok=True)
    if os.path.exists(save_path_radmat):
        shutil.rmtree(save_path_radmat)
    if os.path.exists(save_path_annot):
        shutil.rmtree(save_path_annot)
    print(sequence)
    for template in annotations[sequence]:
        rd_mask = np.load(os.path.join(path,sequence,'annotations','dense',template,'range_doppler.npy'))
        ra_mask = np.load(os.path.join(path,sequence,'annotations','dense',template,'range_angle.npy'))
        rad_mat = np.load(os.path.join('/data/datasets_master/Carrada_RAD/',sequence,'RAD_numpy',template+'.npy'))
        
        # Preprocess
        if flag_DB==True:
            rad_mat = np.log10(rad_mat**2) 


        # Save as pickle type
        os.makedirs(os.path.join(save_path_annot,template))
        with open(os.path.join(save_path_annot,template,'range_doppler.pickle'),'wb') as f:
            pickle.dump(rd_mask,f)
        with open(os.path.join(save_path_annot,template,'range_angle.pickle'),'wb') as f:
            pickle.dump(ra_mask,f)
        with open(os.path.join(save_path_radmat,template+'.pickle'),'wb') as f:
            pickle.dump(rad_mat,f)
        
        

        


a = 1
