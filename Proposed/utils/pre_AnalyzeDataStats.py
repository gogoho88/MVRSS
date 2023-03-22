"""
len of all data: 12666

check & save the statistics (mean, std, min, max) of Carrada datasets
"""
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

path = '/workspace/Dataset/Carrada/Carrada'
path_RAD = '/workspace/Dataset/Carrada_RAD'

with open(os.path.join(path,'light_dataset_frame_oriented.json'), 'r') as fp:
    annotations = json.load(fp)

rdlist_mean = list()
ralist_mean = list()
adlist_mean = list()
rdlist_mean2 = list()
ralist_mean2 = list()
adlist_mean2 = list()
rdlist_min = list()
ralist_min = list()
adlist_min = list()
rdlist_max = list()
ralist_max = list()
adlist_max = list()
radlist_mean = list()
radlist_mean2 = list()
radlist_min = list()
radlist_max = list()

ts = time.time()
for sequence in annotations.keys():
    path_data = os.path.join(path,sequence)

    print(sequence)
    for template in annotations[sequence]:
        rd_matrix = np.load(os.path.join(path_data,'range_doppler_processed',template+'.npy'))
        ra_matrix = np.load(os.path.join(path_data,'range_angle_processed',template+'.npy'))
        ad_matrix = np.load(os.path.join(path_data,'angle_doppler_processed',template+'.npy'))

        rdlist_mean.append(rd_matrix.mean())
        ralist_mean.append(ra_matrix.mean())
        adlist_mean.append(ad_matrix.mean())
        rdlist_mean2.append(np.mean(rd_matrix**2))
        ralist_mean2.append(np.mean(ra_matrix**2))
        adlist_mean2.append(np.mean(ad_matrix**2))
        rdlist_min.append(rd_matrix.min())
        ralist_min.append(ra_matrix.min())
        adlist_min.append(ad_matrix.min())
        rdlist_max.append(rd_matrix.max())
        ralist_max.append(ra_matrix.max())
        adlist_max.append(ad_matrix.max())
te_2D = time.time()-ts

ts = time.time()
for sequence in annotations.keys():
    path_data_RAD = os.path.join(path_RAD,sequence)

    print(sequence+'3D')
    for template in annotations[sequence]:
        rad_matrix = np.load(os.path.join(path_data_RAD,'RAD_numpy',template+'.npy'))

        radlist_mean.append(rad_matrix.mean())
        radlist_mean2.append(np.mean(rad_matrix**2))
        radlist_min.append(rad_matrix.min())
        radlist_max.append(rad_matrix.max())
te_3D = time.time()-ts

# Save
parameter = {}
parameter['rd_stats_preprocessd'] = {
                    "mean": float(np.array(rdlist_mean).mean()),
                    "std": float(np.sqrt(np.array(rdlist_mean2).mean()-np.array(rdlist_mean).mean()**2)),
                    "min_val": float(np.array(rdlist_min).min()),
                    "max_val": float(np.array(rdlist_max).max())
}
parameter['ra_stats_preprocessd'] = {
                    "mean": float(np.array(ralist_mean).mean()),
                    "std": float(np.sqrt(np.array(ralist_mean2).mean()-np.array(ralist_mean).mean()**2)),
                    "min_val": float(np.array(ralist_min).min()),
                    "max_val": float(np.array(ralist_max).max())
}
parameter['ad_stats_preprocessd'] = {
                    "mean": float(np.array(adlist_mean).mean()),
                    "std": float(np.sqrt(np.array(adlist_mean2).mean()-np.array(adlist_mean).mean()**2)),
                    "min_val": float(np.array(adlist_min).min()),
                    "max_val": float(np.array(adlist_max).max())
}
parameter['rad_stats'] = {
                    "mean": float(np.array(radlist_mean).mean()),
                    "std": float(np.sqrt(np.array(radlist_mean2).mean()-np.array(radlist_mean).mean()**2)),
                    "min_val": float(np.array(radlist_min).min()),
                    "max_val": float(np.array(radlist_max).max())
}


save_path = "/workspace/MVRSS/mvrss/config_files/all_stats.json"
with open(save_path, 'w') as f:
    json.dump(parameter, f, indent=4)

        
        


