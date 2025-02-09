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

path = '/data/Carrada'
path_RAD = '/data/datasets_master/Carrada_RAD'

with open(os.path.join(path,'light_dataset_frame_oriented.json'), 'r') as fp:
    annotations = json.load(fp)

#2D
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
# 3D (RAD)
radlist_mean = list()
radlist_mean2 = list()
radlist_min = list()
radlist_max = list()
# 3D (RAD_downsampled)
radmodlist_mean = list()
radmodlist_mean2 = list()
radmodlist_min = list()
radmodlist_max = list()
radmod_rdlist_mean = list()
radmod_ralist_mean = list()
radmod_adlist_mean = list()
radmod_rdlist_mean2 = list()
radmod_ralist_mean2 = list()
radmod_adlist_mean2 = list()
radmod_rdlist_min = list()
radmod_ralist_min = list()
radmod_adlist_min = list()
radmod_rdlist_max = list()
radmod_ralist_max = list()
radmod_adlist_max = list()
# 3D (RAD_downsampled2)
radmod2list_mean = list()
radmod2list_mean2 = list()
radmod2list_min = list()
radmod2list_max = list()
radmod2_rdlist_mean = list()
radmod2_ralist_mean = list()
radmod2_adlist_mean = list()
radmod2_rdlist_mean2 = list()
radmod2_ralist_mean2 = list()
radmod2_adlist_mean2 = list()
radmod2_rdlist_min = list()
radmod2_ralist_min = list()
radmod2_adlist_min = list()
radmod2_rdlist_max = list()
radmod2_ralist_max = list()
radmod2_adlist_max = list()
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
        # Must be updated later ###
        ###

        radlist_mean.append(rad_matrix.mean())
        radlist_mean2.append(np.mean(rad_matrix**2))
        radlist_min.append(rad_matrix.min())
        radlist_max.append(rad_matrix.max())
te_3D = time.time()-ts

ts = time.time()
for sequence in annotations.keys():
    path_data_RAD = os.path.join(path_RAD,sequence)

    print(sequence+'3D_mod')
    for template in annotations[sequence]:
        rad_matrix_mod = np.load(os.path.join(path_data_RAD,'mod_RAD_numpy',template+'.npy'))
        radmod_rd_matrix = rad_matrix_mod.mean(1)
        radmod_ra_matrix = rad_matrix_mod.mean(2)
        radmod_ad_matrix = rad_matrix_mod.mean(0)

        radmodlist_mean.append(rad_matrix_mod.mean())
        radmodlist_mean2.append(np.mean(rad_matrix_mod**2))
        radmodlist_min.append(rad_matrix_mod.min())
        radmodlist_max.append(rad_matrix_mod.max())
        radmod_rdlist_mean.append(radmod_rd_matrix.mean())
        radmod_ralist_mean.append(radmod_ra_matrix.mean())
        radmod_adlist_mean.append(radmod_ad_matrix.mean())
        radmod_rdlist_mean2.append(np.mean(radmod_rd_matrix**2))
        radmod_ralist_mean2.append(np.mean(radmod_ra_matrix**2))
        radmod_adlist_mean2.append(np.mean(radmod_ad_matrix**2))
        radmod_rdlist_min.append(radmod_rd_matrix.min())
        radmod_ralist_min.append(radmod_ra_matrix.min())
        radmod_adlist_min.append(radmod_ad_matrix.min())
        radmod_rdlist_max.append(radmod_rd_matrix.max())
        radmod_ralist_max.append(radmod_ra_matrix.max())
        radmod_adlist_max.append(radmod_ad_matrix.max())
te_3D_mod = time.time()-ts

ts = time.time()
for sequence in annotations.keys():
    path_data_RAD = os.path.join(path_RAD,sequence)

    print(sequence+'3D_mod')
    for template in annotations[sequence]:
        rad_matrix_mod = np.load(os.path.join(path_data_RAD,'mod2_RAD_numpy',template+'.npy'))
        radmod_rd_matrix = rad_matrix_mod.mean(1)
        radmod_ra_matrix = rad_matrix_mod.mean(2)
        radmod_ad_matrix = rad_matrix_mod.mean(0)

        radmod2list_mean.append(rad_matrix_mod.mean())
        radmod2list_mean2.append(np.mean(rad_matrix_mod**2))
        radmod2list_min.append(rad_matrix_mod.min())
        radmod2list_max.append(rad_matrix_mod.max())
        radmod2_rdlist_mean.append(radmod_rd_matrix.mean())
        radmod2_ralist_mean.append(radmod_ra_matrix.mean())
        radmod2_adlist_mean.append(radmod_ad_matrix.mean())
        radmod2_rdlist_mean2.append(np.mean(radmod_rd_matrix**2))
        radmod2_ralist_mean2.append(np.mean(radmod_ra_matrix**2))
        radmod2_adlist_mean2.append(np.mean(radmod_ad_matrix**2))
        radmod2_rdlist_min.append(radmod_rd_matrix.min())
        radmod2_ralist_min.append(radmod_ra_matrix.min())
        radmod2_adlist_min.append(radmod_ad_matrix.min())
        radmod2_rdlist_max.append(radmod_rd_matrix.max())
        radmod2_ralist_max.append(radmod_ra_matrix.max())
        radmod2_adlist_max.append(radmod_ad_matrix.max())
te_3D_mod2 = time.time()-ts

# Save
parameter = {}
parameter['2D'] = {
            "rd_stats_preprocessd": {
                    "mean": float(np.array(rdlist_mean).mean()),
                    "std": float(np.sqrt(np.array(rdlist_mean2).mean()-np.array(rdlist_mean).mean()**2)),
                    "min_val": float(np.array(rdlist_min).min()),
                    "max_val": float(np.array(rdlist_max).max())
                    },
            "ra_stats_preprocessd": {
                    "mean": float(np.array(ralist_mean).mean()),
                    "std": float(np.sqrt(np.array(ralist_mean2).mean()-np.array(ralist_mean).mean()**2)),
                    "min_val": float(np.array(ralist_min).min()),
                    "max_val": float(np.array(ralist_max).max())
                    },
            "ad_stats_preprocessd": {
                    "mean": float(np.array(adlist_mean).mean()),
                    "std": float(np.sqrt(np.array(adlist_mean2).mean()-np.array(adlist_mean).mean()**2)),
                    "min_val": float(np.array(adlist_min).min()),
                    "max_val": float(np.array(adlist_max).max())
                    }
}
parameter['rad_stats'] = {
                    "mean": float(np.array(radlist_mean).mean()),
                    "std": float(np.sqrt(np.array(radlist_mean2).mean()-np.array(radlist_mean).mean()**2)),
                    "min_val": float(np.array(radlist_min).min()),
                    "max_val": float(np.array(radlist_max).max())
}
parameter['rad_mod_stats'] = {
            "rad": {
                    "mean": float(np.array(radmodlist_mean).mean()),
                    "std": float(np.sqrt(np.array(radmodlist_mean2).mean()-np.array(radmodlist_mean).mean()**2)),
                    "min_val": float(np.array(radmodlist_min).min()),
                    "max_val": float(np.array(radmodlist_max).max())
                    },
            "rd_stats_preprocessd": {
                    "mean": float(np.array(radmod_rdlist_mean).mean()),
                    "std": float(np.sqrt(np.array(radmod_rdlist_mean2).mean()-np.array(radmod_rdlist_mean).mean()**2)),
                    "min_val": float(np.array(radmod_rdlist_min).min()),
                    "max_val": float(np.array(radmod_rdlist_max).max())
                    },
            "ra_stats_preprocessd": {
                    "mean": float(np.array(radmod_ralist_mean).mean()),
                    "std": float(np.sqrt(np.array(radmod_ralist_mean2).mean()-np.array(radmod_ralist_mean).mean()**2)),
                    "min_val": float(np.array(radmod_ralist_min).min()),
                    "max_val": float(np.array(radmod_ralist_max).max())
                    },
            "ad_stats_preprocessd": {
                    "mean": float(np.array(radmod_adlist_mean).mean()),
                    "std": float(np.sqrt(np.array(radmod_adlist_mean2).mean()-np.array(radmod_adlist_mean).mean()**2)),
                    "min_val": float(np.array(radmod_adlist_min).min()),
                    "max_val": float(np.array(radmod_adlist_max).max())
                    }
}
parameter['rad_mod2_stats'] = {
            "rad": {
                    "mean": float(np.array(radmod2list_mean).mean()),
                    "std": float(np.sqrt(np.array(radmod2list_mean2).mean()-np.array(radmod2list_mean).mean()**2)),
                    "min_val": float(np.array(radmod2list_min).min()),
                    "max_val": float(np.array(radmod2list_max).max())
                    },
            "rd_stats_preprocessd": {
                    "mean": float(np.array(radmod2_rdlist_mean).mean()),
                    "std": float(np.sqrt(np.array(radmod2_rdlist_mean2).mean()-np.array(radmod2_rdlist_mean).mean()**2)),
                    "min_val": float(np.array(radmod2_rdlist_min).min()),
                    "max_val": float(np.array(radmod2_rdlist_max).max())
                    },
            "ra_stats_preprocessd": {
                    "mean": float(np.array(radmod2_ralist_mean).mean()),
                    "std": float(np.sqrt(np.array(radmod2_ralist_mean2).mean()-np.array(radmod2_ralist_mean).mean()**2)),
                    "min_val": float(np.array(radmod2_ralist_min).min()),
                    "max_val": float(np.array(radmod2_ralist_max).max())
                    },
            "ad_stats_preprocessd": {
                    "mean": float(np.array(radmod2_adlist_mean).mean()),
                    "std": float(np.sqrt(np.array(radmod2_adlist_mean2).mean()-np.array(radmod2_adlist_mean).mean()**2)),
                    "min_val": float(np.array(radmod2_adlist_min).min()),
                    "max_val": float(np.array(radmod2_adlist_max).max())
                    }
}
save_path = "/workspace/MVRSS/mvrss/config_files/all_stats.json"
with open(save_path, 'w') as f:
    json.dump(parameter, f, indent=4)

a = 1
        


