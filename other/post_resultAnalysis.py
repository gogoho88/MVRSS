import os
import json
import numpy as np

path1 = '/workspace/MVRSS/logs/carrada/tmvanet/tmvanet_e300_lr0.0001_s42_3'
path2 = '/workspace/MVRSS/logs/carrada/tmvanet/tmvanet_e600_lr0.0001_s42_0'
result_path1 = os.path.join(path1,'results','results.json')
result_path2 = os.path.join(path2,'results','results.json')
config_path1 = os.path.join(path1,'config.json')
config_path2 = os.path.join(path2,'config.json')

with open(result_path1, 'r') as f:
    result1 = json.load(f)
with open(result_path2, 'r') as f:
    result2 = json.load(f)
with open(config_path1, 'r') as f:
    config1 = json.load(f)
with open(config_path2, 'r') as f:
    config2 = json.load(f)

config_diff = [{key:[config1.get(key),config2.get(key)]} for key in config2.keys() if config1.get(key)!=config2.get(key)]
for i in range(len(config_diff)):
    print(config_diff[i])

print('---------------------')
result_summary = [{'RD_mIOU': [result1['test_metrics']['range_doppler']['miou'], result2['test_metrics']['range_doppler']['miou']]},
                    {'RD_dice': [result1['test_metrics']['range_doppler']['dice'], result2['test_metrics']['range_doppler']['dice']]},
                    {'RD_recall': [result1['test_metrics']['range_doppler']['recall'], result2['test_metrics']['range_doppler']['recall']]},
                    {'RA_mIOU': [result1['test_metrics']['range_angle']['miou'], result2['test_metrics']['range_angle']['miou']]},
                    {'RA_dice': [result1['test_metrics']['range_angle']['dice'], result2['test_metrics']['range_angle']['dice']]},
                    {'RA_recall': [result1['test_metrics']['range_angle']['recall'], result2['test_metrics']['range_angle']['recall']]}
                    ]
for i in range(len(result_summary)):
    print(result_summary[i])
a = 1
