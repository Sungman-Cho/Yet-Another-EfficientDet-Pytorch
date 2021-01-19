import os
import sys
import json

json_file = '/workspace/datasets/VinBigData/annotations/instances_train_images.json'

with open(json_file, 'r') as data_json:
    data_dict = json.load(data_json)

for images in data_dict['images']:
    print(images['file_name'], images['file_name'].split('/')[1])
    images['file_name'] = images['file_name'].split('/')[1]

with open(json_file, 'w') as json_file:
    json.dump(data_dict, json_file)

json_file = '/workspace/datasets/VinBigData/annotations/instances_val_images.json'

with open(json_file, 'r') as data_json:
    data_dict = json.load(data_json)

for images in data_dict['images']:
    print(images['file_name'], images['file_name'].split('/')[1])
    images['file_name'] = images['file_name'].split('/')[1]

with open(json_file, 'w') as json_file:
    json.dump(data_dict, json_file)
