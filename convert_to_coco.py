# reference: https://www.kaggle.com/sreevishnudamodaran/vinbigdata-fusing-bboxes-coco-dataset

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
sns.set(rc={"font.size":9, "axes.titlesize":15, "axes.labelsize":9,
            "axes.titlepad":11, "axes.labelpad":9, "legend.fontsize":7,
            "legend.title_fontsize":7, 'axes.grid':False})

import cv2
import json
import pandas as pd
import glob
import datetime
from tqdm.auto import tqdm
from path import Path
import numpy as np
import random
import shutil
from sklearn.model_selection import train_test_split

from ensemble_boxes import *
import warnings
from collections import Counter

import SimpleITK as sitk

def read_dicom(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    img = img[0]

    np_img = img.astype(np.float32)
    np_img -= np.min(np_img)
    np_img /= np.percentile(np_img, 99)

    np_img[np_img>1] = 1
    np_img *= (2**8-1)
    np_img = np_img.astype(np.uint8)

    np_img = np.stack((np_img,)*3, axis=-1)
    return np_img

img_path = '/workspace/datasets/VinBigData/test_images_dcm'
img_paths = [os.path.join(img_path, i) for i in os.listdir(img_path)]
output_dir = '/workspace/datasets/VinBigData/test_images'
for i, path in tqdm(enumerate(img_paths)):
    img_array = read_dicom(path)

    image_basename = Path(path).stem
    file_name = os.path.join(output_dir, path.split('/')[-1].split('.')[0]+'.jpg')
    cv2.imwrite(file_name, img_array)

'''
##### HAVE TO CHANGE THIS PART FOR YOUR ENVIRONMENT
#csv_path = '/workspace/datasets/train.csv'
#img_path = '/workspace/datasets/train'

#train_output_dir = '/workspace/datasets/coco-style/imgs/train_images'
#val_output_dir = '/workspace/datasets/coco-style/imgs/val_images'

#train_out_file = '/workspace/datasets/coco-style/train_annotations.json'
#val_out_file = '/workspace/datasets/coco-style/val_annotations.json'

if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)

if not os.path.exists(val_output_dir):
    os.makedirs(val_output_dir)

train_annotations = pd.read_csv(csv_path)

# Filtering (to remain only abnormal images)
train_annotations = train_annotations[train_annotations.class_id!=14]
train_annotations['image_path'] = train_annotations['image_id'].map(
                                    lambda x:os.path.join(img_path, str(x)+'.dicom'))

print(train_annotations)
imagepaths = train_annotations['image_path'].unique()
print('Number of Images with abnormalities: ', len(imagepaths))
anno_count = train_annotations.shape[0]
print('Number of Annotations with abnormalities: ', anno_count)

# Classes
labels = [
            '__ignore__', 'Aortic_enlargement', 'Atelectasis', 'Calcification',
            'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung_Opacity',
            'Nodule/Mass', 'Other_lesion', 'Pleural_effusion', 'Pleural_thickening',
            'Pneumothorax', 'Pulmonary_fibrosis'
         ]
viz_labels = labels[1:]

iou_thr = 0.5
skip_box_thr = 0.0001
viz_images = []
sigma = 0.1
label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133],
                [117, 76, 3], [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77],
                [194, 134, 175], [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100]]

# Weight boxes fusion (WBF)


for i, path in tqdm(enumerate(imagepaths[5:8])):
    #img_array = cv2.imread(path)
    img_array = read_dicom(path)
    image_basename = Path(path).stem

    img_annotations = train_annotations[train_annotations.image_id==image_basename]

    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()

    img_before = img_array.copy()
    for box, label in zip(boxes_viz, labels_viz):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        color = label2color[int(label)]
        img = cv2.rectangle(img_before, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3)
        cv2.putText(img_before, viz_labels[int(label)], (int(x_min), int(y_min)-5), cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, color, 1)
    viz_images.append(img_before)

    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []

    boxes_single = []
    labels_single = []

    cls_ids = img_annotations['class_id'].unique().tolist()
    count_dict = Counter(img_annotations['class_id'].tolist())
    print(count_dict)

    for cid in cls_ids:
        # Perform Fusing
        if count_dict[cid] == 1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max',
                                'y_max']].to_numpy().squeeze().tolist())
        else:
            cls_list = img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            bbox = bbox/(img_array.shape[0], img_array.shape[1], img_array.shape[0], img_array.shape[1])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())

            weights.append(1)
    boxes, scores, box_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                    iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(img_array.shape[0], img_array.shape[1], img_array.shape[0], img_array.shape[1])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()

    boxes.extend(boxes_single)
    box_labels.extend(labels_single)

    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        color = label2color[int(label)]
        img = cv2.rectangle(img_after, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
        cv2.putText(img_after, viz_labels[int(label)], (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 1)
    viz_images.append(img_after)

plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9, 'Original Boxes', va='top', ha='center', size=25)
plt.figtext(0.73, 0.9, 'WBF', va='top', ha='center', size=25)
plt.savefig('bbox_fusion_test.jpg')

# Conver to COCO
random.seed(42)
random.shuffle(imagepaths)
train_len = round(0.75*len(imagepaths))

train_paths = imagepaths[:train_len]
val_paths = imagepaths[train_len:]

now = datetime.datetime.now()

data = dict(
    info=dict(
        description=None,
        url=None,
        version=None,
        year = now.year,
        contributor=None,
        date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
    licenses=[dict(
        url=None,
        id=0,
        name=None,
        )],
    images=[
    ],
    type='instances',
    annotations=[
    ],
    categories=[
    ],
)

class_name_to_id = {}
for i, each_label in enumerate(labels):
    class_id = i-1
    class_name = each_label
    if class_id == -1:
        assert class_name == '__ignore__'
        continue
    class_name_to_id[class_name] = class_id
    data['categories'].append(dict(
        supercategory=None,
        id=class_id,
        name=class_name,
        ))



data_train = data.copy()
data_train['images'] = []
data_train['annotations'] = []

for i, path in tqdm(enumerate(train_paths)):
    img_array = read_dicom(path)

    image_basename = Path(path).stem
    file_name = os.path.join(train_output_dir, path.split('/')[-1].split('.')[0]+'.jpg')
    ## Copy Image
    #shutil.copy2(path, train_output_dir)
    cv2.imwrite(file_name, img_array)
    

    ## Add Images to annotation
    data_train['images'].append(dict(
        license=0,
        url=None,
        file_name=os.path.join('train_images', image_basename+'.jpg'),
        height=img_array.shape[0],
        width=img_array.shape[1],
        date_captured=None,
        id=i
    ))

    img_annotations = train_annotations[train_annotations.image_id==image_basename]
    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()

    ## Visualize Original Bboxes every 500th
    if (i%500==0):
        img_before = img_array.copy()
        for box, label in zip(boxes_viz, labels_viz):
            x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
            color = label2color[int(label)]
            img = cv2.rectangle(img_before, (int(x_min), int(y_min)), (int(x_max),int(y_max)),
                                color, 3)
            cv2.putText(img_before, viz_labels[int(label)], (int(x_min), int(y_min)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        viz_images.append(img_before)

    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []

    boxes_single = []
    labels_single = []

    cls_ids = img_annotations['class_id'].unique().tolist()

    count_dict = Counter(img_annotations['class_id'].tolist())

    for cid in cls_ids:
        ## Performing Fusing operation only for multiple bboxes with the same label
        if count_dict[cid]==1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

        else:
            cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()

            ## Normalizing Bbox by Image Width and Height
            bbox = bbox/(img_array.shape[0], img_array.shape[1], img_array.shape[0], img_array.shape[1])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())
            weights.append(1)

    ## Perform WBF
    boxes, scores, box_labels = weighted_boxes_fusion(boxes_list=boxes_list, scores_list=scores_list,
                                                  labels_list=labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(img_array.shape[0], img_array.shape[1], img_array.shape[0], img_array.shape[1])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()
    boxes.extend(boxes_single)
    box_labels.extend(labels_single)

    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        area = ((x_min-x_min)*(y_max-y_min))
        bbox =[
                round(x_min, 1),
                round(y_min, 1),
                round((x_max-x_min), 1),
                round((y_max-y_min), 1)
                ]

        data_train['annotations'].append(dict( id=len(data_train['annotations']), image_id=i,
                                            category_id=int(label), area=area, bbox=bbox,
                                            iscrowd=0))

    ## Visualize Bboxes after operation every 500th
    if (i%500==0):
        img_after = img_array.copy()
        for box, label in zip(boxes, box_labels):
            color = label2color[int(label)]
            img = cv2.rectangle(img_after, (int(box[0]), int(box[1])), (int(box[2]),int(box[3])),
                                color, 3)
            cv2.putText(img_after, viz_labels[int(label)], (int(box[0]), int(box[1])-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        viz_images.append(img_after)

plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
plt.figtext(0.73, 0.9,"WBF", va="top", ha="center", size=25)
plt.savefig('train_label.png')

with open(train_out_file, 'w') as f:
    json.dump(data_train, f, indent=4)



data_val = data.copy()
data_val['images'] = []
data_val['annotations'] = []

iou_thr = 0.5
skip_box_thr = 0.0001
viz_images = []

for i, path in tqdm(enumerate(val_paths)):
    #img_array  = cv2.imread(path)
    img_array = read_dicom(path)
    image_basename = Path(path).stem

    ## Copy Image
    #shutil.copy2(path, val_output_dir)
    file_name = os.path.join(val_output_dir, path.split('/')[-1].split('.')[0]+'.jpg')
    cv2.imwrite(file_name, img_array)

    ## Add Images to annotation
    data_val['images'].append(dict(
        license=0,
        url=None,
        file_name=os.path.join('val_images', image_basename+'.jpg'),
        height=img_array.shape[0],
        width=img_array.shape[1],
        date_captured=None,
        id=i
    ))

    img_annotations = train_annotations[train_annotations.image_id==image_basename]
    boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
    labels_viz = img_annotations['class_id'].to_numpy().tolist()

    ## Visualize Original Bboxes every 500th
    if (i%500==0):
        img_before = img_array.copy()
        for box, label in zip(boxes_viz, labels_viz):
            x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
            color = label2color[int(label)]
            img = cv2.rectangle(img_before, (int(x_min), int(y_min)), (int(x_max),int(y_max)),
                                color, 3)
            cv2.putText(img_before, viz_labels[int(label)], (int(x_min), int(y_min)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        viz_images.append(img_before)

    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []

    boxes_single = []
    labels_single = []

    cls_ids = img_annotations['class_id'].unique().tolist()

    count_dict = Counter(img_annotations['class_id'].tolist())
    for cid in cls_ids:
        ## Performing Fusing operation only for multiple bboxes with the same label
        if count_dict[cid]==1:
            labels_single.append(cid)
            boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

        else:
            cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
            labels_list.append(cls_list)
            bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()

            ## Normalizing Bbox by Image Width and Height
            bbox = bbox/(img_array.shape[0], img_array.shape[1], img_array.shape[0], img_array.shape[1])
            bbox = np.clip(bbox, 0, 1)
            boxes_list.append(bbox.tolist())
            scores_list.append(np.ones(len(cls_list)).tolist())
            weights.append(1)

    ## Perform WBF
    boxes, scores, box_labels = weighted_boxes_fusion(boxes_list=boxes_list, scores_list=scores_list,
                                                  labels_list=labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(img_array.shape[0], img_array.shape[1], img_array.shape[0], img_array.shape[1])
    boxes = boxes.round(1).tolist()
    box_labels = box_labels.astype(int).tolist()
    boxes.extend(boxes_single)
    box_labels.extend(labels_single)

    img_after = img_array.copy()
    for box, label in zip(boxes, box_labels):
        x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
        area = ((x_min-x_min)*(y_max-y_min))
        bbox =[
                round(x_min, 1),
                round(y_min, 1),
                round((x_max-x_min), 1),
                round((y_max-y_min), 1)
                ]

        data_val['annotations'].append(dict( id=len(data_val['annotations']), image_id=i,
                                            category_id=int(label), area=area, bbox=bbox,
                                            iscrowd=0))

    ## Visualize Bboxes after operation
    if (i%500==0):
        img_after = img_array.copy()
        for box, label in zip(boxes, box_labels):
            color = label2color[int(label)]
            img = cv2.rectangle(img_after, (int(box[0]), int(box[1])), (int(box[2]),int(box[3])),
                                color, 3)
            cv2.putText(img_after, viz_labels[int(label)], (int(box[0]), int(box[1])-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        viz_images.append(img_after)

plot_imgs(viz_images, cmap=None)
plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
plt.figtext(0.73, 0.9,"WBF", va="top", ha="center", size=25)
plt.savefig('valid_label.png')

with open(val_out_file, 'w') as f:
    json.dump(data_val, f, indent=4)
'''
