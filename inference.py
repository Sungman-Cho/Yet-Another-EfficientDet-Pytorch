import time
import torch
from torch.backends import cudnn
from backbone import EfficientDetBackbone
import cv2
import numpy as np
import sys
import argparse
import os
import csv
import pathlib

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import *
import confuse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, default=None)
    parser.add_argument('output_path', type=str, default='result')
    parser.add_argument('--weights', type=str, default=None)

    return parser.parse_args()

def inference(args):
    config = confuse.Configuration('VinBigData', __name__)
    config.set_file('config.yaml')

    compound_coef = int(args.weights.split('/')[-1].split('-')[1][1])
    force_input_size = None  # set None to use default size
    img_path = args.input_path

    threshold = config['Inference']['threshold'].get()
    iou_threshold = config['Inference']['iou_threshold'].get()

    mean = config['Property']['mean'].get()
    std = config['Property']['std'].get()

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = config['Property']['classes'].get()
    
    color_list = standard_to_bgr(STANDARD_COLORS)
    
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = config['Property']['input_sizes'].get()
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()
    
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)
    f = open(os.path.join(args.output_path, 'result.csv'), 'w')

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()
    
    for _file in os.listdir(img_path):
        file_path = os.path.join(img_path, _file)
        img = cv2.imread(file_path)
        ori_imgs, framed_imgs, framed_metas = preprocess_video(img, max_size=input_size)
        ori_imgs = ori_imgs[0]
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        # model predict
        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

            out = invert_affine(framed_metas, out)[0]

            predict_result = '{}, '.format(_file.split('.')[0])
            if len(out['rois']) > 0:
                for j in range(len(out['rois'])):
                    x1, y1, x2, y2 = out['rois'][j].astype(np.int)
                    cls = int(out['class_ids'][j]) - 1
                    score = float(out['scores'][j])
                    cv2.rectangle(ori_imgs, (x1, y1), (x2, y2), color_list[cls], 2)
                    cv2.putText(ori_imgs, '{}, {:.3f}'.format(cls, score), (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, color_list[cls], 2)
                    cv2.imwrite(os.path.join(args.output_path, _file), ori_imgs)
                    box_result = '{} {} {} {} {} {} '.format(
                                out['class_ids'][j],
                                format(out['scores'][j], '2.2f'),
                                x1, y1, x2, y2)
                                
                    predict_result += box_result
                predict_result = predict_result[:-1]+'\n'
            else:
                predict_result += '14 1 0 0 1 1\n'
            
            f.write(predict_result)
    f.close()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    assert (args.input_path != None) & (args.weights != None), 'have to input [input_path], [weights] arguments'
    inference(args)
