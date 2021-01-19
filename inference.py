import time
import torch
from torch.backends import cudnn
from backbone import EfficientDetBackbone
import cv2
import numpy as np
import sys
import argparse
import os

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import aspectaware_resize_padding, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--iou_threshold', type=float, default=0.2)

    return parser.parse_args()

def inference(args):

    compound_coef = int(args.weights.split('/')[-1].split('-')[1][1])
    force_input_size = None  # set None to use default size
    img_path = args.input_path

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = args.threshold
    iou_threshold = args.iou_threshold

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    max_size = 512

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = [
            'Aortic_enlargement', 'Atelectasis', 'Calcification',
            'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung_Opacity',
            'Nodule/Mass', 'Other_lesion', 'Pleural_effusion', 'Pleural_thickening',
            'Pneumothorax', 'Pulmonary_fibrosis'
            ]
    
    color_list = standard_to_bgr(STANDARD_COLORS)
    
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    for _file in os.listdir(img_path):
        file_path = os.path.join(img_path, _file)

        with torch.no_grad():
            img = cv2.imread(file_path)
            norm_img = (img[...,::-1]/255 - mean)/std
            img_meta = aspectaware_resize_padding(img, max_size, max_size, means=None)
            x = torch.from_numpy(img_meta[0]).cuda().to(torch.float32).permute(2, 1, 0).unsqueeze(0)

            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
            
            out = out[0]
            meta = img_meta[1]
            print(meta)
            print(out)

            out = invert_affine(meta, out)
            print(out)
            raise
            if len(out['rois']) > 0:
                for j in range(len(out['rois'])):
                    print(_file, out['class_ids'][j], float(out['scores'][j]), out['rois'][j].astype(np.int))
            else:
                print(_file, 14, 1, 0, 0, 1, 1)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    assert (args.input_path != None) & (args.weights != None), 'have to input [input_path], [weights] arguments'
    inference(args)
