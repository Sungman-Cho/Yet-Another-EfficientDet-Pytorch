import torch
import numpy as np
import cv2
import albumentations as A

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale
        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, m=3, p=0.2):
        self.p = p
        self.m = m

        # augmentations regardless of bbox's location
        self.img_transform = A.Compose([
            A.HorizontalFlip(p=self.p),
            A.RandomBrightnessContrast(p=self.p),
            A.OneOf([
                A.MedianBlur(blur_limit=self.m, p=self.p),
                A.MotionBlur(p=self.p),
                A.IAASharpen(p=self.p),
                ], p=self.p),
            A.OneOf([
                A.CLAHE(clip_limit=self.m),
                A.Equalize(),
                ], p=self.p),
            A.OneOf([
                A.GaussNoise(p=self.p),
                A.MultiplicativeNoise(p=self.p),
                ], p=self.p)
                ], bbox_params=A.BboxParams(format='pascal_voc'))

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = image*255
        image = self.img_transform(image=image.astype('uint8'), bboxes=annots)['image']
        image = image/255
        sample = {'img': image, 'annot': annots}

        return sample

class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
