# Yet Another EfficientDet Pytorch

original repo : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

challenge : https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection



## Requirements

### Datasets

* Have to prepare COCO format datasets.

  * before you run below code, please edit some code lines (path setting)
  * after you edit the code lines, please run below code

  ```
  python convert_to_coco.py
  ```



### Project

* Prepare project(.yml) file (please follow original repo's guide)



## Training

### From scratch (Efficient D5 with 2GPUS)

* Training with QUADRO RTX 8000 * 2 

```
CUDA_VISIBLE_DEVICES=2,3 python train.py -c 5 -p VinBigData --batch_size 4 --lr 1e-3
```



## Test

TBD

