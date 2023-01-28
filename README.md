Created by 4868
This repository contains PyTorch implementation for PPL

Our code is based on mmsegmentation and mmdetection.

## Usage

### Requirements

- torch>=1.8.0
- torchvision
- timm
- mmcv-full==1.3.17
- mmseg==0.19.0
- mmdet==2.17.0
- regex
- ftfy
- fvcore

To use our code, please first install the `mmcv-full` and `mmseg`/`mmdet` following the official guidelines ([`mmseg`](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md), [`mmdet`](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)) and prepare the datasets accordingly.

### Pre-trained CLIP Models

Download the pre-trained CLIP models (`RN50.pt`, `RN101.pt`, `VIT-B-16.pt`) and save them to the `pretrained` folder.

### Segmentation

#### Training & Evaluation on ADE20K

To train the PPL model based on CLIP ResNet-50, run:

```
bash dist_train.sh configs/PPL_fpn_res50_512x512_80k.py 8
```

To evaluate the performance with multi-scale testing, run:

```
bash dist_test.sh configs/PPL_fpn_res50_512x512_80k.py /path/to/checkpoint 8 --eval mIoU --aug-test
```

###  Detection

#### Training & Evaluation on COCO
To train our PPL-RN50 using RetinaNet framework, run
```bash
 bash dist_train.sh configs/retinanet_PPL_r50_fpn_1x_coco.py 8
```

To evaluate the box AP of RN50-PPL (RetinaNet), run
```bash
bash dist_test.sh configs/retinanet_PPL_r50_fpn_1x_coco.py /path/to/checkpoint 8 --eval bbox
```
To evaluate both the box AP and the mask AP of RN50-PPL (Mask-RCNN), run
```bash
bash dist_test.sh configs/mask_rcnn_PPL_r50_fpn_1x_coco.py /path/to/checkpoint 8 --eval bbox segm
```

## License
MIT License
