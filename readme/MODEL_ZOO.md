# MODEL ZOO

### Notes

- All models are trained on a cluster with 4 Tesla V100 GPUs.
- The experiments are run with pytorch 1.4.0, CUDA 10.0, and CUDNN 7.5.
- Testing times are measured on a machine with a single Tesla V100 GPU.
- All models are trained on COCO `train2017` and evaluated on `val2017`.
- You could test on COCO *test-dev* adding `--trainval`.
- The models can be downloaded directly from [Google drive](https://drive.google.com/drive/folders/1RGpb7trcdyWDKxAppxKV8dRLxwUxSosN?usp=sharing).

### COCO Object Detection Results

| Model                    | Test time (ms) |   AP               |  Download |
|--------------------------|----------------|--------------------|-----------|
|[ctdet\_coco\_res101](../experiments/ctdet_coco_res101.sh)                     | 164 / 290 / 2462  | 34.3 / 36.0 / 40.7 | [model](https://drive.google.com/file/d/1qHq1MvGbHO9GiCzIpa8K8gtTVloSjvry/view?usp=sharing) |
|[ctdet\_coco\_resdcn101_light](../experiments/ctdet_coco_resdcn101_light.sh)   | 55 / 73 / 498     | 35.7 / 37.2 / 41.5 | [model](https://drive.google.com/file/d/1dJkrtqPXfra_cdSybwvgJraxPWdQ2H4K/view?usp=sharing) |
|[ctdet\_coco\_resdcn101](../experiments/ctdet_coco_resdcn101.sh)               | 167 / 302 / 1590  | 35.9 / 37.3 / 41.6 | [model](https://drive.google.com/file/d/1GrgkbXvDa1DFE7UO4UgEFaTR2SOfKEGD/view?usp=sharing) |
|[ctdet\_coco\_hg104_scratch](../experiments/ctdet_coco_hg104_scratch.sh)       | 111 / 168 / 1326  | 39.3 / 40.9 / 43.7 | [model](https://drive.google.com/file/d/1WmAFwhVVXWek4K2sEfAVCvNv0MsQwiLg/view?usp=sharing) |
|[ctdet\_coco\_hg104_cornernet](../experiments/ctdet_coco_hg104_cornernet.sh)   | 112 / 172 / 1336  | 39.8 / 41.7 / 44.3 | [model](https://drive.google.com/file/d/1AkvYx-zVnDUcGZzLKYloR4iOt0Il_zr8/view?usp=sharing) |
|[ctdet\_coco\_hg104_extremenet](../experiments/ctdet_coco_hg104_extremenet.sh) | 117 / 178 / 1328  | 41.1 / 43.0 / 46.1 | [model](https://drive.google.com/file/d/1Oqp_y0l4tnSZlutl5FbJcj8FYn80j8f-/view?usp=sharing) |


- We show test time and AP with no augmentation / flip augmentation / multi scale (0.6, 0.8, 1, 1.2, 1.5, 1.8) augmentation.
- Testing time includes network forwarding time, decoding time, and nms time (for MS test).

### COCO Instance Segmentation Results

| Model                    |   AP / AP50        |   Box AP / AP50    |  Download |
|--------------------------|--------------------|--------------------|-----------|
|[ctseg\_coco\_resdcn101\_baseline](../experiments/ctseg_coco_resdcn101_baseline.sh)           | 27.2 / 46.4  | 33.9 / 51.3 | [model](https://drive.google.com/file/d/1fKeuU_RH2yEIvewI8_vu7FDGUknNpMcD/view?usp=sharing) |
|[ctseg\_coco\_resdcn101\_light](../experiments/ctseg_coco_resdcn101_light.sh)                 | 28.4 / 48.0  | 35.0 / 52.9 | [model](https://drive.google.com/file/d/11tS0O--nYwlZqHSFX9dvZpDzDZ4Ec4U4/view?usp=sharing) |


- Results are obtained without any test time augmentation.
- For instance segmentation task we extended the model with two new branches: *prototype mask prediction branch* and *attention map prediction branch*. More information could be found in the paper.

### COCO 2D Keypoint Estimation Results

| Model                    |   AP / AP50        |   Box AP / AP50    |  Download |
|--------------------------|--------------------|--------------------|-----------|
|[Baseline (CenterNet)](https://github.com/xingyizhou/CenterNet) | 54.7 / 81.7  | 47.5 / 63.9 | [model](https://drive.google.com/file/d/1VeiRtuXfCbmhQNGV-XWL6elUzpuWN-4K/view) |
|[multi\_pose\_hm\_coco\_dla34\_1x\_light](../experiments/multi_pose_coco_dla34_1x.sh) | 56.9 / 81.6  | 50.1 / 71.4 | [model](https://drive.google.com/file/d/1shBget8H0GklQVvq8ckhlBChpx4asU-f/view?usp=sharing) |
|[multi\_pose\_hp\_coco\_dla34\_1x\_light](../experiments/multi_pose_coco_dla34_1x.sh) | 56.8 / 81.5  | 50.2 / 70.9 | [model](https://drive.google.com/file/d/15qYYygNFHQ855ODWqYc2v5NZn_ZJm_A4/view?usp=sharing) |
|[multi\_pose\_hm\_hp\_coco\_dla34\_1x\_light](../experiments/multi_pose_coco_dla34_1x.sh) | 56.9 / 81.6  | 50.4 / 71.7 | [model](https://drive.google.com/file/d/11SnZh3pdbIGMU7dup7FWtfTZuQarzoMv/view?usp=sharing) |

- Results are presented with test time flip augmentation.
- For fair comparison, following [CenterNet](https://github.com/xingyizhou/CenterNet) we fine-tuned the models from their corresponding [object detection model](https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view).

### KITTI 3D Object Detection

| Model  | AP<sub>e</sub> |AP<sub>m</sub> |AP<sub>h</sub> |  AOS<sub>e</sub> | AOS<sub>m</sub> |AOS<sub>h</sub>| BEV AP<sub>e</sub> | BEV AP<sub>m</sub> | BEV AP<sub>h</sub> | Download |
|--------------------------|----------------|--------------------|-----------| ---------|---------|---------|---------|---------|---------|---------|
|[ddd\_coco\_dla34\_light](../experiments/ddd_sub.sh)  | 89.4  | 79.2  | 69.7  | 85.7  | 75.1  | 65.7  | 35.3  | 29.7  | 24.6  |  [model](https://drive.google.com/file/d/1yKtIFJK34Ht8tT-mUcIgAUeXeW-81B_o/view?usp=sharing) |

- For KITTI dataset please follow the instructions [here](https://github.com/xingyizhou/CenterTrack/blob/master/readme/DATA.md)
- Experimented with the split from is [SubCNN](https://github.com/tanshen/SubCNN).
- Results are obtained without any test time augmentation.
