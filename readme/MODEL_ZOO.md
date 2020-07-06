# MODEL ZOO

### Notes

- All models are trained on a cluster with 4 Tesla V100 GPUs.
- The experiments are run with pytorch 1.1.0, CUDA 10.0, and CUDNN 7.5.
- Testing times are measured on a machine with a single Tesla V100 GPU. 
- The models can be downloaded directly from [Google drive](https://drive.google.com/drive/folders/1dEshWidNf54MRFgNanrrhkdpH_eywkFP?usp=sharing).
- All models are trained on COCO `train2017` and evaluated on `val2017`. 
- We show test time and AP with no augmentation / flip augmentation / multi scale (0.6, 0.8, 1, 1.2, 1.5, 1.8) augmentation. 
- You could test on COCO *test-dev* adding `--trainval`. 
- Testing time includes network forwarding time, decoding time, and nms time (for MS test).

### COCO Results

| Model                    | Test time (ms) |   AP               |  Download | 
|--------------------------|----------------|--------------------|-----------|
|[ctdet\_coco\_res101](../experiments/ctdet_coco_res101.sh)                     | 164 / 290 / 2462  | 34.3 / 36.0 / 40.7 | [model](https://drive.google.com/file/d/1lzRrTr5emJNL0EzfS_9fPv8C2L3tWAAP/view?usp=sharing) |
|[ctdet\_coco\_resdcn101_light](../experiments/ctdet_coco_resdcn101_light.sh)   | 55 / 73 / 498     | 35.7 / 37.2 / 41.5 | [model](https://drive.google.com/file/d/13eDM8sF3Jx-ARdhylt5RXOUV7jCSwuJR/view?usp=sharing) |
|[ctdet\_coco\_resdcn101](../experiments/ctdet_coco_resdcn101.sh)               | 167 / 302 / 1590  | 35.9 / 37.3 / 41.6 | [model](https://drive.google.com/file/d/1pnPfudE81But66hjs17XqehPv9DKKnsu/view?usp=sharing) |
|[ctdet\_coco\_hg104_scratch](../experiments/ctdet_coco_hg104_scratch.sh)       | 111 / 168 / 1326  | 39.3 / 40.9 / 43.7 | [model](https://drive.google.com/file/d/1hgmUkVztx6FbU_wbdo-XTYESLunBCr_F/view?usp=sharing) |
|[ctdet\_coco\_hg104_cornernet](../experiments/ctdet_coco_hg104_cornernet.sh)   | 112 / 172 / 1336  | 39.8 / 41.7 / 44.3 | [model](https://drive.google.com/file/d/1QfA98SBhoQ8Uc5Qf69IK8XfA7oBdUVL6/view?usp=sharing) |
|[ctdet\_coco\_hg104_extremenet](../experiments/ctdet_coco_hg104_extremenet.sh) | 117 / 178 / 1328  | 41.1 / 43.0 / 46.1 | [model](https://drive.google.com/file/d/1BxWecgVIlWHEphKi74C2kbZki0dDszN0/view?usp=sharing) |

 
 
