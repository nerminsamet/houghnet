# Getting Started

This document provides information about how to train and evaluate HoughNet on COCO.
First, make sure that you have completed the [installation](INSTALL.md).

## Dataset preparation

- Download the images (2017 Train, 2017 Val, 2017 Test) from [coco website](http://cocodataset.org/#download).
- Download annotation files (2017 train/val and test image info) from [coco website](http://cocodataset.org/#download).
- If you like to use [minicoco](https://github.com/giddyyupp/coco-minitrain) dataset,
[download](https://drive.google.com/open?id=1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin) the json file and place it under the annotations folder.

  ~~~
  ${COCO_PATH}
  |-- annotations
      |-- instances_train2017.json
      |-- instances_minitrain2017.json
      |-- instances_val2017.json
      |-- image_info_test-dev2017.json
  |-- images
      |-- train2017
      |-- val2017
      |-- test2017
  ~~~


## Evaluation

Download the models you want to evaluate from our [model zoo](MODEL_ZOO.md) and put them in `HoughNet_ROOT/models/`.

To evaluate object detection with **Resnet-101 w DCN** on `val2017`
run

~~~
python src/test.py ctdet --houghnet --exp_id coco_resdcn_101_light --arch resdcn_101 --keep_res --resume  --load_model ./models/ctdet_coco_resdcn101_light.pth --coco_path $COCO_PATH
~~~

This will give an AP of `35.7` on `val2017`. `--keep_res` is for keeping the original image resolution.
Without `--keep_res` it will resize the images to `512 x 512`.
You can add `--flip_test` and `--flip_test --test_scales 0.6,0.8,1,1.2,1.5,1.8` to the above command, for flip test and multi-scale test, respectively.
The expected APs on `val2017` are `37.2` and `41.5`, respectively.

For multi-scale test with *Hourglass* net, run

~~~
python src/test.py ctdet --houghnet --exp_id coco_hg_scratch --arch hourglass --keep_res --resume --flip_test --test_scales 0.6,0.8,1,1.2,1.5,1.8 --load_model ./models/ctdet_coco_resdcn101_light.pth --coco_path $COCO_PATH
~~~

More results could be found in the model zoo.



## Training

You could find all the training scripts in the [experiments](../experiments) folder.
In the case that you don't have 4 GPUs, you can follow the [linear learning rate rule](https://arxiv.org/abs/1706.02677) to adjust the learning rate.
For instance, to train COCO object detection with *Resnet-101 w DCN* model using 4 Tesla V100 GPUs on `train2017`, run

~~~
python src/main.py ctdet --houghnet --exp_id coco_resdcn_101 --arch resdcn_101 --batch_size 44 --master_batch 8 --lr 1.75e-4 --gpus 0,1,2,3 --num_workers 16 --coco_path $COCO_PATH
~~~
or on *minitrain*

~~~
python src/main.py ctdet --houghnet --minicoco --exp_id coco_resdcn_101 --arch resdcn_101 --batch_size 44 --master_batch 8 --lr 1.75e-4 --gpus 0,1,2,3 --num_workers 16 --coco_path $COCO_PATH
~~~

If the training is terminated before finishing, you can use the same command with `--resume` to resume training. It will find the latest model with the same `exp_id`.

Our best Hourglass model is finetuned from the pretrained [ExtremeNet model](https://drive.google.com/file/d/1TG3oBkHqj_QHdOHRF0RLsN4CxWxFTovA/view?usp=sharing) (from the [ExtremeNet repo](https://github.com/xingyizhou/ExtremeNet)).
You need to download and load the model for training (see the [script](../experiments/ctdet_coco_hg104_extremenet.sh)).

We also have another Hourglass model finetuned from the pretrained [CornerNet model](https://drive.google.com/file/d/14X4BdOKqbM1mINOK3V4tCWq5Jz3O0fEf/view?usp=sharing).
You could download the model and load the model for training (see the [script](../experiments/ctdet_coco_hg104_cornernet.sh)).


**For the training and evaluation of instance segmentation, human keypoint detection and 3D object detection tasks please use their corresponding [scripts](../experiments/ctdet_coco_hg104_extremenet.sh)**.   
