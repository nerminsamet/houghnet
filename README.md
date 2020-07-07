# HoughNet: Integrating near and long-range evidence for bottom-up object detection

Official PyTroch implementation of HoughNet. 

> [**HoughNet: Integrating near and long-range evidence for bottom-up object detection**](https://arxiv.org/abs/2007.02355),            
> Nermin Samet, Samet Hicsonmez, [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/),        
> *ECCV 2020. ([arXiv pre-print](https://arxiv.org/abs/2007.02355))*         


## Summary 
Object detection methods typically rely on only local evidence. For example, to detect the mouse in the image below, 
only the features extracted at/around the mouse are used. In contrast, HoughNet is able to utilize long-range (i.e. far away) evidence, too. 
Below, on the right, the votes that support the detection of the mouse are shown: in addition to the local evidence, 
far away but semantically relevant objects, the two keyboards, vote for the mouse. 

<img src="/readme/teaser.png" width="550">

HoughNet is a one-stage, anchor-free, voting-based, bottom-up object detection method. Inspired by the Generalized Hough Transform,
HoughNet determines the presence of an object at a certain location by the sum of the
votes cast on that location. Votes are collected from both near and long-distance locations
based on a log-polar vote field. Thanks to this voting mechanism, HoughNet is able to integrate both near and long-range,
class-conditional evidence for visual recognition, thereby generalizing and enhancing current object detection methodology, 
which typically relies on only local evidence. On the COCO dataset, HoughNet achieves 46.4 AP (and 65.1 AP<sub>50</sub>), 
performing on par with the state-of-the-art in bottom-up object detection and outperforming most  major one-stage and two-stage methods. 
We further validate the effectiveness of HoughNet in another task, namely, "labels to photo" image generation by integrating the 
voting module to two different GAN models and showing that the accuracy is significantly improved in both cases. 

## Highlights
- Hough voting idea is applied through a log-polar vote field to utilize short and long-range evidence in a deep
learning model for generic object detection.
- Our best single model achieves *46.4* AP on COCO test-dev.
- HoughNet is effective for small objects (+2.5 AP points over the baseline).
- We provide Hough voting as a [module](src/lib/models/networks/hough_module.py) to be used in another works.
- We provide COCO `minitrain` as a mini training set for COCO. It is useful for hyperparameter tuning and 
  reducing the cost of ablation experiments. `minitrain` is strongly  positively correlated with the performance of 
  the same model trained on `train2017`. For experiments, 
  object instance statistics and download please refer to [COCO minitrain](https://github.com/giddyyupp/coco-minitrain)

A step-by-step animation of the voting process is provided [here](https://docs.google.com/presentation/d/1TI9gL3RC7URcNI7C72xxBiozbd2aNnYGTPAzU-BNbbo/edit?usp=sharing).

## Results on COCO val2017

| Backbone        | AP / AP<sub>50</sub> | Multi-scale AP / AP<sub>50</sub> |
|:---------------:|:----------:|:----------------------:|
|Hourglass-104    | 43.0 / 62.2 |  46.1 / 64.6         |
|ResNet-101 w DCN | 37.2 / 56.5 |  41.5 / 61.5         |
|ResNet-101       | 36.0 / 55.2 |  40.7 / 60.6         |

All models could be found in [Model zoo](readme/MODEL_ZOO.md).

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Evaluation and Training

For evaluation and training details please refer to [GETTING_STARTED.md](readme/GETTING_STARTED.md).

## Acknowledgement

This work was supported by the AWS Cloud Credits for Research program and by the Scientific and Technological Research Council of Turkey (TUBITAK) through the project titled "Object Detection in Videos with Deep Neural Networks" (grant number 117E054). The numerical calculations reported in this paper were partially performed at TUBITAK ULAKBIM,  High Performance and Grid Computing Center (TRUBA resources). We also thank the authors of [CenterNet](https://github.com/xingyizhou/CenterNet) for their clean code and inspiring work.

## License

HoughNet is released under the MIT License (refer to the [LICENSE](readme/LICENSE) file for details). We developed HoughNet on top of [CenterNet](https://github.com/xingyizhou/CenterNet). Please refer to the License of CenterNet for more detail.

## Citation

If you find HoughNet useful for your research, please cite our paper as follows. 

> N. Samet, S. Hicsonmez, E. Akbas, "HoughNet: Integrating near and long-range evidence for bottom-up object detection", 
> In European Conference on Computer Vision (ECCV), 2020.

BibTeX entry: 
```
@inproceedings{HoughNet, 
  authors = {Nermin Samet and Samet Hicsonmez and Emre Akbas},
  title = {HoughNet: Integrating near and long-range evidence for bottom-up object detection}, 
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020}, 
}
```
