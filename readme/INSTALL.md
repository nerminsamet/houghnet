# Installation


The code was tested on Ubuntu 18.04 with Cuda 10.0, [Anaconda](https://www.anaconda.com/download) Python 3.7 and [PyTorch]((http://pytorch.org/)) v1.4.0.
NVIDIA GPUs are needed for both training and testing.
After installing Anaconda:

0. [Optional but recommended] create a new conda environment.

    ~~~
    conda create --name HoughNet python=3.7
    ~~~
    And activate the environment.

    ~~~
    conda activate HoughNet
    ~~~

1. Clone the repo:

    ~~~
    HoughNet_ROOT=/path/to/clone/HoughNet
    git clone https://github.com/nerminsamet/HoughNet $HoughNet_ROOT
    ~~~

2. Install PyTorch 1.4.0:

    ~~~
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    ~~~

3. Install the requirements:

    ~~~
    pip install -r requirements.txt
    ~~~

4. Install Detectron for instance segmentation task:

    ~~~
    python -m pip install detectron2==0.2.1 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/torch1.4/index.html
    ~~~

5. Compile DCNv2 (Deformable Convolutional Networks):

    ~~~
    cd $HoughNet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~

6. [Optional, only required if you are using multi-scale testing].
Compile NMS if you want to use multi-scale testing.

    ~~~
    cd $HoughNet_ROOT/src/lib/external
    make
    ~~~

7. Download [pretrained models](https://drive.google.com/drive/folders/1RGpb7trcdyWDKxAppxKV8dRLxwUxSosN?usp=sharing) and place them under `$HoughNet_ROOT/models/`.
You could find more information about models in [model zoo](MODEL_ZOO.md).
