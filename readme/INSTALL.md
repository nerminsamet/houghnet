# Installation


The code was tested on Ubuntu 18.04 with Cuda 10.0, [Anaconda](https://www.anaconda.com/download) Python 3.7 and [PyTorch]((http://pytorch.org/)) v1.4.0.
NVIDIA GPUs are needed for both training and testing.
After install Anaconda:

0. [Optional but recommended] create a new conda environment.

    ~~~
    conda create --name HoughNet python=3.7
    ~~~
    And activate the environment.

    ~~~
    conda activate HoughNet
    ~~~

1. Clone this repo:

    ~~~
    HoughNet_ROOT=/path/to/clone/HoughNet
    git clone https://github.com/nerminsamet/HoughNet $HoughNet_ROOT
    ~~~ 

2. Install pytorch 1.4.0:

    ~~~
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    ~~~

3. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~

4. Compile deformable convolutional.

    ~~~
    cd $HoughNet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~

5. [Optional, only required if you are using multi-scale testing]. 
Compile NMS if you want to use multi-scale testing.

    ~~~
    cd $HoughNet_ROOT/src/lib/external
    make
    ~~~

6. Download [pretrained models](https://drive.google.com/drive/folders/1dEshWidNf54MRFgNanrrhkdpH_eywkFP?usp=sharing) and place them to `$HoughNet_ROOT/models/`. 
You could find more information about models in [model zoo](MODEL_ZOO.md).
