# Hands detection app 
This project is a domain-specific  usage of [Keras implementaion of RetinaNet](https://github.com/fizyr/keras-retinanet).

RetinaNet is a deep neural network for object detection and it is described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

Here is RetinaNet trained for a specific object detection - naimely for detecting hands. Dataset used for training is [publicly available](http://www.robots.ox.ac.uk/~vgg/data/hands/). 

## Installation

0) Clone this repository.
1) Create virtual environment for python - execute `virtualenv -p python3 venv` and activate it by executing `source venv/bin/activate`.
2) Install requirements by executing `pip install -r requirements.txt`
3) In the repository, execute `python setup.py install`.
4) As of writing, this repository requires the master branch of `keras-resnet` (run `pip install --upgrade git+https://github.com/broadinstitute/keras-resnet`).


### Usage
By default, pretrained model is used which can be downloaded [here](https://www.dropbox.com/s/docoy4p0hl40v1n/resnet50_csv_17.h5). To use it, move it to the `snapshots` directory without renaming it. However, this model is unfortunately weak and not well trained because of lack of hardware resources.

Script `main.py` in `examples` directory is one of the examples how this model can be used. It comes with two options: annotating JPG image (`-i`) or annotating video (`-v`).

Scheme for usage is the following:
`$ python examples/main.py -option <input_file_path> <output_file_path>`
where `-option` is either `-i` or `-v`.
