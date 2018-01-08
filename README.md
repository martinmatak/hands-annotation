# Hands detection app 
This project is a domain-specific  usage of [Keras implementaion of RetinaNet](https://github.com/fizyr/keras-retinanet).

RetinaNet is a deep neural network for object detection and it is described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

Here is RetinaNet trained for a specific object detection - namely for detecting hands. Dataset used for training is [publicly available](http://www.robots.ox.ac.uk/~vgg/data/hands/). 

## Installation

0) Create virtual environment for python - execute `virtualenv -p python3 venv` and activate it by executing `source venv/bin/activate`. 
1) Clone this repository.
2) Install requirements by executing `pip install -r requirements.txt`
3) In the repository, execute `python setup.py install`.
4) As of writing, this repository requires the master branch of `keras-resnet` (run `pip install --upgrade git+https://github.com/broadinstitute/keras-resnet`).


#### Usage
By default, already trained model (in snapshots directory) is used. This model is unfortunately weak and not well trained because of lack of hardware resources. 

Script `main.py` in `examples` directory is one of the examples how this model can be used. It comes with two options: annotating JPG image (`-i`) or annotating video (`-v`).

Scheme for usage is the following:
`$ python examples/main.py -option <input_file_path> <output_file_path>`
where `-option` is either `-i` or `-v`.
