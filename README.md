# CNN-Fixations

Code for the paper 
**[CNN fixations: An unraveling approach to identify discriminative image regions](https://arxiv.org/abs/1708.06670)**

Konda Reddy Mopuri, Utsav Garg, R. Venkatesh Babu

This repository can be used to visualize predictions for four CNNs namely: AlexNet, VGG-16, GoogLeNet and ResNet-101.

### Usage Instructions

1. In the demo folder add path to caffe installation

2. Install the required dependencies

3. Navigate to the Demo folder and open a ipython notebook

*Add the .caffemodels for the respective architechtures before running the codes*

### Samples

Results of running the demo for one image from ILSVRC validation set

#### AlexNet
![alexnet sample](samples/sample_alexnet.png)

#### VGG 16
![vgg 16 sample](samples/sample_vgg.png)

#### GoogLeNet
![googlenet sample](samples/sample_googlenet.png)

#### ResNet 101
![resnet sample](samples/sample_resnet.png)

[More Samples](http://val.serc.iisc.ernet.in/cnn-fixations/)

The VGG-16 demo now also has the option to use the TensorFlow backend in addition to the default Caffe backend. (Contributed by [Joe Yearsley](https://github.com/joeyearsley))

Contact [Utsav Garg](http://utsavgarg.github.io/) if you have questions.
