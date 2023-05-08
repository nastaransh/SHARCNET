

### Environment

1. load modules
    1. module load python
    2. module load scipy-stack
2. create virtual environment and activate it
    1. virtualenv --no-download ENV
    2. source ENV/bin/activate
    3. pip install --no-index --upgrade pip
3. load modules
    1. pip install --no-index tensorflow
    2. pip install --no-index sckit-learn
    3. pip install --no-index seaborn
  
Lesson 1

linear.py (a linear model)


Lesson 2
Two examples that shows what is learned, each of which is written
in jupyter notebook and python.

mnist.ipynb
mnist.py
vgg16.ipynb
vgg16.py
dog.jpg (sample jpg used in vgg16 example)


Lesson 3
Two examples to demonstrate the use of TF debugger

backward_debug.py (good lr=0.008, bad lr=0.1)
mnist_debug.py


Training data and pre-trained model
Since the compute nodes of Graham don't have access to the internet,
We need to download the training data (mnist and cifar10) and the pre-trained
model (vgg16) that are used in our examples from the login node.
Please run the following lines at python prompt:

import tensorflow as tf
tf.keras.datasets.mnist.load_data()
tf.keras.datasets.cifar10.load_data()
tf.keras.applications.vgg16.VGG16(). If this doesn't work, then download from
https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5 into ~/.keras/models

The first time you run the above python commands, the data and the model will be
downloaded into ~/.keras.
