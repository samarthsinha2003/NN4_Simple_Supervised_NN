# Simple Supervised Neural Network with PyTorch

This is an example implementation of a simple supervised neural network using PyTorch. The neural network is trained and evaluated on the MNIST dataset, which consists #of images of handwritten digits.

<h3>Getting Started</h3>
To run the code, you'll need to have PyTorch installed. You can install PyTorch using pip:

pip install torch


You'll also need to download the MNIST dataset. You can do this by running the following command:

python -m torchvision.datasets.mnist


This will download the dataset to the ./data directory.

<h3>Usage</h3>
You can run the code by running the train.py script:

python train.py


This will train the neural network on the MNIST dataset and evaluate its accuracy on the test set.

<h3>Model Architecture</h3>
The neural network used in this example has 3 hidden layers with 256, 128, and 10 units respectively. The input to the network is a 28x28 grayscale image, which is flattened into a vector of size 784. The output of the network is a probability distribution over the 10 possible digit classes.

<h3>Training</h3>
The neural network is trained using the cross-entropy loss function and the stochastic gradient descent (SGD) optimizer. The learning rate for each parameter group in the optimizer is set to 0.01. The network is trained for 10 epochs with a batch size of 32.

<h3>Evaluation</h3>
The accuracy of the neural network is evaluated on the test set of the MNIST dataset. The accuracy is calculated as the percentage of correctly classified images out of the total number of images in the test set.
