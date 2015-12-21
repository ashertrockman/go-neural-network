# go-neural-network

A feedforward neural network that learns with backpropagation (Mean Squared Error) and stochastic gradient descent.

Currently gets 3.2% error on the MNIST database ([using GoMNIST](https://github.com/petar/GoMNIST)) with two 300-node layers trained on 600k examples (approximately 70 minutes).

Includes a function to visualize learned representations:

<img src="http://ashertrockman.github.io/assets/nn-weights.gif" />

# Todo

* Concurrency
* Weight decay
