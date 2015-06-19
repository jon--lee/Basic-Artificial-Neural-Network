# Basic Neural Network Architecture


This project is simply a demonstration of a basic neural network architecture.
In the root directory you can find the neural network jar file which contains
all the necessary tools of creating a network.
This network was created by Jonathan Lee under the guidance of Dr. Eric Nelson of the Harker School.

What are neural networks and what purpose do they serve? The short answer: They are awesome and they make the world a better place.
The long answer: [Artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network).
You can also read one of my papers on neural networks or perceptrons to get an idea of what they are, how they are created and some of their applications. This one implements some form of gradient descent with backpropagation. The network is limited to one input layer, one hidden layer and one output layer.

This was originally a source file not a jar, so it is not really intended to be used. And even if you tried using it, the interface isn't really refined. If you really want to though, you can just import the jar file or the source code into your project. To create the neural network, You need to pass to the constructor some training sets (module can also be found in root directory) and Two double arrays of weights since there are three layers (technically these weights can be random, but choose your dimensions carefully as they correspond to the number of nodes in each layer). You also need to give it a lambda factor which just helps the learning rate and a maximum number of iterations. These last two matter only if you want to train the network. If you already have weights that work for you and you only want to run the network, you can give these random values.

It is recommended that you systematically normalize the inputs in some way usually so that they are between zero and one or negative one and one. There are some ways to do it that are better than others, but for something simple I recommend min-max normalization or just general scaling by a constant factor. 

You can train the network by calling the minimization method and you can run the network by calling the calculateOutput method. You must pass the calculateOutput method a double array of inputs.

I haven't written any documentation here, but the source code is thoroughly documented.

UPDATE May 12, version 1.0.0: The current version of the network was the one used to demonstrate Optical Character Recognition not Facial Recognition.
