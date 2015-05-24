# Basic Neural Network Architecture


This project is simply a demonstration of a basic neural network architecture.
In the root directory you can find the neural network jar file which contains
all the necessary tools of creating a network.
This network was created by Jonathan Lee under the guidance of Dr. Eric Nelson of the Harker School.

What are neural networks and what purpose do they serve? The short answer: They are awesome and they make the world a better place.
The long answer: [Artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network).
You can also read one of my papers on neural networks or perceptrons to get an idea of what they are, how they are created and some of their applications. This one implements gradient descent and backpropagation. This is a network with one input layer, one hidden layer and one output layer.

May 20

How can I integrate this network into my own Java project? Just import the most recent library and you can start using it. If you view the source code, it has detailed information about every method and how each one works. Here's a quick how-to:

**Get your your training sets.** You can decide they way that you create them, but there is a TrainingSet class in the library. Set the inputs and outputs using arrays of doubles as shown in the constructor:

```double[] inputs = yourMethodOfGettingInputs();
double[] outputs = yourMethodOfGettingOutputs();
TrainingSet set = new TrainingSet(inputs, outputs);```

Or you can do it manually with setter methods:

```TrainingSet set = new TrainingSet();
set.setInputs(inputs);
set.setOutputs(outputs);```

Remember the idea is with those inputs you want the network to produce those outputs. They don't have to be the same length or have any other formal correlation. The beauty of networks is that you get to decide.


**Decide what weights you want.** The heart of the network is the weights and if you already have predetermined weights you can use those.

This is a source file and not a formal library, so it is not really intended to be used. And even if you tried using it, the interface isn't really refined. If you really want to though, you can just copy and paste the file into your project. To create the neural network, You need to pass to the constructor some training sets (module can also be found in root directory) and Two double arrays of weights since there are three layers (technically these weights can be random, but choose your dimensions carefully as they correspond to the number of nodes in each layer). You also need to give it a lambda factor which just helps the learning rate and a maximum number of iterations. These last two matter only if you want to train the network. If you already have weights that work for you and you only want to run the network, you can give these random values.

It is recommended that you systematically normalize the inputs in some way usually so that they are between zero and one or negative one and one. It does not really matter how you do it, but normalization has been known to improve backpropagation. I recommend min-max normalization or just general scaling by a constant factor. 

You can train the network by calling the minimization method and you can run the network by calling the calculateOutput method. You must pass the calculateOutput method a double array of inputs.

Soon I will have a jar file up and a better interface is coming soon but the algorithm will likely remain the same.

UPDATE May 12, version 1.0.0: The current version of the network was the one used to demonstrate Optical Character Recognition.