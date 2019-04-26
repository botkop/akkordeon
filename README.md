"What I cannot create, I do not understand." - Richard Feynman.


This project shows how to train an artificial neural network in an actor framework. Traditional neural networks are monolithic blobs trained on static hardware infrastructure. Here I propose an approach that distributes the components of a neural net (layers, data providers...) over multiple processes that run independently (async and concurrent), possibly on different machines. It also allows to dynamically add or remove training, validation and test modules.


# Akkordeon: Training a neural net with Akka

The world is asynchronous. 

This project shows how to train a neural net with Akka.

The mechanics are as follows:

A layer is embedded in a [gate](#gate). A gate is an actor. 
The results of the forward and backward pass are passed as messages from one gate to the next.
Calculations inside a layer are performed asynchronously from other layers.
Thus, a layer does not have to wait for the backward pass in order to perform the forward pass of the next batch.

Every gate has its optimizer.
Optimization on a layer runs asynchronously from other layers. 
To alleviate the 'delayed gradient' problem, I use an implementation of the ['Asynchronous Stochastic Gradient Descent with Delay Compensation'](https://arxiv.org/abs/1609.08326) optimizer.

Data providers are embedded in [sentinels](#sentinel) and implemented as actors. You can have mutiple sentinels running at the same time, each with a subset of the training data for example.
This also allows me to run the training and validation phases concurrently.

All actors can be deployed on a single machine or in a cluster of machines, leveraging both horizontal and vertical computing power.

## Components

![components](doc/training.png "Logo Title Text 1")


### Gate
A gate is similar to a layer. 
Every gate is an actor. 
Whereas in a traditional network there is only one optimizer for the complete network, here every gate has its optimizer. 
There is however no difference in functionality, since optimizers do not share data between layers. 

A gate can consist of an arbitrarily complex network in itself. 
You can put multiple convolutional, pooling, batchnorm, dropouts, ... and so on in one gate. 
Or you can assign them to different gates, thus distributing the work over multiple actors.

### Network
A network is a sequence of gates.
The sequence is open. 
You can attach multiple sentinels, each with its data provider, to the network.

### Sentinel
The sentinel is an actor, and does a couple of things:
- provide data, through the data provider, for training, validation and test
- calculate and report loss and accuracy during training and validation
- trigger the forward pass for each batch during training, validation and test
- trigger the backward pass for each batch when training

You can attach multiple sentinels to a network. 
Typically, one or more sentinels are provided for training, and one for validation. 
The latter runs every 20 seconds for example, whereas the training sentinels run continuously.

## Prepare

After having cloned/downloaded the source code of this project, get the MNIST dataset by executing the script `scripts/download_mnist.sh`
or by manually downloading the files from the URLs in the script, and putting them in a folder `data/mnist`.

You will need [sbt](https://www.scala-sbt.org/download.html) to build the project.

## Build and run

### Single JVM
```
sbt 'runMain botkop.akkordeon.SimpleAkkordeon'
```

This will produce output similar to this:

```
[info] tdp        epoch:     1 loss:  2.939994 duration: 7105.075212ms scores: (0.22618558114035087)
[info] tdp        epoch:     2 loss:  1.848889 duration: 2339.476822ms scores: (0.4044360040590681)
[info] tdp        epoch:     3 loss:  1.463448 duration: 2278.748975ms scores: (0.5158070709745762)
[info] tdp        epoch:     4 loss:  1.136699 duration: 2245.955278ms scores: (0.6229231711161338)
[info] tdp        epoch:     5 loss:  0.968350 duration: 2309.301106ms scores: (0.6776098002821712)
[info] tdp        epoch:     6 loss:  0.880695 duration: 2259.42184ms scores: (0.7060564301781735)
[info] tdp        epoch:     7 loss:  0.892328 duration: 2856.552759ms scores: (0.7027704402551813)
[info] vdp        epoch:     1 loss:  0.866831 duration: 1768.835725ms scores: (0.7107204861111112)
```

### Multiple JVMs
In this scenario, I show how to deploy the neural net on one JVM, and the sentinels on other JVMs.
The JVMs can be deployed on the same machine, or on different machines.
Note that when deploying the sentinels on separate machines, you will need to make the data accessible on those machines.

Another scenario that comes to mind is to split the network itself in separate entities, and deploy those on different JVMs.
Let's say that for now, I leave this as an exercise for the reader.

Obtain the IP address of the machine on which you want to run the neural net. 
If you run all JVMs on the same machine, then you can use `127.0.0.1`.
Append a free port number separated by colon:
```
export NNADDR=192.168.1.23:25520
```
Start the neural net in a terminal window:
```
sbt "runMain botkop.akkordeon.examples.NetworkApp $NNADDR"
```
Obtain the IP address of a machine on which you want to run a sentinel.
If you run all JVMs on the same machine, then you can use `127.0.0.1`.
The parameter `60000` is the number of samples from the data set you want to use. 
Start a training sentinel in another terminal:
```
MY_IP=192.168.0.158
sbt "runMain botkop.akkordeon.examples.SentinelApp $MY_IP train 60000 $NNADDR"
```
And another one:
```
MY_IP=192.168.0.159
sbt "runMain botkop.akkordeon.examples.SentinelApp $MY_IP train 3000 $NNADDR"
```
Also start a validation sentinel. 
```
MY_IP=192.168.0.160
sbt "runMain botkop.akkordeon.examples.SentinelApp $MY_IP validate 10000 $NNADDR"
```


# References

- [Asynchronous Stochastic Gradient Descent with Delay Compensation](https://arxiv.org/abs/1609.08326)
- [An analysis of the Delayed Gradients Problem in asynchronous SGD](https://pdfs.semanticscholar.org/716b/a3d174006c19220c985acf132ffdfc6fc37b.pdf)

