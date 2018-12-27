"What I cannot create, I do not understand." - Richard Feynman.

# Akkordeon: Training a neural net with Akka

The world is asynchronous. 

This project shows how to train a neural net with Akka.

The mechanics are as follows:

Every layer is an actor. 
The results of the forward and backward pass are passed as messages from one layer to the next.
Calculations inside a layer are performed asynchronously from other layers.
Thus, a layer does not have to wait for the backward pass in order to perform the forward pass of the next batch.

Every layer has its own optimizer.
Thus, optimization itself runs asynchronously from other layers. 
To alleviate the 'delayed gradient' problem, we use an implementation of the '[Asynchronous Stochastic Gradient Descent with Delay Compensation'](https://arxiv.org/abs/1609.08326) optimizer.

The actor model allows us to run the training and validation phases concurrently, and so data providers are implemented as actors.

## Prepare

After having cloned/downloaded the source code of this project, get the MNIST dataset by executing the script `scripts/download_mnist.sh`
or by manually downloading the files from the URLs in the script, and putting them in a folder `data/mnist`.

You will need [sbt](https://www.scala-sbt.org/download.html) to build the project.

## Build and run

```
sbt 'runMain botkop.akkordeon.hash.SimpleAkkordeon'
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




