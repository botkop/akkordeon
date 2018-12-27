"What I cannot create, I do not understand." - Richard Feynman.

# Akkordeon: Training a neural net with Akka

The world is asynchronous. 
So how come AI researchers are so obsessed with optimizing synchronous processes? 
Because that's what their toolbox allows them to do.

This project shows how to train a neural net with Akka.

The mechanics are as follows:

Every layer is an actor. 
The results of the forward and backward pass are passed as messages from one layer to another.
Calculations inside a layer are performed asynchronously from other layers.
Thus, a layer does not have to wait for the backward pass in order to perform the forward pass of the next batch.

Every layer has its own optimizer.
Thus, optimization itself runs asynchronously from other layers. 
To alleviate the 'delayed gradient' problem, we use an implementation of the 'Asynchronous Stochastic Gradient Descent with Delay Compensation' optimizer.

The actor model allows us to run the training and validation phases concurrently.






