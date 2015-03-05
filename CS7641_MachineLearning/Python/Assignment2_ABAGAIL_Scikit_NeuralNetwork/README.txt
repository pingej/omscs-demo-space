This is a python (or jython) wrapper around the ABAGAIL neural network class. There is a demo included using the iris data set. The data format is meant to be easily compatible with scikit-learn data, though you have to pass data back and forth through text files, since this is jython and scikit-learn is python. You can't have it all, I guess.

To run the demo, build ABAGAIL then from the command line run

jython -Dpython.path=<path to ABAGAIL.jar> ScikitAbagailNeuralNetwork.py


********************************************
Heisenberg & Schrodinger get pulled over.

Heisenberg is driving and the cop asks him "Do you know how fast you were going?"

"No, but I know exactly where I am" Heisenberg replies.

The cop says "You were doing 55 in a 35."

Heisenberg throws up his hands and shouts "Great! Now I'm lost!"

The cop thinks this is suspicious and orders him to pop open the trunk.

He checks it out and says "Do you know you have a dead cat back here?"

"We do now, asshole!" says Schrodinger.
********************************************


