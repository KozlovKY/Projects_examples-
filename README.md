## QNODE: Learning quantum dynamics using latent neural ODEs vers 2.0
https://arxiv.org/abs/2110.10721

## Abstract
The  motivation of this project was written in the state above, few words about my work : 
In this project I try to train QNODE on real experimental data, in order to obtain a model that can predict the dynamics of a qubits, the resulting density matrix can be used for other performance calculations.
I wish I could use model for calculations on  two or  multiple qubits

## Work
In work i used the code in the state above, and increase some fragments.


1. The initial step is - check model on one cubit 

The state of a qubit can be represented on a Bloch sphere

Below you can see results in training process 

## Result
<p align="center">
<img src="gifs/latentdynamsclosed.gif" width="250" height="250">

## Review
Now I have problems with model :  on 7500 epoch the ELBO can't decrease, so I try to fix it 

