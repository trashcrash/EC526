# 2D HMC phi^4 theory

## Features

This code contains many of the structures one might expect to find in a finite element
field theory computation. We list the features here and place more elaborate
explanations in the code itself.

1. HMC
2. Lattice Percolation
3. Swendsen-Wang cluster decomposition

## Using the code.

All of the variables are hardcoded for ease of development. Physical variables such as
\lambda, \musqr are defined the paramater structure. Algorithmic and problem
size variables are defined as #define in the program preamble.

To compile, simply type make in the source directory, and run with ./phi

Happy Hacking!