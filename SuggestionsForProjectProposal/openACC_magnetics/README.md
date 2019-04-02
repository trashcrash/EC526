# QFEparallel

A simple 2D HMC phi^4 theory accelerated with OpenACC. Tested with PGI 18.4 on
6.0 compute capable architecture.

## Features and Philosophy

This code contains many of the structures one might expect to find in a finite element
field theory computation. We list the features here and place more elaborate
explanations in the code itself.

1. HMC
2. Lattice Percolation
3. Swendsen-Wang cluster decomposition
4. Improved 2Pt correlation function estimator with Fourier Transfrom

## Using the code.

You will need a PGI compiler. We have tested against pgc++ 18.4. Edit the Makefile
to denote your desired achitecture by uncommenting the `COMPILER' variable.

All of the variables are hardcoded for ease of development. Physical variables such as
\lambda, \musqr are defined the paramater structure. Algorithmic and problem
size variables are defined as #define in the program preamble.

We have provided a timing routine to compare serial code with PGI acceleated code.
Simply uncomment the `TIMING_COMP' define in the preamble. Warm up iterations are always
done with the PGI routines so that properly thermalised data is used in the comparisons.

To compile, simply type make in the source directory, and run with ./phi

Happy Hacking!