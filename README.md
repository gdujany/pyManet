# pyManet
MANchester Energy Test based on numpy

This package implements the energy test, a statistical tool to measure
the agreement between two distributions of points in a multidimentionale space.
This can be useful for [CP violation](https://en.wikipedia.org/wiki/CP_violation) studies comparing for example the distribution of particles and antiparticles
over the phase-space to look for local CP violation
(for details see for example [here](https://arxiv.org/abs/1105.5338v1)).

Basic functions are implemented to perform the Energy Test and to evaluate the significance of 
the result using permutations.

## Requirements
Manet requires [NumPy](http://www.numpy.org/), a common Python package for scientific computing.
