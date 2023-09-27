# Stochastic_GMMS_memristor_model
Alpha-stable based stochastic GMMS memristor model

## WHAT IS IT?
This is an implementation that incorporates the variability of the hysteresis cycles within the Generalized Mean Metastable Switch Memristor Model (GMMS) based on the empirical alpha-stable characterization for the probability distributions of the Set and Reset thresholds of a typical Knowm SDC memristor. 

## HOW IT WORKS

To produce sequences of values with the distributions of interest, namely alpha-stable and Gaussian distributions, incorporating these within the GMMS model to introduce the desired variability at the V_OFF and V_ON thresholds, in a simple and efficient way (with less computational time consumption although at the cost of requiring some reserved memory space), Monte Carlo roulette principle was employed, using tables of 1000 records and an index that determines the retrieved value by means of a uniform distribution. In all cases, the tables must store predetermined values, which correspond to an ordered sequence of samples and whose empirical distribution profile corresponds precisely to the desired random variable.

## HOW TO USE IT

### Stochastic GMMS memristor model

Demo code: Execute the python main code  stochastic_GMMS_memristor_demo.py.


### Alpha-stable parameters estimation and simulation

Demo code: Execute the python main code  alpha-stable_demo.py.


## THINGS TO NOTICE

The prposed model can be considered as acceptable approximations for simulation purposes, especially, taking into account that the variation in behavior between memristors of the same type, even within the same package, is notorious and significant. In such circumstances, it makes no practical sense to maximize the accuracy of approximations during the characterization of any random variable under study and, much less, to expect a simulation to faithfully reproduce the distribution profile of a certain reference random variable. It is sufficient for all this to be done in an approximate manner.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


## EXTENDING THE MODEL

Fork the code to make the model more complicated, detailed, or accurate.


## RELATED MODELS

- Generalized Mean Metastable Switch Memristor Model (GMMS)
* T. Molter and A. Nugent, “The Mean Metastable Switch Memristor Model in Xyce,” Knowm web page, 2017. https://knowm.org/the-mean-metastable-switch-memristor-model-in-xyce/
 
Specifically, to write this code, I took as a starting point the GMMS model improved in:
* V. Ostrovskii, P. Fedoseev, Y. Bobrova, and D. Butusov, “Structural and Parametric Identification of Knowm Memristors,” Nanomaterials, Vol.12. No. 63, pp. 1–20, 2022.
https://doi.org/10.3390/nano12010063

## HOW TO CITE

If you mention this model in a publication, I ask that you include these citations for the model:

* Laguna-Sanchez, G.A. (2022).  Alpha-stable based stochastic GMMS memristor model.  https://github.com/galaguna/Stochastic_GMMS_memristor_model. 

## COPYRIGHT AND LICENSE

Copyright 2022 Gerardo Abel Laguna-Sanchez.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or(at your option) any later version. See <http://www.gnu.org/licenses/>
