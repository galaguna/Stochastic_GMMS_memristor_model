#******************************************************************
# Stochastic simulator for Knowm memristor based on the GMMS model
# with snapbacks modeling according to Ostrovskii el al:
#   Structural and Parametric Identification of Knowm Memristors. \
#   Nanomaterials 2022, 12, 63. https://doi.org/10.3390/nano12010063
#******************************************************************
# Programmer: G. Laguna-Sanchez
# Date: Sept 13, 2023
# Universidad Automoma Metropolitana
# Unidad Lerma, Mexico
#******************************************************************
# Includes random variables with both normal and alpha-stable distributions.
#******************************************************************
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See <http://www.gnu.org/licenses/>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
#**************************************************************************
#Schematic diagram for reference
#**************************************************************************
#
#                      (Vm)
#                       |
#                       |
#        (V)_____MMM____|______po______(GND)
#                 Rs        Memristor
#
#**************************************************************************
#
import sys
import matplotlib.pyplot as plt
import numpy as np
import MonteCarlo_S_alpha_rand_core as sa
import MonteCarlo_Gaussian_rand_core as gn


#*****************************************************
# Playground area:
#*****************************************************

if __name__ == '__main__':
    print("Hello, simulating Knowm memristor with stochastic GMMS")

    # *****************************************************
    # Parameters for simulation:
    # *****************************************************

    #Physical constants:
    q = 1.6 * 10**(-19) #Elemental charge
    k = 1.3 * 10**(-23) #Boltzman's constant

    #Constants for the GMMS model:
    alpha_f = 10**(-7)
    alpha_r = 10**(-7)
    beta_f = 8
    beta_r = 8
    phi = 0.88

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #Specific parameters and global variables
    #for stochastic behavior:
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Roulette_segments = 1000 #Quantization levels for the generated random numbers
    m_dev = 0.2 #Maximum deviation from statistical mode

    #V_OFF as a alpha-stable variable:
    m_alpha = 1.535     #From estimation
    m_beta = -0.609     #From estimation
    m_sigma = 0.013     #From estimation
    m_mu = -0.082       #From estimation
    Stable_Roulette_Table= sa.build_S_alpha_roulette(alpha=m_alpha, beta=m_beta, sigma=m_sigma, mu=m_mu, max_dev=m_dev, N_samples=Roulette_segments)

    #V_ON as a gaussian variable:
    m_mean = 0.217      #From estimation
    m_dev_std = 0.035   #From estimation
    Gaussian_Roulette_Table= gn.build_gaussian_roulette(mean=m_mean,dev_std=m_dev_std,max_dev=m_dev,N_samples=Roulette_segments)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #Specific parameters for the simulation run:
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Ron = 13000 #ON state resistence
    Roff = 460000 #OFF state resistence
    Tao = 0.00006 #Memristor time constant
    SamplesByTao = 30 #Implicit sampling resolution
    #T= 298.5 #Temperature in Kelvin degrees
    # (Note: The typical average ambient temperature does not
    #        result in the experimentally observed behavior.)
    T= 108.5 #Temperature in Kelvin degrees
    Rs = 46250  #Resistor in series with the memristor
    Vp = 0.7  #Input peak voltage
    Reseted = True  #Initial state of the memristor


    # *****************************************************
    # Working variables for simulation:
    # *****************************************************
    dt= Tao/SamplesByTao # Sampling period
    dt2Tao = dt / Tao
    beta = q /(k * T) #Variable of MMS model
    t = np.arange(0.0, 0.18, dt)   #Time vector
    size = len(t)
    V = Vp * np.sin(2*np.pi*10*t) #10Hz sinusoidal signal and peak voltage Vp

    X = np.zeros(size)  #Vector for memristor state variable X
    G = np.zeros(size)  #Vector for memristor conductance
    I = np.zeros(size)  #Vector for series current
    Vm = np.zeros(size)  #Vector for memristor voltage

    #*****************************************************
    #Graphication:
    #*****************************************************
    plt.figure(1)
    plt.title('Hysteresis Im vs Vm')
    plt.ylabel('Im [Ampers]')
    plt.xlabel('Vm [Volts]')

    # *****************************************************
    # N Timeline simulations:
    # *****************************************************
    N = 2  #For N simulations
    for n in range(N):
        print("Simulation %i of %i" % (n+1,N))
        #Initial condition:
        if Reseted:
            X[0] = 0.0
            G[0] = (1/Roff)
        else:
            X[0] = 1.0
            G[0] = (1/Ron)
        #*****************************************************
        #Simulation with resolution dt= Tao/SamplesByTao:
        #*****************************************************
        #Voff = -0.1    #Static typical value for V_OFF
        Voff= sa.S_alpha_sample (Stable_Roulette_Table) #Random value for V_OFF as alpha-stable variable
        #Von_threshold = 0.2 #Static typical threshold value for V_ON
        Von_threshold = gn.gaussian_sample(Gaussian_Roulette_Table) #Random value for V_ON as gaussian variable

        for i in range(1, size):

            #Voltage divider:
            if G[i-1] != 0.0: #Division by zero validation
                Rm = 1.0 / G[i-1]   #Resistance of the memristor according to the latest update
            else:
                Rm = sys.float_info.max;   #Maximum value as infinite
            I[i] = V[i] / (Rm + Rs) #Current in series circuit
            Vm[i] = V[i] * Rm / (Rm + Rs) #Voltage drop across the memristor

            #Update state variable and conductance of the memristor (MMS model):
            sqrtX = np.sqrt(X[i-1])
            Von = Von_threshold + (0.1 * np.cos(4 * np.pi * sqrtX / (1.7 - X[i-1]))) /(1 + 10 * sqrtX)  #V_ON according Ostrovskii el al
            X[i] = dt2Tao * ((1- X[i-1]) * (1/(1 + np.exp(-beta *(Vm[i]-Von)))) - X[i-1] * (1-(1/(1 + np.exp(-beta *(Vm[i]-Voff)))))) + X[i-1]
            G[i] = (X[i]/Ron)+((1-X[i])/Roff)

            #Schottky diode effect adjustment (GMMS model):
            Is = alpha_f * np.exp(beta_f * Vm[i]) - alpha_r * np.exp(beta_r * Vm[i]) #Schottky diode current
            I[i] = phi * I[i] + (1.0 - phi) * Is #The total current flowing through Rs is adjusted
            Vm[i] = V[i] - Rs * I[i] #The voltage drop across the memristor is adjusted
            G[i] += Is / Vm[i] #The total equivalent conductance of the memristor is adjusted


        plt.plot(Vm, I)

    plt.show()

