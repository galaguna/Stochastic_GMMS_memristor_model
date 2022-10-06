# ******************************************************************
# Code to generate and characterize gaussian random variable
# ******************************************************************
# Programmer: G. Laguna-Sanchez
# Date: Oct 06, 2022
# Universidad Automoma Metropolitana
# Mexico
# **********************************************************************
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See <http://www.gnu.org/licenses/>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
# **********************************************************************

import numpy as np

#Generate gaussian pdf
def gaussian_pdf(mean, dev_std, x):
    """
    :param mean: mean or expectation
    :param dev_std: standard deviation
    :param x: vector with random variable in a desired range
    :return: probability distribution function
    """
    N = len(x)
    pdf= np.zeros(N)
    p1 = 1.0 / (dev_std * np.sqrt(2*np.pi))
    p2 = 1.0 / (2 * dev_std ** 2)
    for i in range(N):
        pdf[i] = p1 * np.exp(-p2 * (x[i]-mean)**2)

    return pdf


#Calculate the CDF from a PMF by numerical integration
def CDF_from(PMF):
    """
    :param PMF: PMF sequence
    :return: integrated PMF sequence
    """
    acc = 0
    dcf = np.zeros(len(PMF))
    for i in range(len(PMF)):
        acc += PMF[i]
        dcf[i] = acc
    dcf *= (1.0/acc) #Normalization
    return dcf

#Linearly interpolate a value of x (in the domain) starting from a value of the function (contradomain) and from two points
#This returns the maximum value indicated when y2==y1
def xinterpolation_from(y,x1,y1,x2,y2,max):
    """
    :param y: function value
    :param x1: x coordinate for Point 1
    :param y1: y coordinate for Point 1
    :param x2: x coordinate for Point 2
    :param y2: y coordinate for Point 2
    :param max: maximium value
    :return: interpolation value
    """
    if (x2 == x1) and (y2 == y1):
        x = x1
    else:
        if (y2 == y1):
            x = max
        else:
            x = x1 + (y - y1) * ((x2 - x1) / (y2 - y1))

    return x


#Generate roulette array for gaussian random variable
def build_gaussian_roulette(mean,dev_std,max_dev,N_samples):
    """
    :param mean: mean or expectation
    :param dev_std: standard deviation
    :param max_dev: Maximum absolute deviation from mu value
    :param N_samples: size of the resulting roulette array
    :return: the roulette array
    """

    quantization_levels = 1000   #Resolution
    X_step = (2 * max_dev) / quantization_levels

    x = np.arange(mean-max_dev + X_step , mean+max_dev + X_step, X_step) #Range for gaussian variable

    #Generate PMF:
    PMF = gaussian_pdf(mean, dev_std, x)

    #Generate CDF:
    CDF_func = CDF_from(PMF)

    #Build table:
    ST = np.zeros(N_samples)
    y_delta = 1.0 / N_samples
    y = 0.0
    for n in range(N_samples):
        y += y_delta
        where_bigerthan_y = np.where(CDF_func > y)
        if (len(where_bigerthan_y[0]) > 0):
            ix = where_bigerthan_y[0][0]
        else:
            ix = len(CDF_func)-1

        if ix > 0:
            x2 = x[ix]
            y2 = CDF_func[ix]
            x1 = x[ix - 1]
            y1 = CDF_func[ix - 1]

            ST[n] = xinterpolation_from(y, x1, y1, x2, y2, max_dev)
        else:
            x2 = x[1]
            y2 = CDF_func[1]
            x1 = x[0]
            y1 = CDF_func[0]

            ST[n] = xinterpolation_from(y, x1, y1, x2, y2, max_dev)

    return ST

#Generate gaussian variable
def gaussian_sample(roulette_array):
    """
    :param roulette_array: array to generate a gaussian variable X~gaussian(mean,dev_std)
    :return: random gaussian variable
    """
    size = len(roulette_array)
    ix = int(np.random.rand(1) * size)
    return roulette_array[ix]


