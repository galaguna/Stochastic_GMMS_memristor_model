# ******************************************************************
# Code to generate and characterize alpha-stable random variable
# ******************************************************************
# Programmers: G. Laguna-Sanchez and M. Lopez-Guerrero
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

#Function q() for fractile method
def frac_q_fun(i, N):
    """
    :param i: index of the sample in turn
    :param N: total number of samples
    :return: resulting quotient
    """
    return (2.0*i-1.0)/(2.0*N)

#Function to determine the sign of the value theta
def getsign(theta):
    """
    :param theta: value
    :return:
    -1 for negatives
    +1 for positives
    0 for zero
    """
    s = 0
    if theta < 0:
        s = -1
    else:
        if theta > 0:
            s = 1

    return s

#Simplified characteristic function for
# symmetric alpha stable variable with mu=0
def sym_S_alpha_func(theta,alpha,sigma):
    """
    :param theta: function domain variable
    :param alpha: stability parameter
    :param sigma: scale parameter
    :return: function value
    """
    return ((sigma*theta)**alpha)*np.tan(np.pi*alpha/2.0)

#Given the relation y=mx+b,
#estimate of the slope m and the constant b by linear regression
def lse_linear_regression(x,y):
    """
    :param x: vector with values for the x-axis
    :param y: vector with values for y-axis
    :return:
    p_m: estimated slope
    p_b: estimated constant b
    """

    p_m = 0
    p_b = 0

    if (x.shape == y.shape) and (x.shape[0] == 1 or x.shape[1] == 1):

        if x.shape[0] == 1:
            x_row = x
            y_row = y
        else:
            x_row = x.T
            y_row = y.T

        N = x_row.shape[1]
        uc = np.ones((N, 1))

        Sxx = np.dot(x_row, x_row.T)
        Sxy = np.dot(x_row, y_row.T)

        Sx = np.dot(x_row, uc)
        Sy = np.dot(y_row, uc)


        x_mean = np.mean(x)
        y_mean = np.mean(y)

        p_m = (Sxy-(Sx*Sy/N))/(Sxx-(Sx**2)/N)
        p_b = y_mean-(p_m*x_mean)

    return p_m[0][0], p_b[0][0]

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

#Obtain the PMF from the characteristic function of a random variable.
def PMF_fromCharac(Charac_func):
    """
    :param Charac_func: complex sequence of even length
    :return: PMF sequence
    """
    N = len(Charac_func)
    half_ix = N//2
    F = np.concatenate([Charac_func[0:half_ix+1], np.conj(Charac_func[half_ix-2 : : -1])])
    f = np.fft.ifft(F)
    PMF = np.abs(np.concatenate([f[half_ix : ], f[0 : half_ix]]))

    return PMF

#Characteristic function for alpha-stable random variable with variable mu shift
def Stable_Characteristic(omega,alpha,sigma,beta,mu):
    """
    :param omega: vector with sequence for omega values
    :param alpha: stability parameter
    :param sigma: skewness parameter
    :param beta: scale parameter
    :param mu: shift parameter
    :return: vector with function values
    """
    N = len(omega)
    Phi = np.empty(N, dtype=complex)
    if alpha == 1.0:
        for i in range(N):
            Phi[i] = np.exp(-sigma * np.abs(omega[i]) * (1.0 - 1.0j * beta * (2.0 / np.pi) * getsign(omega[i]) * np.log(np.abs(omega[i]))) + 1.0j * mu * omega[i])

    else:
        for i in range(N):
            Phi[i] = np.exp(-(sigma ** alpha) * ((np.abs(omega[i])) ** alpha) * (1.0 + 1.0j * beta * getsign(omega[i]) * np.tan(np.pi * alpha / 2.0)) + 1.0j * mu * omega[i])

    return Phi

#Characteristic function for Kogon's criterion
def alpha_sigma_sampled_characteristic_func(samples,Psi_len):
    """
    :param samples: vector with samples of the random variable
    :param Psi_len: number of samples of the Psi characteristic function
    :return:
    x: values for the x-axis of the Psi function
    y: values for the y-axis of the Psi function
    """
    N_samples = len(samples)
    step = (1.0 - 0.1) / (Psi_len - 1)
    omega = 0.1

    PSI = np.empty(Psi_len, dtype=complex)
    x = np.zeros(Psi_len)
    y = np.zeros(Psi_len)

    for k in range(Psi_len):
        PSI[k] = 0.0
        for n in range(N_samples):
            tmp = omega * samples[n]
            PSI[k] += np.cos(tmp) + 1.0j * np.sin(tmp)

        PSI[k] /= N_samples

        x[k] = np.log(omega)
        y[k] = np.log(-np.log(np.sqrt((np.imag(PSI[k])) ** 2 + (np.real(PSI[k])) ** 2)))

        omega += step

    return x, y

#Characteristic function for Koutrouvelis regression algorithm
def beta_mu_sampled_characteristic_func(samples,Psi_len,alpha,sigma):
    """
    :param samples: vector with samples of the random variable
    :param Psi_len: number of samples of the Psi characteristic function
    :param alpha: estimation for stability parameter
    :param sigma: estimation for scale parameter
    :return:
    x: values for the x-axis of the Psi function
    y: values for the y-axis of the Psi function
    """
    N_samples = len(samples)
    step = (1.0 - 0.1) / (Psi_len - 1)
    omega = 0.1

    PSI = np.empty(Psi_len, dtype=complex)
    x = np.zeros(Psi_len)
    y = np.zeros(Psi_len)

    for k in range(Psi_len):
        PSI[k] = 0.0
        for n in range(N_samples):
            tmp = omega * samples[n]
            PSI[k] += np.cos(tmp) + 1.0j * np.sin(tmp)

        PSI[k] /= N_samples

        x[k] = sym_S_alpha_func(omega, alpha, sigma)
        y[k] = np.arctan(np.imag(PSI[k]) / np.real(PSI[k]))

        omega += step

    return x, y

#Generate roulette array for stable alpha random variable X~S_alpha(alpha,beta,sigma,mu)
def build_S_alpha_roulette(alpha,beta,sigma,mu,max_dev,N_samples):
    """
    :param alpha: stability parameter
    :param sigma: scale parameter
    :param beta: skewness parameter
    :param mu: shift parameter
    :param max_dev: Maximum absolute deviation from mu value
    :param N_samples: size of the resulting roulette array
    :return: the roulette array
    """

    omega_max = (1.0/sigma)*100 #Empirically obtained ratio, which guarantees a suitable interval to contain the Phi function profile with enough slack.
    quantization_levels = 5000  #Minimum resolution to work well
    omega_step = omega_max / quantization_levels
    X_step = (2 * max_dev) / quantization_levels

    x = np.arange(mu - max_dev + X_step , mu + max_dev + X_step, X_step) #Range for alpha stable variable

    #Generate cacarteristic function for reference:
    w = np.arange(omega_step, omega_max, omega_step)   #Range for the parameter omega (frequency) in the characteristic function
    Phi = Stable_Characteristic(w, alpha, sigma, beta, 0.0) #Reference Phi function is calculated with mu=0

    #Generate PMF:
    PMF = PMF_fromCharac(Phi)

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

    #******************************************************************
    # Final adjustment
    #******************************************************************
    #First estimation to ajdust sigma:
    m_alpha, m_beta, m_sigma, m_mu = S_alpha_identification(ST)
    ST *= (sigma / m_sigma)
    #Second estimation to ajdust mu:
    m_alpha, m_beta, m_sigma, m_mu = S_alpha_identification(ST)
    ST += (mu - m_mu)

    return ST

#Generate alpha-stable variable
def S_alpha_sample(roulette_array):
    """
    :param roulette_array: array to generate a stable alpha variable X~S_alpha(alpha,beta,sigma,mu)
    :return: random stable alpha variable
    """
    size = len(roulette_array)
    ix = int(np.random.rand(1) * size)
    return roulette_array[ix]

#Function for estimating alpha, beta and scale parameters of alpha-stable variables,
#on the basis of McCulloch-Kogon-Koutruvelis algorithms
def S_alpha_identification(samples):
    """
    :param samples: samples vector of a random variable
    :return:
    alpha: stability parameter
    scale: scale parameter (sigma or gamma)
    beta: skewness parameter
    mu: shift parameter
    """
    Size = len(samples);
    PSI_LEN = 40; #Sufficient size to give good results
    samples_mean = np.mean(samples);
    mu = samples_mean;
    sequence = np.sort(samples);

    #**************************************************
    #McCulloch's method by fractiles:
    #**************************************************

    #With 0.72 fractil:
    i_max = (2.0 * Size * 0.72 + 1.0) / 2.0
    i_min = (2.0 * Size * 0.72 - 1.0) / 2.0
    ix = int((i_min + i_max) / 2.0)
    fractile_72 = sequence[ix] + (sequence[ix + 1] - sequence[ix]) * ((0.72 - frac_q_fun(ix, Size)) / (frac_q_fun(ix + 1, Size) - frac_q_fun(ix, Size)))

    #With 0.28 fractil:
    i_max = (2.0 * Size * 0.28 + 1.0) / 2.0
    i_min = (2.0 * Size * 0.28 - 1.0) / 2.0
    ix = int((i_min + i_max) / 2.0)
    fractile_28 = sequence[ix] + (sequence[ix + 1] - sequence[ix]) * ((0.28 - frac_q_fun(ix, Size)) / (frac_q_fun(ix + 1, Size) - frac_q_fun(ix, Size)))

    #Coarse estimation for alpha and corresponding normalization:
    sigma = (fractile_72 - fractile_28) / 1.654
    sequence = (sequence - samples_mean) / sigma

    #**************************************************
    #Kogon's criterion for frequency sampling:
    #**************************************************

    x_samples, y_samples = alpha_sigma_sampled_characteristic_func(sequence,PSI_LEN)
    x_row = np.array([x_samples])
    y_row = np.array([y_samples])

    #**************************************************
    #Koutrouvelis regression method:
    #**************************************************

    [m, b] = lse_linear_regression(x_row, y_row)
    alpha = m
    scale = np.exp(b / m) * sigma

    x_samples, y_samples = beta_mu_sampled_characteristic_func(sequence, PSI_LEN, alpha, np.exp(b / m))
    x_row = np.array([x_samples])
    y_row = np.array([y_samples])

    [m, b] = lse_linear_regression(x_row, y_row)
    beta = m

    return alpha,beta,scale,mu

#Generate noise sequence with alpha-stable distribution
def S_alpha_noise(len,alpha,beta,sigma,mu,max_dev):
    """
    :param len: length of the required noise sequence
    :param alpha: stability parameter (0 < alpha <= 2)
    :param beta: skewness parameter (-1 <= beta <= 1)
    :param sigma: scaling parameter (sigma >= 0)
    :param mu: shift parameter (any real number)
    :param max_dev: Maximum absolute deviation from mu value
    :return: the noise sequence
    """
    samples = 1000  # Roulette size
    #Generate prototype roulette:
    S_alpha_roulette = build_S_alpha_roulette(alpha,beta,sigma,mu,max_dev,samples)

    #Synthesize noise:

    n = np.zeros(len)
    for i in range(len):
        n[i] = S_alpha_sample(S_alpha_roulette)


    return n

