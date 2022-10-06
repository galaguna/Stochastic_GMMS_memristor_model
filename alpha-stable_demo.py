# ******************************************************************
# Demo code to generate and characterize alpha-stable random variable
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
import matplotlib.pyplot as plt
import MonteCarlo_S_alpha_rand_core as sa

if __name__ == '__main__':
    # Test bench area
    print("Wellcome to alpha-stable noisy world ")

    n = sa.S_alpha_noise(len=1000, alpha=1.5, beta=1.0, sigma=1.0, mu=0.0, max_dev=15.0)
    m_alpha, m_beta, m_sigma, m_mu = sa.S_alpha_identification(n)
    print("Estimated m_alpha=%f, m_beta=%f, m_sigma=%f, m_mu=%f"%(m_alpha, m_beta, m_sigma, m_mu))

    plt.figure(1)
    plt.plot(n)
    plt.title('Alpha-stable sequence')
    plt.ylabel('Value')
    plt.xlabel('Samples')

    plt.figure(2)
    plt.plot(np.sort(n))
    plt.title('CDF profile')
    plt.ylabel('Value')
    plt.xlabel('Samples')
    plt.grid()
    plt.show()
