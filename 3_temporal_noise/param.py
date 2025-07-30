import numpy as np
from AAfunctions.fun_clean import *

## PARTICLES ##
N = np.logspace(0, 5, 26).astype(int)
N = np.unique(N)

## RUNTIME ##
T = [10**4, 10**4, 10**4, 10**4]
method = 'Powell'
th_t = 10**(-10)

## TEMPORAL CORRELATION ##
temp_cor = [] # Array of all temporal correlation functions
temp_spec = [] # Array of all sepectrums

#gaussian
si_t_g = 0.5
A_t_g = 1.0
temp_cor += [lambda dt, dn, A=A_t_g, si=si_t_g: temp_gaussian_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_g, si=si_t_g: temp_gaussian_spectrum(w, A, si)]


# white
si_t_w = 1.0
A_t_w = 1.0
temp_cor += [lambda dt, dn, A=A_t_w, si=si_t_w: temp_white_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_w, si=si_t_w: temp_white_spectrum(w, A, si)]

# linear
si_t_l = 0.5
A_t_l = 1.0
temp_cor += [lambda dt, dn, A=A_t_l, si=si_t_l: temp_linear_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_l, si=si_t_l: temp_linear_spectrum(w, A, si)]

# ohmic
si_t_o = 2.0
A_t_o = 5.0
temp_cor += [lambda dt, dn, A=A_t_o, si=si_t_o: temp_ohmic_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_o, si=si_t_o: temp_ohmic_spectrum(w, A, si)]


## SAVING ##
name_file = "data.npy"
