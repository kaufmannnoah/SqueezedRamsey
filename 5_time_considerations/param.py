import numpy as np
from AAfunctions.fun_clean import *

## PARTICLES ##
N = 1

method = 'Powell'

## CUTOFF ##
th_t = 10**(-8)

## TEMPORAL CORRELATION ##
temp_cor = [] # Array of all temporal correlation functions
temp_spec = [] # Array of all sepectrums
si_ar = []
A_ar = []

#gaussian
si_t_g = 0.01
A_t_g = 10
temp_cor += [lambda dt, dn, A=A_t_g, si=si_t_g: temp_gaussian_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_g, si=si_t_g: temp_gaussian_spectrum(w, A, si)]
si_ar += [si_t_g]
A_ar += [A_t_g]

# white
si_t_w = 0.01
A_t_w = 10
temp_cor += [lambda dt, dn, A=A_t_w, si=si_t_w: temp_white_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_w, si=si_t_w: temp_white_spectrum(w, A, si)]
si_ar += [si_t_w]
A_ar += [A_t_w]

# linear
si_t_l = 0.01
A_t_l = 10
temp_cor += [lambda dt, dn, A=A_t_l, si=si_t_l: temp_linear_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_l, si=si_t_l: temp_linear_spectrum(w, A, si)]
si_ar += [si_t_l]
A_ar += [A_t_l]

# ohmic
si_t_o = 0.01
A_t_o = 10
temp_cor += [lambda dt, dn, A=A_t_o, si=si_t_o: temp_ohmic_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_o, si=si_t_o: temp_ohmic_spectrum(w, A, si)]
si_ar += [si_t_o]
A_ar += [A_t_o]

#Runtime
TT_list = [np.logspace(-1, 3, 41) / si_t_g, np.logspace(-1, 3, 41) / si_t_w, np.logspace(-1, 3, 41) / si_t_l, np.logspace(-1, 3, 41) / si_t_o]

## SAVING ##
name_file = "data.npy"