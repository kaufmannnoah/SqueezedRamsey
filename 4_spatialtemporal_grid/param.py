import numpy as np
from AAfunctions.fun_clean import *

## PARTICLES ##
#N= np.logspace(0, 3, 16).astype(int)
#N= np.logspace(0, 5, 26).astype(int)
#N= np.logspace(0, 2, 10).astype(int)
N= np.logspace(0, 4, 25).astype(int)


N = np.unique(N)

## RUNTIME ##
T = np.array([[10**4, 10**4, 10**4, 10**5], [10**4, 10**4, 10**4, 10**5], [10**4, 10**4, 10**3, 10**3], [10**4, 10**4, 10**3, 10**3]])*2

method = 'Powell'

## CUTOFF ##
th_t = 10**(-8)
th_s = 10**(-8)

## TEMPORAL CORRELATION ##
temp_cor = [] # Array of all temporal correlation functions
temp_spec = [] # Array of all sepectrums

#gaussian
si_t_g = 0.5
A_t_g = 4
temp_cor += [lambda dt, dn, A=A_t_g, si=si_t_g: temp_gaussian_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_g, si=si_t_g: temp_gaussian_spectrum(w, A, si)]

# white
si_t_w = 0.5
A_t_w = 4
temp_cor += [lambda dt, dn, A=A_t_w, si=si_t_w: temp_white_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_w, si=si_t_w: temp_white_spectrum(w, A, si)]

# linear
si_t_l = 0.5
A_t_l = 4
temp_cor += [lambda dt, dn, A=A_t_l, si=si_t_l: temp_linear_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_l, si=si_t_l: temp_linear_spectrum(w, A, si)]

# ohmic
si_t_o = 0.5
A_t_o = 4
temp_cor += [lambda dt, dn, A=A_t_o, si=si_t_o: temp_ohmic_correlations(dn, dt, A, si)]
temp_spec += [lambda w, A=A_t_o, si=si_t_o: temp_ohmic_spectrum(w, A, si)]

## SPATIAL CORRELATION ##
spat_cor = []
spat_spec = []

#gaussian
si_s_g = 0.5
A_s_g = 4
spat_cor += [lambda n, A=A_s_g, si=si_s_g: spat_gaussian_correlation(n, A, si)]
spat_spec += [lambda k, A=A_s_g, si=si_s_g: spat_gaussian_spectrum(k, A, si)]

# white
si_s_w = 0.5
A_s_w = 4
spat_cor += [lambda n, A=A_s_w, si=si_s_w: spat_white_correlation(n, A, si)]
spat_spec += [lambda k, A=A_s_w, si=si_s_w: spat_white_spectrum(k, A, si)]

# linear
si_s_l = 0.5
A_s_l = 4
spat_cor += [lambda n, A=A_s_l, si=si_s_l: temp_linear_correlations(n, A, si)]
spat_spec += [lambda k, A=A_s_l, si=si_s_l: temp_linear_spectrum(k, A, si)]

# ohmic
si_s_o = 0.5
A_s_o = 4
spat_cor += [lambda n, A=A_s_o, si=si_s_o: temp_ohmic_correlations(n, A, si)]
spat_spec += [lambda k, A=A_s_o, si=si_s_o: temp_ohmic_spectrum(k, A, si)]

## SAVING ##
name_file = 'data.npy'
