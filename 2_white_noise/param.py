import numpy as np
from AAfunctions.fun_clean import *

## CUTOFF ##
th_t = 10**(-8)
method = 'Powell'

## TEMPORAL CORRELATION ##
A_t = 1
sigma_t = 1
t_cor = lambda dt, l: temp_white_correlations(l, dt, A_t, sigma_t)

##############
## Change L ##
##############

## PARTICLES ##
N_1 = np.logspace(0, 8, 49).astype(int)
N_1 = np.unique(N_1)

## RUNTIME ##
T_1 = 10**3

##############
## Change T ##
##############

## PARTICLES ##
N_2 = 100

## RUNTIME ##
T_2 = np.logspace(1, 5, 25)

## SAVING ##
name_file_N = "data_N.npy"
name_file_T = "data_T.npy"
