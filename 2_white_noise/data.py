from AAfunctions.fun_clean import *
from param import *
import numpy as np

out_N = np.ones((len(N_1), 6))
out_T = np.ones((len(T_2), 6))

## FIXED T ##
_, _, t0_sep = opt_uncertainty_squeezedramsey_tcor_separable(N_1[0], T_1, t_cor, t0= 0.2, t_lim=[0.1, 10], th_t= th_t, method= method)
_, k0, t0 = opt_uncertainty_squeezedramsey_tcor(N_1[0], T_1, t_cor, t0= t0_sep, k0= 0.96, t_lim=[0.1, 10], k_lim=[0.5, 1], th_t= th_t, method= method)
t_max = t0_sep
for idn, n in enumerate(N_1):
    if idn%1 == 0: np.save(str(n), np.ones(1))
    out_N[idn, :3] = opt_uncertainty_squeezedramsey_tcor(n, T_1, t_cor, t0= t0, k0= k0, t_lim=[t_max / (1.1 * np.sqrt(n)), 1.1 * t_max], k_lim=[1 / (1.1 * np.sqrt(n)), 1], th_t= th_t, method= method)
    out_N[idn, 3:] = opt_uncertainty_squeezedramsey_tcor_separable(n, T_1, t_cor, t0= t0_sep, t_lim=[t_max / (1.1 * np.sqrt(n)), 1.1 * t_max], th_t= th_t, method= method)
    _, k0, t0, _, _, t0_sep = out_N[idn]

## FIXED N ##
_, _, t0_sep = opt_uncertainty_squeezedramsey_tcor_separable(N_2, T_2[0], t_cor, t0= 0.2, t_lim=[0.1, 10], th_t= th_t, method= method)
_, k0, t0 = opt_uncertainty_squeezedramsey_tcor(N_2, T_2[0], t_cor, t0= t0_sep, k0= 0.96, t_lim=[0.1, 10], k_lim=[0.9 / np.sqrt(N_2), 1], th_t= th_t, method= method)
t_max = t0_sep
for idt, t in enumerate(T_2):
    if idt%10 == 0: np.save(str(t), np.ones(1))
    out_T[idt, :3] = opt_uncertainty_squeezedramsey_tcor(N_2, t, t_cor, t0= t0, k0= k0, t_lim=[t_max / (1.1 * np.sqrt(N_2)), 1.1 * t_max], k_lim=[1 / (1.1 * np.sqrt(N_2)), 1], th_t= th_t, method= method)
    out_T[idt, 3:] = opt_uncertainty_squeezedramsey_tcor_separable(N_2, t, t_cor, t0= t0_sep, t_lim=[t_max / (1.1 * np.sqrt(N_2)), 1.1 * t_max], th_t= th_t, method= method)
    _, k0, t0, _, _, t0_sep = out_T[idt]

np.save(name_file_N, out_N)
np.save(name_file_T, out_T)