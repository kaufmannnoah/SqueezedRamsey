from AAfunctions.fun_clean import *
from param import *
import numpy as np

out = np.ones((len(T), len(N), 8))

for idt, t in enumerate(T):
    np.save(str(idt), np.ones(1))
    t_cor = temp_cor[idt]
    _, _, t0_sep = opt_uncertainty_squeezedramsey_tcor_separable(N[0], t, t_cor, t0= 0.2, t_lim=[0.01, 10], th_t= th_t, method= method)
    _, k0, t0 = opt_uncertainty_squeezedramsey_tcor(N[0], t, t_cor, t0= t0_sep, k0= 0.96, t_lim=[0.01, 10], k_lim=[0.9 / np.sqrt(N[0]), 1], th_t= th_t, method= method)
    t_max = t0_sep
    t0_ghz = t0_sep
    for idn, n in enumerate(N):
        out[idt, idn, :3] = opt_uncertainty_squeezedramsey_tcor(n, t, t_cor, t0= t0, k0= k0, t_lim=[t_max / (1.1 * np.sqrt(n)), 1.1 * t_max], k_lim=[1 / (1.1 * np.sqrt(n)), 1], th_t= th_t, method= method)
        out[idt, idn, 3:6] = opt_uncertainty_squeezedramsey_tcor_separable(n, t, t_cor, t0= t0_sep, t_lim=[t_max / (1.1 * np.sqrt(n)), 1.1 * t_max], th_t= th_t, method= method)
        out[idt, idn, 6:] = opt_uncertainty_ghzramsey(n, t, t_cor, t0= t0_ghz, t_lim=[t_max / (1.1 * n**(5/4)), 1.1 * t_max / np.sqrt(n)], th_t= th_t, method= method)
        _, k0, t0, _, _, t0_sep, _, t0_ghz = out[idt, idn]
np.save(name_file, out)