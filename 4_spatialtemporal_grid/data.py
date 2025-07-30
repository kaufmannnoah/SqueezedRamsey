from fun_clean import *
from param import *
import numpy as np

out = np.ones((len(spat_cor), len(temp_cor), len(N), 6))

for id_s in range(len(spat_cor)):
    for id_t in range(len(temp_cor)):
        np.save(str(10 * (id_s + 1) + id_t + T[id_s, id_t]), np.ones(1))
        _, _, t0_sep = opt_uncertainty_squeezedramsey_separable(N[0], T[id_s, id_t], temp_cor[id_t], spat_cor[id_s], t0= 0.02, t_lim=[0.001, 10], th_t= th_t, th_s= th_s, method= method)
        _, k0, t0 = opt_uncertainty_squeezedramsey(N[0], T[id_s, id_t], temp_cor[id_t], spat_cor[id_s], t0= t0_sep, k0= 0.96, t_lim=[0.01, 10], k_lim=[0.9 / np.sqrt(N[0]), 1], th_t= th_t, th_s= th_s, method= method)
        t_max = t0_sep
        for idn, n in enumerate(N):
            out[id_s, id_t, idn, :3] = opt_uncertainty_squeezedramsey(n, T[id_s, id_t], temp_cor[id_t], spat_cor[id_s], t0= t0, k0= k0, t_lim=[t_max / (1.1 * np.sqrt(n)), 1.1 * t_max], k_lim=[1 / (1.1 * np.sqrt(n)), 1], th_t= th_t, th_s= th_s, method= method)
            out[id_s, id_t, idn, 3:] = opt_uncertainty_squeezedramsey_separable(n, T[id_s, id_t], temp_cor[id_t], spat_cor[id_s], t0= t0_sep, t_lim=[t_max / (1.1 * np.sqrt(n)), 1.1 * t_max], th_t= th_t, th_s= th_s, method= method)
            _, k0, t0, _, _, t0_sep = out[id_s, id_t, idn]
np.save(name_file, out)
