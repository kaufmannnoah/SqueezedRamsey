from AAfunctions.fun import *
from param import *
import numpy as np

out = np.ones((len(TT_list), len(TT_list[0]), 3))

for ind, T in enumerate(TT_list):
    t_cor = temp_cor[ind]
    for idt, t in enumerate(T):
        out[ind, idt, :] = opt_uncertainty_squeezedramsey_tcor_separable(N, t, t_cor, t0= 1, t_lim=[0.01, min(100, t)], th_t= th_t, method= method)
        for i in range(1):
            test = opt_uncertainty_squeezedramsey_tcor_separable(N, t, t_cor, t0= 1/(10 * i+1), t_lim=[0.001, min(100, t)], th_t= th_t, method= method)
            if test[0] < out[ind, idt, 0]: out[ind, idt, :] = test

np.save(name_file, out)