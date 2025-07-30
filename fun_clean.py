import numpy as np
from scipy.special import erf
import scipy as sc
import matplotlib.pyplot as plt


###########################
##  SPIN SQUEEZED STATE  ##
###########################

## SQUARED NORMALIZATION ##
def norm2(N, kappa):
    m = np.linspace(-N/2, N/2, num= N + 1)
    return 1 / np.sum(np.exp(-2 * m**2 / (N * kappa**2)))

## FIRST MOMENTS ##
def first_moments(N, kappa):
    J_x = 0
    J_y = 0
    m = np.linspace(-(N - 1)/2, (N - 1)/2, num= N)
    n2 = norm2(N, kappa)
    term1 = np.exp(-(2 * m**2  + 1/2) / (N * kappa**2))
    term2 = np.sqrt(1/4 * (N + 1)**2 - m**2)    
    return J_x, J_y, np.sum(-n2 * term1 * term2)

## SECOND MOMENTS ##
def second_moments(N, kappa):
    n2 = norm2(N, kappa)
    #J_yy
    m = np.linspace(-N/2, N/2, num= N+1)
    J_yy = n2 * np.sum(m**2 * np.exp(-2 * m**2 / (N * kappa**2)))
    #J_n: J+J-  J-J+
    J_n = N/2 * (N/2 + 1) - J_yy    
    #J_o: J+J+ = J-J-
    m = np.linspace(-N/2 + 1, N/2 - 1, num= N - 1)
    J_o = n2 * np.sum(np.exp(-2 * (m**2 + 1) / (N * kappa**2)) * np.sqrt((N/2 * (N/2 + 1))**2 - m**2 * N * (N/2 + 1) + m**2 * (m**2 - 1)))
    #J_xx
    J_xx = 1/2 * (J_n - J_o)
    #J_zz
    J_zz = 1/2 * (J_n + J_o)
    return J_xx, J_yy, J_zz

## SQUEEZED STATE FUNCTIONS ##
def squeezed_state(N, kappa):
    J_x, J_y, J_z = first_moments(N, kappa)
    J_xx, J_yy, J_zz = second_moments(N, kappa)
    return J_x**2, J_y**2, J_z**2, J_xx, J_yy, J_zz


###########################################
##  TEMPORAL NOISE CORRELATION FUNCTION  ##
###########################################

## WHITE NOISE ##
## NOISE SPECTRUM ##
def temp_white_spectrum(omega, A= 1.0, sigma= 0):
    return A * np.ones_like(omega)

## AUTOCORRELATION OF NOISE ##
def temp_white_autocorrelation(t, A= 1.0, sigma= 0):
    if np.isscalar(t):
        return temp_white_autocorrelation(np.array([t]), A, sigma)[0]
    temp = np.zeros_like(t, dtype=float)
    temp[np.where(t == 0)] = A
    return temp

## NOISE CORRELATIONS BETWEEN TWO EXPERIMENTS ##
def temp_white_correlations(l, tau, A= 1.0, sigma= 0):
    if np.isscalar(l):
        return temp_white_correlations(np.array([l]), tau, A, sigma)[0]
    temp = np.zeros_like(l, dtype=float)
    temp[np.where(l == 0)] = A * tau
    return temp

## GAUSSIAN NOISE ##
## NOISE SPECTRUM ##
def temp_gaussian_spectrum(omega, A= 1.0, sigma= 1.0):
    return A * np.exp(-omega**2 / (2 * sigma**2))

## AUTOCORRELATION OF NOISE ##
def temp_gaussian_autocorrelation(t, A= 1.0, sigma= 1.0):
    return A / np.sqrt(2 * np.pi) * sigma * np.exp(-t**2 * sigma**2 / 2)

## NOISE CORRELATIONS BETWEEN TWO EXPERIMENTS ##
def temp_gaussian_correlations(l, tau, A= 1.0, sigma= 1.0):
    term1 = np.sqrt(2 / (np.pi * sigma**2)) * (np.exp(-(l+1)**2 * sigma**2 * tau**2 /2) + np.exp(-(l-1)**2 * sigma**2 * tau**2 /2) - 2 * np.exp(-l**2 * sigma**2 * tau**2 / 2))
    term2 = tau * (l + 1) * erf((l + 1) * sigma * tau / np.sqrt(2)) + tau * (l - 1) * erf((l - 1) * sigma * tau / np.sqrt(2)) - 2 * tau * l * erf(l * sigma * tau / np.sqrt(2))
    return A / 2 * (term1 + term2)

## LINEAR NOISE ##
## NOISE SPECTRUM ##
def temp_linear_spectrum(omega, A= 1.0, sigma= 1.0):
    return A * (1 - np.exp(-np.abs(omega) / sigma))

## AUTOCORRELATION OF NOISE ##
def temp_linear_autocorrelation(t, A= 1.0, sigma= 1.0):
    if np.isscalar(t):
        return temp_linear_autocorrelation(np.array([t]), A, sigma)[0]
    temp = np.zeros_like(t, dtype=float)
    temp[np.where(t == 0)] = 1
    return A / np.pi * (np.pi * temp - sigma / (1 + sigma**2 * t**2))

## NOISE CORRELATIONS BETWEEN TWO EXPERIMENTS ##
def temp_linear_correlations(l, tau, A= 1.0, sigma= 1.0):
    if np.isscalar(l):
        return temp_linear_correlations(np.array([l]), tau, A, sigma)[0]
    temp = np.zeros_like(l, dtype=float)
    temp[np.where(l == 0)] = 1
    term1 = A * tau * temp
    term2 = A * tau / np.pi * (2 * l * np.arctan(sigma * l * tau) - (l-1) * np.arctan(sigma * (l-1) * tau) - (l+1) * np.arctan(sigma * (l+1) * tau))
    term3 = A / (2 * np.pi *sigma) * (np.log(1 + sigma**2 * tau**2 * (l + 1)**2) + np.log(1 + sigma**2 * tau**2 * (l - 1)**2) - 2 * np.log(1 + sigma**2 * tau**2 * l**2))
    return term1 + term2 + term3

## OHMIC NOISE ##
## NOISE SPECTRUM ##
def temp_ohmic_spectrum(omega, A= 1.0, sigma= 1.0):
    return A * np.abs(omega, dtype= float) / sigma * np.exp(-np.abs(omega, dtype= float) / sigma)

## AUTOCORRELATION OF NOISE ##
def temp_ohmic_autocorrelation(t, A= 1.0, sigma= 1.0):
    return A * sigma / np.pi * (1 - sigma**2 * t**2) / (1 + sigma**2 * t**2)**2

## NOISE CORRELATIONS BETWEEN TWO EXPERIMENTS ##
def temp_ohmic_correlations(l, tau, A= 1.0, sigma= 1.0):
    return A / (2 * np.pi * sigma) * (np.log(1 + (l + 1)**2 * sigma**2 * tau**2) + np.log(1 + (l - 1)**2 * sigma**2 * tau**2) - 2 * np.log(1 + l**2 * sigma**2 * tau**2)) 


###########################################
##  SPATIAL NOISE CORRELATION FUNCTION  ##
###########################################

## WHITE NOISE ##
## NOISE SPECTRUM ##
def spat_white_spectrum(k, A= 1.0, sigma= 0):
    return A * np.ones_like(k)

## CORRELATION OF NOISE ##
def spat_white_correlation(n, A= 1.0, sigma= 0):
    if np.isscalar(n):
        return spat_white_correlation(np.array([n]), A, sigma)[0]
    temp = np.zeros_like(n, dtype=float)
    temp[np.where(n == 0)] = A
    return temp

## GAUSSIAN NOISE ##
## NOISE SPECTRUM ##
def spat_gaussian_spectrum(k, A= 1.0, sigma= 1.0):
    return A * np.exp(-k**2 / (2.0 * sigma**2))

## CORRELATION OF NOISE ##
def spat_gaussian_correlation(n, A= 1.0, sigma= 1.0):
    if np.isscalar(n):
        return spat_gaussian_correlation(np.array([n]), A, sigma)[0]
    temp = np.zeros_like(n, dtype=float)
    args = np.where(n * sigma < 10)
    args2 = np.where(n * sigma >= 10)
    temp[args] = np.real(A * sigma / (2.0 * np.sqrt(2.0 * np.pi)) * np.exp(-n[args]**2 * sigma**2 / 2.0) * (erf((np.pi - 1.j * n[args] * sigma**2) / (np.sqrt(2.0) * sigma)) + erf((np.pi + 1.j * n[args] * sigma**2) / (np.sqrt(2.0) * sigma))))
    temp[args2] = A * sigma / np.sqrt(2 * np.pi) * (np.exp(-n[args2]**2 * sigma**2 / 2.0) - (-1)**n[args2] * np.sqrt(2.0) * np.pi * sigma / (np.pi**2 + n[args2]**2 * sigma**4) * np.exp(-np.pi**2/(2.0 * sigma**2)))
    return temp
    
## LINEAR NOISE ##
## NOISE SPECTRUM ##
def spat_linear_spectrum(k, A= 1.0, sigma= 1.0):
    return A * (1 - np.exp(np.abs(k) / sigma))

## CORRELATION OF NOISE ##
def spat_linear_correlation(n, A= 1.0, sigma= 1.0):
    if np.isscalar(n):
        return spat_linear_correlation(np.array([n]), A, sigma)[0]
    temp = np.zeros_like(n, dtype=float)
    temp[np.where(n == 0)] = 1
    return A * (temp - sigma / np.pi * (1.0 - (-1)**(n) * np.exp(-np.pi/sigma)) / (1.0 + sigma**2 * n**2))

## OHMIC NOISE ##
## NOISE SPECTRUM ##
def spat_ohmic_spectrum(k, A= 1.0, sigma= 1.0):
    return A * np.abs(k, dtype= float) / sigma * np.exp(-np.abs(k, dtype= float) / sigma)

## CORRELATION OF NOISE ##
def spat_ohmic_correlation(n, A= 1.0, sigma= 1.0):
    return A / np.pi * (sigma * (1 - sigma**2 * n**2) - np.exp(-np.pi / sigma) * (np.pi + sigma + np.pi * n**2 * sigma**2 - n**2 * sigma**3) * (-1)**n) / (1 + sigma**2 * n**2)**2


##############################################################
##  RAMSEY SPECTROSCOPY UNCERTAINTY CALCULATIONS SQUEEZING  ##
##############################################################

## FULL FORMULA ##
def uncertainty_squeezedramsey(N, kappa, T, tau, t_cor, s_cor, th_s = 10 ** (-8), th_t = 10 ** (-8)):
    # Squeezed moments
    if kappa == 1: J_z2, J_yy, J_zz = [N**2 / 4, N / 4, N**2 / 4] # Handle separable case separately, because kappa = 1 is only an approximation
    else: _, _, J_z2, _, J_yy, J_zz = squeezed_state(N, kappa)
    
    # Repetitions
    if tau > T: tau = T  # Ensure tau does not exceed total time
    L = int(T / tau) # Calculate the number of repetitions
    TT = L * tau  # Adjust total time to be a multiple of tau

    # create the spatial correlation array
    s_cor_arr = s_cor(np.arange(N))
    s_cor_arr_th_arg = np.where(np.abs(s_cor_arr) > th_s * s_cor_arr[0])[0]
    s_cor_arr_th = s_cor_arr[s_cor_arr_th_arg]
    s_cor_rest = N**2 - N - (2 * np.sum(N - s_cor_arr_th_arg[1:])) # number of connections bellow threshold


    # create the temporal correlation array
    t_cor_arr = t_cor(tau, np.arange(L))
    t_cor_arr_th_arg = np.where(np.abs(t_cor_arr) > th_t * t_cor_arr[0])[0]
    t_cor_arr_th = t_cor_arr[t_cor_arr_th_arg]

    # handle single particle case
    if N == 1:
        term1 = L / 4 * np.exp(s_cor_arr[0] * t_cor_arr[0])
        term3 = J_z2 / N * 2 * np.sum((L - np.arange(1, L)) * np.sinh(s_cor_arr[0] * t_cor_arr[1:]))
        return (term1 + term3) / (T**2 * J_z2)

    # Nominator
    term1  = N * L / 4 * np.exp(s_cor_arr[0] * t_cor_arr[0])

    term2  = - L / (4 * (N - 1)) * (2 * np.sum((N - s_cor_arr_th_arg[1:]) * np.exp(s_cor_arr_th[1:] * t_cor_arr[0])) + s_cor_rest)

    term3 = 2 * N * np.sum((L - np.arange(1, L)) * np.sinh(s_cor_arr[0] * t_cor_arr[1:]))
    for n in s_cor_arr_th_arg[1:]:
        term3 += 4 * (N - n) * np.sum((L - t_cor_arr_th_arg[1:]) * np.sinh(s_cor_arr[n] * t_cor_arr_th[1:]))
    term3 *= J_z2 / N**2 

    term4 = 2 * L * J_zz / (N * (N - 1)) * np.sum((N - s_cor_arr_th_arg[1:]) * np.sinh(s_cor_arr_th[1:] * t_cor_arr[0]))

    term5 = L * J_yy / (N * (N - 1)) * (2 * np.sum((N - s_cor_arr_th_arg[1:]) * np.cosh(s_cor_arr_th[1:] * t_cor_arr[0])) + s_cor_rest)

    # Denominator
    denom = TT**2 * J_z2

    return (term1 + term2 + term3 + term4 + term5) / denom

## ONLY TEMPORAL CORRELATIONS ##
def uncertainty_squeezedramsey_tcor(N, kappa, T, tau, t_cor, th_t = 10 ** (-8)):   
    # Squeezed moments
    if kappa == 1: J_z2, J_yy = [N**2 / 4, N / 4] # Handle separable case separately, because kappa = 1 is only an approximation
    else: _, _, J_z2, _, J_yy, _ = squeezed_state(N, kappa)
    
    # Repetitions
    if tau > T: tau = T  # Ensure tau does not exceed total time
    L = int(T / tau) # Calculate the number of repetitions
    TT = L * tau  # Adjust total time to be a multiple of tau

    # create the temporal correlation array
    t_cor_arr = t_cor(tau, np.arange(L))
    t_cor_arr_th_arg = np.where(np.abs(t_cor_arr) > th_t * t_cor_arr[0])[0]
    t_cor_arr_th = t_cor_arr[t_cor_arr_th_arg]

    # #Nominator
    term1  = N * L / 4 * (np.exp(t_cor_arr[0]) - 1)
    term2 = 2 * J_z2 / N * np.sum((L - t_cor_arr_th_arg[1:]) * np.sinh(t_cor_arr_th[1:]))
    term3 = L * J_yy

    # Denominator
    denom = TT**2 * J_z2

    return (term1 + term2 + term3) / denom


####################################
##  OPTIMAL UNCERTAINTY SQUEEZING ##
####################################

def opt_uncertainty_squeezedramsey(N, T, t_cor, s_cor, t0= 1, k0= 0.96, t_lim= [0.001, 10], k_lim= [0.001, 1], th_s= 10**(-10), th_t= 10**(-10), method= 'Powell'):
    opt_fun = lambda x: uncertainty_squeezedramsey(N, x[1], T, x[0], t_cor, s_cor, th_s= th_s, th_t= th_t)
    result = sc.optimize.minimize(opt_fun, np.array([t0, k0]), bounds=[(t_lim[0], t_lim[1]), (k_lim[0], k_lim[1])], method= method)
    return result.fun, result.x[1], result.x[0]

def opt_uncertainty_squeezedramsey_tcor(N, T, t_cor, t0= 1, k0= 0.96, t_lim= [0.001, 10], k_lim= [0.001, 1], th_t= 10**(-10), method= 'Powell'):
    opt_fun = lambda x: uncertainty_squeezedramsey_tcor(N, x[1], T, x[0], t_cor, th_t= th_t)
    result = sc.optimize.minimize(opt_fun, np.array([t0, k0]), bounds=[(t_lim[0], t_lim[1]), (k_lim[0], k_lim[1])], method= method)
    return result.fun, result.x[1], result.x[0]

def opt_uncertainty_squeezedramsey_separable(N, T, t_cor, s_cor, t0= 1, t_lim= [0.001, 10], th_s= 10**(-10), th_t= 10**(-10), method= 'Powell'):
    opt_fun = lambda x: uncertainty_squeezedramsey(N, 1, T, x[0], t_cor, s_cor, th_s= th_s, th_t= th_t)
    result = sc.optimize.minimize(opt_fun, np.array([t0]), bounds=[(t_lim[0], t_lim[1])], method= method)
    return result.fun, 1.0, result.x[0]

def opt_uncertainty_squeezedramsey_tcor_separable(N, T, t_cor, t0= 1, t_lim= [0.001, 10], th_t= 10**(-10), method= 'Powell'):
    opt_fun = lambda x: uncertainty_squeezedramsey_tcor(N, 1, T, x[0], t_cor, th_t= th_t)
    result = sc.optimize.minimize(opt_fun, np.array([t0]), bounds=[(t_lim[0], t_lim[1])], method= method)
    return result.fun, 1.0, result.x[0]

########################################################
##  RAMSEY SPECTROSCOPY UNCERTAINTY CALCULATIONS GHZ  ##
########################################################

## FULL FORMULA ##
def uncertainty_ghzramsey(N, T, tau, t_cor, th_t = 10 ** (-8)):
    # Repetitions
    if tau > T: tau = T  # Ensure tau does not exceed total time
    L = int(T / tau) # Calculate the number of repetitions
    TT = L * tau  # Adjust total time to be a multiple of tau

    # create the temporal correlation array
    t_cor_arr = t_cor(tau, np.arange(L))
    t_cor_arr_th_arg = np.where(np.abs(t_cor_arr) > th_t * t_cor_arr[0])[0]
    t_cor_arr_th = t_cor_arr[t_cor_arr_th_arg]

    term1 = L * np.exp(N * t_cor_arr[0])
    term2 = 2 * np.sum((L - t_cor_arr_th_arg[1:]) * np.sinh(t_cor_arr_th[1:] * N))
    return (term1 + term2) / (N**2 * TT**2)

## OPTIMIZATION ##
def opt_uncertainty_ghzramsey(N, T, t_cor, t0= 1, t_lim= [0.001, 10], th_t= 10**(-10), method= 'Powell'):
    opt_fun = lambda x: uncertainty_ghzramsey(N, T, x[0], t_cor, th_t= th_t)
    result = sc.optimize.minimize(opt_fun, np.array([t0]), bounds=[(t_lim[0], t_lim[1])], method= method)
    return result.fun, result.x[0]

#######################
## PLOTTING SETTINGS ##
#######################

# Define a function to set consistent plot styles
def set_plot_styles():
    plt.rcParams.update({
        #'mathtext.fontset': 'cm',
        'font.family': 'DejaVu Sans',
        'font.size': 8,          # Set font size
        'axes.linewidth': 0.5,      # Axis line width        
        'axes.titlesize': 9,     # Title font size
        'axes.labelsize': 8,     # Axis label font size
        'lines.linewidth': 1.5,   # Line width
        'lines.markersize': 1.5,    # Marker size
        'legend.edgecolor': 'gray',
        'xtick.labelsize': 7,     # X-tick label font size
        'ytick.labelsize': 7,     # Y-tick label font size
        'xtick.major.width': 0.5, # X-tick major line width
        'ytick.major.width': 0.5 # Y-tick major line width
    })

def colors():
    return ['#E69F00', '#0072B2', '#D55E00', '#009E73']