#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.optimize import minimize
from scipy.optimize import least_squares
from iminuit.cost import LeastSquares
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from astropy.io import fits

import argparse
import os
import sys
import time


#######################################


plt.rcParams['figure.figsize'] = (8, 6)   
plt.rcParams['font.size'] = 16         
plt.rcParams['axes.titlesize'] = 16     
plt.rcParams['legend.fontsize'] = 14    
plt.rcParams['savefig.dpi'] = 150       
plt.rcParams['axes.labelsize'] = 16      


#######################################


DESI_EDR_PARAMETERS = (
    7.63089e-02, -2.52054e+00, -1.27968e-01,
    3.67469e+00, 2.85951e-01, 7.33473e+02)

# set pivot points for k and z using fiducial power estimate
PD13_PIVOT_K = 0.009  # note: k_0
PD13_PIVOT_Z = 3.0    # note: z0 = 3

gh_degree = 25
gausshermite_xi_deg, gausshermite_wi_deg = np.polynomial.hermite.hermgauss(int(gh_degree))
YY1, YY2 = np.meshgrid(gausshermite_xi_deg, gausshermite_xi_deg, indexing='ij')
WW1, WW2 = np.meshgrid(gausshermite_wi_deg, gausshermite_wi_deg, indexing='ij')

# set defaults for p1d fit
default_numvpoints = 2**12
default_dv = 12
default_v_array = np.arange(default_numvpoints) * default_dv
k_arr  = 2. * np.pi * np.fft.rfftfreq((2 * default_numvpoints)-1, d=default_dv) + 1e-12 

flux_fitting_z_array = np.linspace(1.8, 5.0, 500)


#######################################


def read_data(filename, expect_two_columns=False):
    """Reads data from a file (.txt or .fits). Returns a 1D or 2D array."""
    if filename.endswith('.txt'):
        data = np.loadtxt(filename)
    elif filename.endswith('.fits'):
        with fits.open(filename) as hdul:
            data = hdul[1].data  # Assuming the first extension holds the table
    else:
        raise ValueError("Unsupported file format. Use .txt or .fits")

    # If the data is supposed to be two columns (e.g. power spectrum or flux), check it
    if expect_two_columns:
        if data.ndim == 1:
            raise ValueError(f"Expected two columns in {filename}, but found only one.")
        if data.shape[1] != 2:
            raise ValueError(f"Expected two columns in {filename}, but found {data.shape[1]}.")
        return data[:, 0], data[:, 1]  # Return the two columns as separate arrays

    # Otherwise, return as a 1D array (for redshift bins, for example)
    if data.ndim > 1:
        raise ValueError(f"Expected a 1D array from {filename}, but got shape {data.shape}.")
    return data.flatten()  # Return a flattened 1D array


def parse_redshift_target(input_value):
    """Parses the redshift target argument, allowing either a file or a single float value."""
    if os.path.isfile(input_value):  # If it's a file, read the data
        return read_data(input_value)
    try:
        # Try converting input to a float
        z_value = float(input_value)
        return np.array([z_value])  # Convert single float to an array
    except ValueError:
        raise ValueError("Invalid input for --redshift_bins. Provide a valid .txt/.fits file or a single float value.")


def turner24_mf(z):
    tau_0 = -2.46e-3
    gamma = 3.62
    return np.exp(tau_0 * (1 + z)**gamma)    


def evaluatePD13Lorentz(X, A, n, alpha, B, beta, lmd):
    k, z = X
    q0 = k / PD13_PIVOT_K + 1e-10

    result = (A * np.pi / PD13_PIVOT_K) * np.power(
        q0, 2. + n + alpha * np.log(q0)) / (1. + lmd * k**2)

    if z is not None:
        x0 = (1. + z) / (1. + PD13_PIVOT_Z)
        result *= np.power(q0, beta * np.log(x0)) * np.power(x0, B)

    return result
    

def PD13Lorentz_DESI_EDR(zlist):
    # Start with an empty array the size / shape of input k and z arrays
    p1d_edr_fit = np.empty((zlist.size, k_arr.size))
    
    # Evaluate P1D for each (k,z), using DESI EDR Param. def. above
    for i, z in enumerate(zlist):
        p1d_edr_fit[i] = evaluatePD13Lorentz((k_arr, z), *DESI_EDR_PARAMETERS)

    return p1d_edr_fit


def a2_z(zp, nu = 2.82, z0 = PD13_PIVOT_Z):
    return np.power((1. + zp) / (1.+z0), -nu)


def a_z(zp, nu = 2.82, z0 = PD13_PIVOT_Z):
    return np.sqrt(np.power((1. + zp) / (1.+z0), -nu))


def t_of_z(zp, tau0 = 0.55, tau1 = 5.1, z0 = PD13_PIVOT_Z):
    return tau0 * np.power((1. + zp) / (1.+z0), tau1)


def n_z(zp, nu, sigma2, z0 = PD13_PIVOT_Z):
    return np.exp(-a2_z(zp, nu, z0) - sigma2)


def x_of_z(zp, tau0, tau1, nu, sigma2, z0 = PD13_PIVOT_Z):
    return t_of_z(zp, tau0, tau1, z0) * np.exp(-a2_z(zp, nu, z0) * sigma2)


def Flux_d_z(delta_g, z, tau0, tau1, nu, sigma2, z0 = PD13_PIVOT_Z):
    e1 = np.exp(2 * a2_z(z, nu / 2, z0) * np.sqrt(2 * sigma2) * delta_g)
    e2 = x_of_z(z, tau0, tau1, nu, sigma2, z0)
    return np.exp(-e2 * e1)


def lognMeanFluxGH(z, tau0, tau1, nu, sigma2, z0 = PD13_PIVOT_Z, degree = gh_degree):
    XIXI, ZZ = np.meshgrid(gausshermite_xi_deg, z)
    Y = Flux_d_z(XIXI, ZZ, tau0, tau1, nu, sigma2, z0)
    result = np.dot(Y, gausshermite_wi_deg)

    return result / np.sqrt(np.pi)
    

def process_power_file(power_file): 
    if power_file:
        try:
            data = np.loadtxt(power_file)  
            if data.shape[1] < 3:  # Ensure at least 3 columns
                print("Error: Power spectrum file must have at least 3 columns (z, k, P1D).")
                return {}, np.array([])

            zlist = data[:, 0]  
            k_array = data[:, 1]
            P1D_array = data[:, 2] 

            grouped_data = {}
            for i, z in enumerate(zlist):
                if z not in grouped_data:
                    grouped_data[z] = {'k': [], 'P1D': []}
                grouped_data[z]['k'].append(k_array[i])
                grouped_data[z]['P1D'].append(P1D_array[i])

            return grouped_data, np.unique(zlist)  # Ensure zlist is unique
        except Exception as e:
            print(f"Error reading power spectrum file: {e}")
            return {}, np.array([])  # Return empty dict and array
    else:
        print("\nNo (P,k,z) file provided, using default model (DESI EDR)\n")
        return {}, np.array([])  # Return empty dict and array


def compute_velocity_properties(k_array):
    """
    Compute velocity grid properties from an input k_array.

    Input:
        k_array 

    Returns:
        dv (float): Velocity spacing
        numvpoints (int): Number of velocity points
    """
    numvpoints = len(k_array)  
    
    if numvpoints < 2:
        raise ValueError("k_array must have at least two points to determine spacing.")

    # Determine dv 
    k_min, k_max = np.min(k_array), np.max(k_array)
    
    # Check if k-array is uniformly spaced
    dk_values = np.diff(k_array)
    if np.allclose(dk_values, dk_values[0]):  
        dk = dk_values[0]
        dv = np.pi / (dk * numvpoints) 
    else:
        # More general case: Use range of k values 
        print('k-array is not evenly spaced')

    return dv, numvpoints

    
def fit_mean_flux(flux_array, flux_z_array, z0):
    flux_fit_precision = 1e-5
    Err_flux_fit = flux_array * flux_fit_precision + 1e-8

    def flux_fit_cost(tau0, tau1, nu, sigma2):
        d = (flux_array - lognMeanFluxGH(flux_z_array, tau0, tau1, nu, sigma2, z0)) / Err_flux_fit
        return d.dot(d)
            
    # Set initial guesses for fitting parameters
    tau0_0, tau1_0, nu_0, sigma2_0 = 0.55, 5.1, 2.82, 2.0

    mini = Minuit(flux_fit_cost, tau0_0, tau1_0, nu_0, sigma2_0)
    mini.errordef = Minuit.LEAST_SQUARES
    mini.migrad()

    return mini.values


#######################################


def plot_mean_flux(z, target_flux, bestfit_values, flux_model, z0 = PD13_PIVOT_Z): 
    print('Saving: Mean_Flux_Fit.png')
    tau0, tau1, nu, sigma2 = bestfit_values
    plt.figure()
    plt.plot(z, lognMeanFluxGH(z, *bestfit_values, z0), color='tab:blue', 
             ls='-', label='Best Fit', lw='6', alpha = 0.5)
    plt.plot(z, target_flux, color='black', ls='--', label=f'Model: {flux_model}')
    plt.xlabel('z')
    plt.ylabel(r'$\bar F(z) $')
    plt.legend(loc='upper right')
    txt=f'tau0 = {tau0}\ntau1 = {tau1}\nnu = {nu}\nsigma2 = {sigma2}'
    plt.text(0.02, 0.1, txt, ha='left', va='center', fontsize=14, transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig('Mean_Flux_Fit')


def plot_target_power(z, k_array_input, p1d_input, k_array_fine, p1d_fine):
    print(rf'Saving: {z}_P1D_target.png')
    plt.figure()
    
    plt.loglog(k_array_input, p1d_input, alpha = 0.7, 
               label=f'input P1D, z={z}, N={k_array_input.size}')
    plt.loglog(k_array_fine, p1d_fine, color='tab:orange', 
               label=f'interpolated, z={z}, N={k_array_fine.size}', ls='--')
    plt.legend()
    plt.xlabel(r'k $[km/s]^{-1}$')
    plt.ylabel(r'$P_{1D}$ ')
    plt.savefig(rf'{z}_P1D_target.png')


def plot_target_xif(z, new_v_array, xif_interp_fit, v_array_downsampled, xif_target_downsampled, dv):
    print(rf'Saving: {z}_xi_F_target.png')
    plt.figure()
    plt.semilogx(new_v_array[1:], xif_interp_fit[1:], 
                 label='interpolated, N =' +str(new_v_array.size))
    plt.semilogx(v_array_downsampled, xif_target_downsampled, 
                 label='downsampled, N =' +str(v_array_downsampled.size), ls='--', marker='o')
    plt.xlabel('v [km/s]')
    plt.ylabel(r'$\xi_F$')
    plt.title(r'$\xi_F$ Target,  z = '+str(z))
    plt.vlines(dv, 0, 0.20, color='black', ls='--', label=f'dv = {dv:.2f} km/s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(rf'{z}_xi_F_target.png')


def plot_xif_fit(z, v_array_downsampled, xi_f_target, xi_f_optimized, dv):
    print(rf'Saving: {z}_xi_F_fit.png')
    plt.figure()
    plt.semilogx(v_array_downsampled, xi_f_target, alpha=0.5, 
                 label = r'$\xi_F$ Target', color = 'tab:blue')
    plt.semilogx(v_array_downsampled, xi_f_optimized,  marker='o', 
                 label=r"$\xi_F$ Fit", color="tab:orange", ls='--')
    plt.axvline(x=dv, color='black', linestyle='--', 
                label=f"dv: {dv:.2f} km/s")
    plt.legend()
    plt.xlabel('v [km/s]')
    plt.ylabel(r'$\xi_F$')
    plt.title(rf'$\xi_F$ Fit vs Data  (z = {z})')
    plt.tight_layout()
    plt.savefig(rf'{z}_xi_F_fit.png')

    dif_xi_f = xi_f_target - xi_f_optimized
    abs_dif_xi_f = np.abs(dif_xi_f)
    max_y = dif_xi_f[np.argmax(abs_dif_xi_f)] 
    max_x = v_array_downsampled[np.argmax(abs_dif_xi_f)]
    
    print(rf'Saving: {z}_xi_F_fit_residual.png')
    plt.figure()
    plt.semilogx(v_array_downsampled, dif_xi_f, 
                 label = f'z = {z}', color = 'tab:blue')
    plt.axvline(x=dv, color='black', linestyle='--', 
                label=f"dv: {dv:.2f} km/s")
    plt.xlabel('v [km/s]')
    plt.ylabel(r'$\Delta \xi_F$')
    plt.legend()
    plt.title(r"$\xi_F$ Residual (Model - Fit)")
    plt.tight_layout()
    plt.savefig(rf'{z}_xi_F_fit_residual.png')


def plot_xig_fit(z, v_array_downsampled, xi_g_optimized, 
                 zero_point, v_extrapolated, xi_g_extrapolated):
    print(rf'Saving: {z}_xi_G_fit.png')
    plt.figure()
    plt.plot(v_array_downsampled, xi_g_optimized, 
             'o', label=rf'$\xi_g$ Fit')
    plt.plot(0, zero_point, 'ro', 
             label=f'Fixed Point (0, {zero_point:.3f})')
    plt.plot(v_extrapolated, xi_g_extrapolated, '-', label='CS Extrapolation')
    plt.xscale('log')  
    plt.xlabel('v [km/s]')
    plt.ylabel(r'$\xi_g$')
    plt.legend()
    plt.grid()
    plt.ylim(None, xi_g_extrapolated.max()+0.1)
    plt.tight_layout()
    plt.savefig(rf'{z}_xi_G_fit.png')


def plot_xi_f_recovered(z, v_fine, xif_fine, 
                        v_array_downsampled, xif_target_downsampled, 
                        v_extrapolated, xi_f_optimized_extrapolated, dv):
    print(rf'Saving: {z}_xi_F_recovered.png')
    plt.figure()
    plt.semilogx(v_fine, xif_fine, label=f're-interp., N = {v_fine.size}', 
                 color='tab:green', lw=5, alpha=0.3)
    plt.semilogx(v_array_downsampled, xif_target_downsampled, 
                 label=f'downsampled Target, N = {v_array_downsampled.size}', 
                 ls='-', marker='o', color='tab:blue')
    plt.semilogx(v_extrapolated, xi_f_optimized_extrapolated, 
                 label=r"$\xi_F$ Fit, N = "+str(v_extrapolated.size), 
                 color="tab:orange", ls='--')
    plt.axvline(x=dv, color='black', linestyle='--', label=f"dv: {dv:.2f}")
    plt.xlabel('v [km/s]')
    plt.ylabel(r'$\xi_F$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(rf'{z}_xi_f_recovery.png')


def plot_recovered_power(z, k_array_input, p1d_input, w_k, mirrored_fit_k_arr, 
                         mirrored_fit_power, w_fit_k, e_p1d, z_id, delta_P_real):
    print(rf'Saving: {z}_recovered_power.png')
    temp_k = k_array_input[w_k]
    temp_p = p1d_input[w_k]
    temp_e = np.full_like(temp_k, e_p1d[z_id])
    
    log_k_values = np.log10(temp_k) 
    log_k_min = np.min(log_k_values)
    log_k_max = np.max(log_k_values)
    alpha_values = 0.05 + (1 - 0.2) * (log_k_max - log_k_values) / (log_k_max - log_k_min)

    plt.figure()
    for k_val, p_val, err, alpha_val in zip(temp_k, temp_p, temp_e, alpha_values):
                plt.errorbar(k_val, p_val, yerr=err, color='tab:blue', alpha=alpha_val)
    plt.loglog(k_array_input[w_k], p1d_input[w_k], color='tab:blue', 
                       label=f'Model') 
    plt.loglog(mirrored_fit_k_arr[w_fit_k], mirrored_fit_power[w_fit_k], 
                       color='tab:orange', ls='--', label = f'Best Fit') 
    plt.axvspan(0.05, 0.1, alpha=0.2, color='grey')
    plt.ylabel(rf'$P(k)$   (z = {z})')
    plt.xlabel(r'k $[km/s]^{-1}$')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(rf'{z}_recovered_power.png')
    
    print(rf'Saving: {z}_power_residual.png')
    plt.figure()
    plt.semilogx(k_array_input, delta_P_real, label = f"(Model - Best Fit) / Model")
    plt.axvspan(0.05, 1.0, alpha=0.2, color='grey')
    plt.xlim([10e-5,10e-2])
    plt.ylim(-delta_P_real.max(),delta_P_real.max())
    plt.xlabel(r'k $[km/s]^{-1}$')
    plt.ylabel(rf"$\Delta$ P / P   (z = {z})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(rf'{z}_power_residual.png')   


#######################################


def main():
    parser = argparse.ArgumentParser(description="Process P1D and k arrays from file with redshift parameters.")
    parser.add_argument('--power_file', type=str, help='Path to input (P,k) file (.txt) containing k and P1D arrays')
    parser.add_argument('--flux_file', type=str, help='Path to input (F,z) file (.txt) containing z and mean flux arrays')
    parser.add_argument('--z_target', type=str, required=True, help='Path to input file (.txt) containing target redshift values, OR a single float input')
    
    parser.add_argument('--plot_mean_flux', action='store_true', help='Generate and save a figure of the Mean Flux (default: False)')
    parser.add_argument('--plot_target_power', action='store_true', help='Generate and save a figure of the target P1D (default: False)')
    parser.add_argument('--plot_target_xif', action='store_true', help='Generate and save a figure of the target Xi_F (default: False)')
    parser.add_argument('--plot_xig_fit', action='store_true', help='Generate and save a figure of the Xi_G fit (default: False)')
    parser.add_argument('--plot_xif_fit', action='store_true', help='Generate and save a figure of the Xi_F fit (default: False)')
    parser.add_argument('--plot_xif_recovered', action='store_true', help='Generate and save a figure of the Xi_F, recovered from Xi_G best fit (default: False)')
    parser.add_argument('--plot_recovered_power', action='store_true', help='Generate and save a figure of the P1D, recovered from Xi_F best fit (default: False)')
    
    args = parser.parse_args()

    z0 = PD13_PIVOT_Z
    
    ### Process redshift target (required) ###
    try:
        z_target = parse_redshift_target(args.z_target)
    except Exception as e:
        print(f"Error reading redshift file: {e}")
        return

    
    ### Process power data (optional) ###
    grouped_data, zlist = process_power_file(args.power_file) 

    if not grouped_data:
        # If no P1D file provided, use default model
        k_array = k_arr
        zlist = z_target
        P1D_array = PD13Lorentz_DESI_EDR(z_target)
        
        # Create dv and numvpoints arrays filled with default values
        dv_array = np.full(len(zlist), default_dv)
        numvpoints_array = np.full(len(zlist), default_numvpoints)

        # Check if k_array and P1D_array have matching lengths for each redshift
        for i, z in enumerate(zlist):
            if len(k_array) != len(P1D_array[i]):
                print(f"Error: Mismatch - k array has {len(k_array)} elements, but P1D array has {len(P1D_array[i])} elements.")
                return
    else: 
        # process grouped data from file (if applicable)
        print('\n(P,k,z) file provided, using as target P1D\n')
    
        P1D_array = []  
        k_array = []  
        dv_array = []
        numvpoints_array = []
    
        for z in z_target:
            if z in grouped_data:
                k_values = np.array(grouped_data[z]['k'])
                P1D_values = np.array(grouped_data[z]['P1D'])
        
                if k_values.size == 0:
                    print(f"Warning: k_values is empty for redshift {z}")
                
                k_array.append(k_values)
                P1D_array.append(P1D_values)
                
                # Compute velocity properties
                dv, numvpoints = compute_velocity_properties(k_values)
                dv_array.append(dv)
                numvpoints_array.append(numvpoints)  
            
            else:
                print(f"Warning: No data found for redshift {z}")
                sys.exit(1)
                
        P1D_array = np.array(P1D_array, dtype=object)
        k_array = np.array(k_array, dtype=object)
        dv_array = np.array(dv_array)
        numvpoints_array = np.array(numvpoints_array)

        
    ### Process flux data  (optional) ###
    if args.flux_file:
        print('\n(F,z) file provided, using as target mean flux')
        try:
            flux_z_array, flux_array = read_data(args.flux_file, expect_two_columns=True)
            flux_model = 'User Input'
        except Exception as e:
            print(f"Error reading flux file: {e}")
            return
    else:
        print(f"No (F,z) file provided, using default model (Turner et al., 2024)\n")
        flux_array = turner24_mf(flux_fitting_z_array)
        flux_z_array = flux_fitting_z_array
        flux_model = 'Turner et al., 2024'

    if len(flux_z_array) != len(flux_array):
        print(f"Error: Mismatch - z array has {len(flux_z_array)} elements, but flux array has {len(flux_array)} elements.")
        return

    print("\nData successfully assigned.")
    
    
    ### FIT MEAN FLUX  ###
    print('\n###  Fitting Mean Flux Parameters  ###\n')
    
    tau0, tau1, nu, sigma2 = fit_mean_flux(flux_array, flux_z_array, z0)

    print(f'tau0 = {tau0}')
    print(f'tau1 = {tau1}')
    print(f'nu = {nu}')
    print(f'sigma2 = {sigma2}')

    ##############################################
    if args.plot_mean_flux:
        plot_mean_flux(flux_z_array, flux_array, 
                       fit_mean_flux(flux_array, flux_z_array, z0), flux_model)
    ##############################################    

    # now define these functions with values from mean flux fit above
    def lognXiFfromXiG_pointwise(z, xi_gauss, tau0, tau1, nu, sigma2, z0):
        """
        Arguments
        ---------
        z: float
            Single redshift
        tau0, tau1: float
            Amplitude (tau0) and power (tau1) of optical depth
        nu: float
            Slope of growth (a(z) -> D(z))
        xi_gauss: float
            Single xi_g value from Gaussian random field
        """
        xi_sine = np.clip(xi_gauss / sigma2, -1, 1)
        xi_cosine = np.sqrt(1 - xi_sine**2)
        XI_VEC = np.array([xi_sine, xi_cosine])
    
        YY2_XI_VEC_WEIGHTED = np.dot(XI_VEC, np.array([YY1, YY2]).transpose(1, 0, 2))
    
        # Redshift-dependent computations
        mean_flux_z = lognMeanFluxGH(z, tau0, tau1, nu, sigma2, z0)
        sigmaz = a2_z(z, nu / 2, z0) * np.sqrt(sigma2)
        tempxz = x_of_z(z, tau0, tau1, nu, sigma2, z0)
        delta1 = YY1 * sigmaz * 2 * np.sqrt(2)
        delta2 = YY2_XI_VEC_WEIGHTED * sigmaz * 2 * np.sqrt(2)
    
        F1 = np.exp(-tempxz * np.exp(delta1))
        F2 = np.exp(-tempxz * np.exp(delta2))
        D1 = F1 / mean_flux_z - 1
        D2 = F2 / mean_flux_z - 1
        tempfunc = WW1 * WW2 * D1 * D2
    
        # Return a single xi_f scalar
        xi_f = np.sum(tempfunc) / np.pi
        return xi_f

    def objective(xi_g, z, xi_f_target, tau0, tau1, nu, sigma2, z0):
        """
        Compute the difference between the target xi_f values and the xi_f values
        calculated from xi_g using lognXiFfromXiG_pointwise.
    
        Parameters
        ----------
        xi_g : array-like
            Current guess for xi_g values.
        z : float
            Redshift at which xi_f is calculated.
        xi_f_target : array-like
            Target xi_f values, in this case, xif_edr_fit.
    
        Returns
        -------
        array-like
            Residuals between calculated and target xi_f.
        """
        xi_f_calculated = np.array([lognXiFfromXiG_pointwise(z, xi_g_i, tau0, tau1, nu, sigma2, z0) for xi_g_i in xi_g])
        return xi_f_calculated - xi_f_target

    
    # FITTING POWER 
    print("\n\n###  Fitting Power  ###")
    
    for z in z_target:
        idx = np.where(z_target == z)[0]
        redshift_index = idx[0]
        safe_z = str(z_target[redshift_index]).replace('.', '-')  # For naming figures
        
        dv = dv_array[redshift_index]
        numvpoints = numvpoints_array[redshift_index]
        
        # print('redshift index = '+str(redshift_index) + ', corresponding to z = '+str(z_target[redshift_index]))
        print(f"\nProcessing redshift: {z}\n")
        print(f'Mean Flux (z = {z}): {lognMeanFluxGH(z, tau0, tau1, nu, sigma2, z0)[0]}')

        if grouped_data:
            k_array_input = k_array[redshift_index]
            k_array_input = np.array(k_array_input, dtype=float)
            p1d_input = P1D_array[redshift_index]
        else: 
            k_array_input = k_array[1:]
            p1d_input = P1D_array[redshift_index][1:]
            
        # interpolate input power to a set size
        new_size = 2**20
        new_v_array = np.arange(new_size) * dv
        k_array_fine = np.linspace(k_array_input.min(), k_array_input.max(), new_size)
        
        cs = CubicSpline(np.ravel(k_array_input), np.ravel(p1d_input))
        p1d_fine = cs(k_array_fine)
        
        ##############################################
        if args.plot_target_power:
            plot_target_power(safe_z, k_array_input, 
                              p1d_input, k_array_fine, p1d_fine)
        ##############################################

        # calculate target xi_f from p_f
        xif_interp_fit = (np.fft.irfft(p1d_fine))[:new_size] / dv

        # downsample xi_f (logarithmically)
        downsample_size = 2**5

        velocity_abs = np.abs(new_v_array[1:])  
        log_v_min, log_v_max = np.log10(1 + velocity_abs.min()), np.log10(1 + velocity_abs.max())
        v_array_downsampled = np.logspace(log_v_min, log_v_max, downsample_size) - 1  # Shift back

        cs = CubicSpline(velocity_abs, xif_interp_fit[1:]) 
        xif_target_downsampled = cs(v_array_downsampled) 

        dv_downsampled = np.diff(v_array_downsampled)  # Compute dv
        
        ##############################################
        if args.plot_target_xif:
            plot_target_xif(safe_z, new_v_array, xif_interp_fit, 
                            v_array_downsampled, xif_target_downsampled, dv)
        ##############################################
        
        # now solve for xi_G
        print('\n(Fitting xi_f)')
        print(f"N-Points:     {downsample_size}")
        time_1 = time.strftime("%H:%M:%S")
        print("Start Time:  ", time_1)
        start_time = time.time()

        xi_f_target = xif_target_downsampled
        xi_g_initial_guess = np.full(xi_f_target.size, 0.1)
        
        result = least_squares(objective, xi_g_initial_guess, args=(z_target[redshift_index], xi_f_target, tau0, tau1, nu, sigma2, z0))
        xi_g_optimized = result.x

        time_2 = time.strftime("%H:%M:%S")
        print("End Time:    ", time_2)
        
        end_time = time.time()  
        elapsed_time = end_time - start_time 
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Elapsed Time: {minutes} min {seconds} sec\n")

        print(f"sigma2 from Mean Flux fit:      {sigma2}")
        print(f"Calculated sigma2 from Xi_G(0): {xi_g_optimized[0]}")
        print(f"Difference:                     {np.abs(sigma2-xi_g_optimized[0])}\n")
       
        xi_f_optimized = np.array([lognXiFfromXiG_pointwise(z_target[redshift_index], xi_g_i, sigma2, tau0, tau1, nu, z0) for xi_g_i in xi_g_optimized])
        
        ##############################################
        if args.plot_xif_fit:
            plot_xif_fit(safe_z,  v_array_downsampled, 
                         xi_f_target, xi_f_optimized, dv)
        ##############################################
        
        # extend the xi_g fit to zero 
        linear_extrapolation = interp1d(v_array_downsampled, xi_g_optimized, kind='linear', fill_value="extrapolate")
        zero_point = linear_extrapolation(0)
        print(f"zero_point: {zero_point}")
       
        v_extrapolated = np.logspace(np.log10(1), np.log10(max(v_array_downsampled)), num=100)  
        v_extrapolated = np.insert(v_extrapolated, 0, 0) 

        xi_g_new_vals = np.insert(xi_g_optimized, 0, zero_point)
        v_extended = np.insert(v_array_downsampled, 0, 0)

        spline = CubicSpline(v_extended, xi_g_new_vals, bc_type='natural')
        xi_g_extrapolated = spline(v_extrapolated)

        # save (v_extrapolated, xi_g_extrapolated) to txt file
        export_data = np.column_stack((v_extrapolated, xi_g_extrapolated))
        np.savetxt(rf'{safe_z}_output.txt', export_data, fmt="%.6f", delimiter="\t", header="Velocity\tXi_G_fit")

        ##############################################
        if args.plot_xig_fit:
            plot_xig_fit(safe_z, v_array_downsampled, xi_g_optimized, 
                         zero_point, v_extrapolated, xi_g_extrapolated)
        ##############################################
        
        # recover xi_f
        xi_f_optimized_extrapolated = np.array([lognXiFfromXiG_pointwise(z_target[redshift_index], xi_g_i, tau0, tau1, nu, sigma2, z0) for xi_g_i in xi_g_extrapolated])

        v_fine = np.linspace(v_extrapolated.min(), v_extrapolated.max(), 2**20) 
        cs = CubicSpline(v_extrapolated, xi_f_optimized_extrapolated)  
        xif_fine = cs(v_fine)  

        ##############################################
        if args.plot_xif_recovered:
             plot_xi_f_recovered(safe_z, v_fine, xif_fine, v_array_downsampled, 
                                 xif_target_downsampled, v_extrapolated, 
                                 xi_f_optimized_extrapolated, dv)
        ##############################################
        
        # recover P_F (k)
        dv_fine = np.diff(v_fine) 
        new_dv = np.mean(dv_fine)

        fit_power = np.fft.rfft(xif_fine) * new_dv * 2
        fit_k_arr = 2 * np.pi * np.fft.rfftfreq(len(xif_fine), d=new_dv)
        
        p1d_precision = 1e-1
        w_k = (k_array_input > 1e-4) & (k_array_input < 0.05)     # Window for k_array
        w_fit_k = (fit_k_arr > 1e-4) & (fit_k_arr < 0.05)         # Window for fit_k_arr
        w_k_fine = (k_array_fine > 1e-4) & (k_array_fine < 0.05)  # Window for k_array_fine
        ptrue = p1d_input[w_k]
        e_p1d = p1d_precision * ptrue + 1e-8

            
        # mirror xi_f and then recover P1D
        v_spacing = v_fine[1] - v_fine[0]  # Compute the step size
        v_mirrored = np.concatenate([v_fine, v_fine + v_fine[-1] + v_spacing])
        xi_f_mirrored = np.concatenate([xif_fine, xif_fine[::-1]])
        mirrored_fit_power = np.fft.rfft(xi_f_mirrored) * v_spacing 
        mirrored_fit_k_arr = 2 * np.pi * np.fft.rfftfreq(len(xi_f_mirrored/2), d=v_spacing)
        
        w_k = (k_array_input > 1e-4) & (k_array_input < 0.05)  
        w_fit_k = (mirrored_fit_k_arr > 1e-4) & (mirrored_fit_k_arr < 0.05)  
        w_k_fine = (k_array_fine > 1e-4) & (k_array_fine < 0.05) 

        # # optional: extend to smaller scales (higher k)
        # w_k = (k_array_input > 1e-4) & (k_array_input < 0.1)  
        # w_fit_k = (mirrored_fit_k_arr > 1e-4) & (mirrored_fit_k_arr < 0.1) 
        # w_k_fine = (k_array_fine > 1e-4) & (k_array_fine < 0.1) 
        
        # Compute residual
        fit_power_interp_2 = np.interp(k_array_input, mirrored_fit_k_arr, mirrored_fit_power)
        delta_P = np.real((p1d_input.real - fit_power_interp_2.real) / p1d_input.real)
        delta_P_real = delta_P.real
        
        ##############################################
        if args.plot_recovered_power:
            plot_recovered_power(safe_z, k_array_input, p1d_input, w_k, 
                                 mirrored_fit_k_arr, mirrored_fit_power, 
                                 w_fit_k, e_p1d, redshift_index, delta_P_real)        
        ##############################################

if __name__ == "__main__":
    main()
