#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import scipy as sp
from scipy.fft import fft, ifft, rfft, irfft
from scipy.stats import binned_statistic 
from scipy import integrate
from scipy.interpolate import interp1d
from numpy.polynomial.hermite import hermgauss
from pathlib import Path

import argparse
import os
import sys
import time
from datetime import datetime
import random

#######################################

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.labelsize'] = 16 
plt.rcParams['savefig.bbox'] = 'tight'

#######################################

PD13_PIVOT_K = 0.009  # note: k_0
PD13_PIVOT_Z = 3.0    # note: z0 = 3

c = 299792.458  # speed of light in km/s
lambda_0 = 1216 # rest wavelength in Angstroms (for Lyα)
lambda_min = 3600  # minimum wavelength in Angstroms
lambda_max = 9800  # maximum wavelength in Angstroms

size = 2**20
dv=1
velocity_grid = np.arange(size) - size/2
v_min = (lambda_min / lambda_0 - 1) * c
v_max = (lambda_max / lambda_0 - 1) * c


min_slice = 65000
max_slice = 70000

gh_degree = 25
gausshermite_xi_deg, gausshermite_wi_deg = np.polynomial.hermite.hermgauss(
    int(gh_degree))
YY1, YY2 = np.meshgrid(gausshermite_xi_deg, gausshermite_xi_deg, indexing='ij')
WW1, WW2 = np.meshgrid(gausshermite_wi_deg, gausshermite_wi_deg, indexing='ij')

# # set defaults for p1d fit
# # default_numvpoints = 2**20
# default_dv = 1.0
# default_numvpoints = 2**12
# # default_dv = 10.0
# default_v_array = np.arange(default_numvpoints) * default_dv
# k_arr = 2. * np.pi * \
#     np.fft.rfftfreq((2 * default_numvpoints)-1, d=default_dv) + 1e-12

#######################################


def parse_redshift_target(input_value):
    """
    Parses the redshift target argument, allowing:
    - A single float value (e.g., "2.0").
    - A comma-separated list of redshift values (e.g., "2.0,2.2,2.4").
    - A file containing redshift values (.txt)

    Args:
        input_value (str): file path, single value, or comma-separated 
                string of  values.

    Returns:
        np.ndarray: An array of target redshift values.

    Raises:
        ValueError: If the input is neither a valid file nor a properly 
                formatted float/comma-separated list.
    """
    # Check if the input is a file
    if os.path.isfile(input_value):
        return read_data(input_value)

    # Check if input contains multiple comma-separated redshifts
    if "," in input_value:
        try:
            redshifts = np.array([float(z.strip())
                                 for z in input_value.split(",")])
            return redshifts
        except ValueError:
            raise ValueError(
                "Invalid input for --redshift_bins. Ensure all values are "
                "valid floats separated by commas."
            )

    # Otherwise, try to parse it as a single float
    try:
        return np.array([float(input_value)])
    except ValueError:
        raise ValueError(
            "Invalid input for --redshift_bins. Provide a valid .txt/.fits file, "
            "a single float value, or a comma-separated list of floats.")


# def process_cf_file(safe_z, user_path=None):
#     """
#     Loads and processes a correlation function file into k and P(k) arrays.

#     Args:
#         safe_z (str): Redshift label with dots replaced by dashes.
#         user_path (str or Path, optional): Optional user-specified file path.

#     Returns:
#         tuple: (k_array, power_array)
#     """
#     try:
#         if user_path:
#             cf_path = Path(user_path)
#         else:
#             cf_path = Path(f"{safe_z}_xiG_full_output.txt")

#         if not cf_path.exists():
#             raise FileNotFoundError(f"Could not find file: {cf_path}")

#         data = np.loadtxt(cf_path)  
#         v = data[:, 0]
#         xiG = data[:, 1]
#         v_spacing = v[1] - v[0]

#         power_array = np.fft.rfft(xiG) * v_spacing
#         # k_array = 2 * np.pi * np.fft.rfftfreq(len(xiG), d=v_spacing)
#         k_array = 2 * np.pi * np.fft.rfftfreq(len(xiG), d=v_spacing)

#         assert len(power_array) == len(k_array), "Mismatch between P(k) and k array!"

#         plt.loglog(k_array, power_array, label='test')
#         plt.legend()
#         plt.savefig('test.png')
#         return k_array, power_array.real

#     except Exception as e:
#         raise RuntimeError(f"Failed to process CF file for z={safe_z}:\n{e}")


def process_power_file(safe_z, user_path=None):
    """
    Loads and processes a Gaussian power file into k and P(k) arrays for later use.

    Args:
        safe_z (str): Redshift label with dots replaced by dashes.
        user_path (str or Path, optional): Optional user-specified file path.

    Returns:
        tuple: (k_array, power_array)
    """
    try:
        if user_path:
            power_path = Path(user_path)
        else:
            base_dir = os.path.dirname(__file__)
            p1d_dir = os.path.join(base_dir, '..', 'P_G')
            power_path = Path(os.path.join(p1d_dir, f'P_G-{safe_z}.txt'))

        if not power_path.exists():
            raise FileNotFoundError(f"Could not find file: {power_path}")

        data = np.loadtxt(power_path)
        k_array = data[:, 0]
        power_array = data[:, 1]

        return k_array, power_array

    except Exception as e:
        raise RuntimeError(f"Failed to process power file for z={safe_z}:\n{e}")


def parse_fitting_params(input_str=None, default=(0.67377, 5.31008, 2.16175, 1.50381)):
    """
    Parses lognormal parameters from a comma-separated string or a file.

    Args:
        input_str (str, optional): Input string (comma list or path to .txt file).
        default (tuple): Default values (tau0, tau1, nu, sigma2).

    Returns:
        tuple: Parsed (tau0, tau1, nu, sigma2)
    """
    if input_str is None:
        return default

    try:
        # Check if it's a file
        path = Path(input_str)
        if path.exists():
            with open(path) as f:
                contents = f.read().replace(',', ' ').split()
                values = list(map(float, contents))
        else:
            # Assume comma-separated string
            values = list(map(float, input_str.split(',')))

        if len(values) != 4:
            raise ValueError("Exactly 4 parameters required (tau0, tau1, nu, sigma2)")

        return tuple(values)

    except Exception as e:
        raise ValueError(f"Failed to parse lognormal parameters: {e}")
        

def lambda_c(z, lambda0=lambda_0):
    """
    Computes the central rest-frame wavelength at a given redshift.

    Args:
        z (float or np.ndarray): Redshift value(s).
        lambda0 (float, optional): Rest-frame reference wavelength (default: lambda_0).

    Returns:
        float or np.ndarray: Redshifted central wavelength(s).
    """
    lambda_c = (1 + int(z)) * lambda0
    return lambda_c


def generate_wavelength_grid(velocity_grid, z, lambda_min=lambda_min, lambda_max=lambda_max, lambda0=lambda_0):
    """
    Converts a velocity grid to a wavelength grid at a given redshift.

    Args:
        velocity_grid (np.ndarray): Grid of velocities (km/s).
        z (float): Target redshift.
        lambda_min (float, optional): Minimum observed-frame wavelength (default: 3600 Å).
        lambda_max (float, optional): Maximum observed-frame wavelength (default: 9800 Å).
        lambda0 (float, optional): Rest-frame reference wavelength (default: lambda_0).

    Returns:
        np.ndarray: Corresponding wavelength grid (in Å).
    """
    v_min = (lambda_min / lambda_0 - 1) * c
    v_max = (lambda_max / lambda_0 - 1) * c
    wavelength_field = lambda_c(z) * np.exp(velocity_grid / c)
    return wavelength_field


def generate_gaussian_random_field(size=size, seed=None):
    """
    Generates a 1D Gaussian random field.

    Args:
        size (int or tuple, optional): Shape of the output array (default: 2**20).
        seed (int, optional): Seed for RNG. If None, uses default random generator.

    Returns:
        np.ndarray: Gaussian random field.
    """
    rng = np.random.default_rng(seed)
    gaussian_random_field = rng.normal(size=size)
    return gaussian_random_field


def delta_transform_1d(file_k_array, file_power_array, 
                       gaussian_random_field_k, dv):
    """
    Transforms a Gaussian white noise field in k-space to a correlated
    Gaussian field in velocity-space, using an imported power spectrum.

    Parameters:
    - file_k_array: array of k values from the power spectrum file [s/km]
    - file_power_array: array of P(k) values [arbitrary units]
    - gaussian_random_field_k: white noise field in rfft space
    - dv: spacing of the velocity grid [km/s]

    Returns:
    - delta_b_tilde: scaled k-space field
    - delta_b: real-space Gaussian field
    - P_k: interpolated and used power spectrum
    """
    N_rfft = gaussian_random_field_k.shape[0]
    N = 2 * (N_rfft - 1)  # size of real-space grid
    k = np.fft.rfftfreq(N, d=dv) * 2 * np.pi  # k in units of 1/km

    power_interp = interp1d(file_k_array, file_power_array, 
                            kind='linear', bounds_error=False, 
                            fill_value=0.0)
    P_k = power_interp(k)
    
    # Sanitize P_k to ensure it's all finite and non-negative
    P_k = np.where((P_k > 0) & np.isfinite(P_k), P_k, 0.0)
    delta_b_tilde = gaussian_random_field_k * np.sqrt(P_k / dv)

    # Inverse rFFT to real-space Gaussian field
    delta_b = np.fft.irfft(delta_b_tilde, n=N)

    return delta_b_tilde, delta_b, P_k


def a2_z(zp, nu, z0=PD13_PIVOT_Z): # nu=2.16175
    return np.power((1. + zp) / (1.+z0), -nu)


def a_z(zp, nu, z0=PD13_PIVOT_Z): # nu=2.16175
    return np.sqrt(np.power((1. + zp) / (1.+z0), -nu))


def lognormal_transform(delta_z, sigma2_z):
    n_z = np.exp( (2 * (delta_z) - (sigma2_z)))
    return(n_z)


def t_of_z(zp, tau0, tau1, z0=PD13_PIVOT_Z): # tau0=673.77e-3, tau1=5.31008
    return tau0 * np.power((1. + zp) / (1.+z0), tau1)


def x_of_z(t_z, n_z):
    return t_z * n_z
    

def x_z(z, sigma2, tau0, tau1, nu, z0=PD13_PIVOT_Z):
    return t_of_z(z, tau0, tau1, z0) * np.exp(-a2_z(z, nu, z0) * sigma2)


def f_of_z(x_z):
    return np.exp(-x_z)


def turner24_mf(z):
    """
    Computes the mean flux based on the Turner+2024 model.

    Args:
        z (float or np.ndarray): The target redshift value.

    Returns:
        float or np.ndarray: The mean flux value(s) corresponding to 
                the input redshift(s).
    """
    tau_0 = -2.46e-3
    gamma = 3.62
    return np.exp(tau_0 * (1 + z)**gamma)   


def prefactor(variance):
    prefactor = 1 / (np.sqrt(variance) * np.sqrt(2 * np.pi))
    return(prefactor)


def mean_F(z, variance, tau0, tau1, nu, z0=PD13_PIVOT_Z):
    integrand = lambda x: np.exp((-(x**2) / (2 * variance)) - ((x_z(z, variance, tau0, tau1, nu)) * np.exp(2 * (a_z(z, nu)) * x)))
    integral = integrate.quad(integrand, -np.inf, np.inf)[0]
    value = prefactor(variance) * integral
    return(value)


def export_transmission(z_safe, v_array, f_array):
    """
    Exports velocity and flux arrays to a transmission file with a unique ID.

    Args:
        z_safe (str): Redshift label with dots replaced by dashes.
        v_array (np.ndarray): Velocity array.
        f_array (np.ndarray): Flux (transmission) array.
    """
    # Build export path
    base_dir = os.path.dirname(__file__)
    trans_dir = os.path.join(base_dir, '..', 'transmission_files', z_safe)
    os.makedirs(trans_dir, exist_ok=True)

    # Generate unique ID (timestamp + random digits)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_suffix = f"{random.randint(0, 9999):04d}"
    unique_id = f"{timestamp}_{rand_suffix}"

    # File name
    filename = f"transmission_{z_safe}_{unique_id}.txt"
    filepath = os.path.join(trans_dir, filename)

    # Export
    export_data = np.column_stack((v_array, f_array))
    np.savetxt(filepath, export_data, fmt="%.6e", delimiter="\t",
               header="Velocity [km/s]\tFlux")

    print(f"Saved transmission file: {filepath}")








#######################################


def plot_gaussian_field(z, field, space='v', sliced='y'):
    """
    Plots a 1D Gaussian random field in velocity or Fourier (k) space.

    Args:
        z (float or str): Redshift label for filename.
        field (np.ndarray): The field to plot (velocity or k-space).
        space (str, optional): Plotting mode - 'v' for velocity space or 'k' for k-space (default: 'v').
        sliced (str, optional): Whether to slice the data ('y' or 'n'). Default is 'y'.

    Saves:
        A PNG image of the plotted field named `{z}_Gaussian_Field_{space}.png`.
    """
    space = space.lower()
    if space == 'v':
        filename = f'{z}_Gaussian_Field_v.png'
        title = 'Gaussian Random Field, Velocity Space'
        xlabel = 'Points'
        ylabel = 'Velocity [km/s]'
        data = field
    elif space == 'k':
        filename = f'{z}_Gaussian_Field_k.png'
        title = 'RFFT(1D field), k-space (Real part)'
        xlabel = 'k-modes'
        ylabel = 'Amplitude'
        data = np.real(field)
    else:
        raise ValueError("Invalid space argument. Use 'v' or 'k'.")

    if sliced.lower() == 'y':
        data = data[min_slice:max_slice]

    print(f'Saving {filename}')
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)


def plot_gaussian_power(z, kmodes, field):
    """
    Plots the underlying model power spectrum. 

    Args:
        z (float or str): Redshift label for filename.
        kmodes (np.ndarray): The kmodes corresponding to the field.
        field (np.ndarray): The field to plot (in k-space).

    Saves:
        A PNG image of the plotted field named `{z}_gaussian_power.png`.
    """
    
    filename = f'{z}__gaussian_power.png'
    print(f'Saving {filename}')
    plt.figure()
    plt.loglog(kmodes, field, label=f'z = {z}', color='tab:orange',  ls='--')
    plt.axvspan(0.05, 1.0, alpha=0.2, color='grey')
    xlabel = r'kmodes $[km/s]^{-1}$'
    plt.ylabel( r"$P_G(k)$")
    plt.savefig(filename)   


def plot_delta_field(z, kmodes, velocity_grid, field, space='v', sliced='y'):
    """
    Plots the delta field in velocity, kmode, or redshift space.

    Args:
        z (float or str): Redshift label for filename.
        kmodes (np.ndarray): x-axis values for k-space.
        velocity_grid (np.ndarray): x-axis values for velocity space (not currently used but could be).
        field (np.ndarray): The field to plot.
        space (str, optional): Plotting mode - 'k' for kmodes, 'v' for velocity space, or 'z' for redshifted velocity space (default: 'v').
        sliced (str, optional): Whether to slice the data ('y' or 'n'). Default is 'y'.
        min_slice (int, optional): Start index for slicing. Default is 0.
        max_slice (int, optional): End index for slicing. Default is None (to end of array).

    Saves:
        A PNG image of the plotted field named `{z}_delta_field_{space}.png`.
    """
    space = space.lower()
    if space == 'k':
        filename = f'{z}_delta_field_k.png'
        xlabel = r'$k$-modes [$\mathrm{{(km/s)}}^{-1}$]'
        ylabel = r"$\tilde\delta_b(k)$"
        data = np.real(field)
        x_ax = kmodes
    elif space == 'v':
        filename = f'{z}_delta_field_v.png'
        xlabel = 'Velocity [km/s]'
        ylabel = r"$\delta_b(v)$"
        data = field
        x_ax = None  # let matplotlib default to index positions
    elif space == 'z':
        filename = f'{z}_delta_field_z.png'
        xlabel = 'Velocity [km/s]'
        ylabel = rf'$\delta_b(z = {z})$'
        data = field
        x_ax = None
    else:
        raise ValueError("Invalid space argument. Use 'k', 'v', or 'z'.")

    if sliced.lower() == 'y':
        data = data[min_slice:max_slice]
        if x_ax is not None:
            x_ax = x_ax[min_slice:max_slice]

    print(f'Saving {filename}')
    plt.figure()
    if x_ax is not None:
        plt.plot(x_ax, data)
    else:
        plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

    
def plot_nz(z, field, sliced='y'):
    """
    Plots the lognormally transformed field. 

    Args:
        z (float or str): Redshift label for filename.
        field (np.ndarray): The field to plot (in velocity space).
        sliced (str, optional): Whether to slice the data ('y' or 'n'). Default is 'y'.

    Saves:
        A PNG image of the plotted field named `{z}_nz_field.png`.
    """
    if sliced.lower() == 'y':
        field = field[min_slice:max_slice]
        
    filename = f'{z}_nz_field.png'
    print(f'Saving {filename}')
    plt.figure()
    plt.plot(field)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel( rf"n(z = {z})")
    plt.savefig(filename)   


def plot_optical_depth(z, field, sliced='y'):
    """
    Plots the optical depth in velocity space. 

    Args:
        z (float or str): Redshift label for filename.
        field (np.ndarray): The field to plot (optical depth).
        sliced (str, optional): Whether to slice the data ('y' or 'n'). Default is 'y'.

    Saves:
        A PNG image of the plotted field named `{z}_optical_depth.png`.
    """
    if sliced.lower() == 'y':
        field = field[min_slice:max_slice]
        
    filename = f'{z}_optical_depth.png'
    print(f'Saving {filename}')
    plt.figure()
    plt.plot(field)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel(rf'$\tau(z = {z})$')
    plt.savefig(filename)   


def plot_transmission(z, safe_z, velocity_grid, field, variance, tau0, tau1, nu, space='v', sliced='y'):
    """
    Plots the transmission field in velocity or wavelength space.

    Args:
        z (float or str): Redshift label for filename.
        field (np.ndarray): The field to plot (transmitted fluc).
        v_or_w (str, optional): Plotting mode - 'v' for velocity space or 'w' for wavelength space (default: 'v').
        sliced (str, optional): Whether to slice the data ('y' or 'n'). Default is 'y'.

    Saves:
        A PNG image of the plotted field named `{z}_transmission_field_{space}.png`.
    """
    space = space.lower()
    if space == 'v':
        filename = f'{safe_z}_transmission_field_v.png'
        xlabel = 'Velocity [km/s]'
        ylabel = f"F(z = {safe_z})"
        data = field
        x_ax = None
    elif space == 'w':
        wavelength_grid = generate_wavelength_grid(velocity_grid, z)
        filename = f'{safe_z}_transmission_field_w.png'
        xlabel = 'Wavelength (Å)'
        ylabel = f"F(z = {safe_z})"
        data = field
        x_ax = wavelength_grid
    else:
        raise ValueError("Invalid space argument. Use 'v' or 'z'.")

    mean_flux = mean_F(z, variance, tau0, tau1, nu)

    if sliced.lower() == 'y':
        data = data[min_slice:max_slice]
        if x_ax is not None:
            x_ax = x_ax[min_slice:max_slice] 
    print(f'Saving {filename}')
    plt.figure()
    if x_ax is not None:
        plt.plot(x_ax, data)
    else:
        plt.plot(data)
    plt.axhline(y=0, color='black', ls='--')
    plt.axhline(y=1, color='black', ls='--')
    plt.axhline(y=mean_flux, color='tab:red', ls='--', 
               label=rf'$\overline{{F}}(z) = {mean_flux:.2f}$')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left')
    plt.savefig(filename)


#######################################


def main():
    parser = argparse.ArgumentParser(description="Generate random, 1D lognormal mock spectra for Lyman-alpha forest analyis, given redshift (required) and a gaussian correlation function (optional).") # include some correlation functions as .txt with the code / github, so it can pull from those? otherwise take user input? 
        
    # parser.add_argument('--cf_files', type=str, required=False,
    #                     help='Optional comma-separated list of correlation function file paths')    
    parser.add_argument('--power_files', type=str, required=False,
                        help='Optional comma-separated list of gaussian power file paths')   
    parser.add_argument('--z_target', type=str, required=True,
                        help='Path to input file (.txt) containing target redshift values, OR a single float input')
    parser.add_argument('--N_mocks', type=int, default=1,
                        help='Number of mocks per redshift to generate (optional, default = 1)')
    
    parser.add_argument('--fit_params', type=str, required=False,
                        help='Optional comma-separated values (tau0,tau1,nu,sigma2) or a file with these values. If omitted, uses defaults.')
      
    parser.add_argument('--plot_gaussian_field', action='store_true',
                        help='Generate and save a figure of initial Gaussian random grid (default: False)')
    parser.add_argument('--plot_gaussian_power', action='store_true',
                        help='Generate and save a figure of the underlying gaussian power k-space (default: False)') 
    parser.add_argument('--plot_delta_k', action='store_true',
                        help='Generate and save a figure of the delta field in k-space (default: False)')        
    parser.add_argument('--plot_delta_v', action='store_true',
                        help='Generate and save a figure of the delta field in velocity space (default: False)')
    parser.add_argument('--plot_delta_z', action='store_true',
                        help='Generate and save a figure of the redshifted delta field in velocity space (default: False)')
    parser.add_argument('--plot_nz', action='store_true',
                        help='Generate and save a figure of the lognormally transformed field (default: False)')
    parser.add_argument('--plot_optical_depth', action='store_true',
                        help='Generate and save a figure of the optical depth (default: False)')
    parser.add_argument('--plot_transmission_v', action='store_true',
                        help='Generate and save a figure of the transmission field in velocity space (default: False)')
    parser.add_argument('--plot_transmission_w', action='store_true',
                        help='Generate and save a figure of the transmission field in wavelength space (default: False)')
    parser.add_argument('--plot_mean_flux', action='store_true',
                        help='Generate and save a figure of the measured mean flux (default: False)')
    parser.add_argument('--plot_power', action='store_true',
                        help='Generate and save a figure of the measured 1D power (default: False)')
        
    args = parser.parse_args()

    z0 = PD13_PIVOT_Z

    ### Process Input Data ###
    # z_target (required)
    try:
        z_target = parse_redshift_target(args.z_target)
    except Exception as e:
        print(f"Error reading redshift file: {e}")
        return

    # # cf (required)
    # cf_files = args.cf_files.split(',') if args.cf_files else [None] * len(z_target)
    # if len(cf_files) != len(z_target):
    #     if args.cf_files:
    #         raise ValueError("Number of CF files must match number of redshifts.")
    # # Otherwise, user didn't supply files, which is OK
    
    # gaussian power (required)
    power_files = args.power_files.split(',') if args.power_files else [None] * len(z_target)
    if len(power_files) != len(z_target):
        if args.power_files:
            raise ValueError("Number of power files must match number of redshifts.")
    # Otherwise, user didn't supply files, which is OK

    # fitting params (optional)
    fitting_params = parse_fitting_params(args.fit_params)
    tau0, tau1, nu, sigma2 = fitting_params

    
    # generate mocks for each target redshift
    for z, power_file in zip(z_target, power_files):
        print(f'\nProcessing z = {z}')
        safe_z = str(z).replace('.', '-')
        k_array, power_array = process_power_file(safe_z, power_file)

        for i in range(args.N_mocks):
            # z = redshift
            # i = number of mocks to generate
       
            gaussian_random_field_v = generate_gaussian_random_field()
            gaussian_random_field_k = np.fft.rfft(gaussian_random_field_v) 

            kmodes = (np.fft.rfftfreq(n=size, d=dv) * 2 * np.pi) + 1e-12 

            delta_b_tilde, delta_b_v, P_k = delta_transform_1d(k_array, power_array, gaussian_random_field_k, dv)
                
            variance_1d = delta_b_v.var() # sigma^2
            delta_b_z = delta_b_v * a_z(z, nu)
            redshifted_variance_1d = variance_1d * a2_z(z, nu)
            
            n_z = lognormal_transform(delta_b_z, redshifted_variance_1d)
                
            t_z = t_of_z(z, tau0, tau1)

            x_z = x_of_z(t_z, n_z)

            f_z = f_of_z(x_z)

            export_transmission(safe_z, velocity_grid, f_z)

    

            
        
            ### SAVE PLOTS ##
            print("\n###  Saving Figures  ###\n")
            if args.plot_gaussian_field:
                plot_gaussian_field(safe_z, gaussian_random_field_v, space='v', sliced='y')

                
            if args.plot_gaussian_power:
                plot_gaussian_power(safe_z, k_array, power_array)

                
            if args.plot_delta_k:
                plot_delta_field(safe_z, kmodes, velocity_grid, delta_b_tilde, space='k', sliced='n')
            if args.plot_delta_v:
                plot_delta_field(safe_z, kmodes, velocity_grid, delta_b_v, space='v', sliced='y')
            if args.plot_delta_z:
                plot_delta_field(safe_z, kmodes, velocity_grid, delta_b_z, space='z', sliced='y')
            if args.plot_nz:
                plot_nz(safe_z, n_z, sliced='y')
            if args.plot_optical_depth:
                plot_optical_depth(safe_z, x_z, sliced='y')      
            if args.plot_transmission_v:
                plot_transmission(z, safe_z, velocity_grid, 
                                  f_z, variance_1d, tau0, tau1, nu, space='v', sliced='y')
            if args.plot_transmission_w:
                plot_transmission(z, safe_z, velocity_grid,
                                  f_z, variance_1d, tau0, tau1, nu, space='w', sliced='y')




                
            # if args.plot_mean_flux:
            #     plot_mean_flux()
        
            # if args.plot_power:
            #     plot_power()

if __name__ == "__main__":
    main()

