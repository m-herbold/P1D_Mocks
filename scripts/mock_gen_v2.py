#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import pandas as pd
import argparse
import random
import time
import os

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from datetime import datetime
from scipy import integrate
from pathlib import Path


#######################################


plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['savefig.bbox'] = 'tight'

# Colorblind friendly colors
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00', 
                  '#000000', '#FFFFFF']


#######################################


DESI_EDR_PARAMETERS = (
    7.63089e-02, -2.52054e+00, -1.27968e-01,
    3.67469e+00, 2.85951e-01, 7.33473e+02)

Naim_etal_2020_param = (
    0.066, -2.685, -0.22,
    3.59, -0.18, 0.53)

PD13_PIVOT_K = 0.009  # note: k_0
PD13_PIVOT_Z = 3.0    # note: z0 = 3

c = 299792.458     # speed of light in km/s
lambda_0 = 1216    # rest wavelength in Angstroms (for Lyα)
lambda_min = 3600  # minimum wavelength in Angstroms
lambda_max = 9800  # maximum wavelength in Angstroms

size = 2**22
dv_fid = 1.0
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

# fitting parameters from mean flux
tau0_fid = 673.77e-3
tau1_fid = 5.31008
nu_fid = 2.16175
sigma2_fid = 1.50381


#######################################


def read_data(filename, expect_two_columns=False):
    """
    Reads data from a .txt file and returns as array(s).

    Args:
        filename (str): Path to input file. Supported format: .txt.
        expect_two_columns (bool): If True, expects exactly two columns in the file.

    Returns:
        np.ndarray: 1D array if expect_two_columns=False.
        tuple[np.ndarray, np.ndarray]: Two 1D arrays if expect_two_columns=True.

    Raises:
        ValueError: If an unsupported file format is provided.
        ValueError: If `expect_two_columns` is True but the file does not
                    contain exactly two columns.
        ValueError: If `expect_two_columns` is False but the data is not 1D.
    """
    if filename.endswith('.txt'):
        data = np.loadtxt(filename)
    else:
        raise ValueError("Unsupported file format. Use .txt")

    # If the data is supposed to be two columns
    # (e.g. power spectrum or flux), check it
    if expect_two_columns:
        if data.ndim == 1:
            raise ValueError(
                f"Expected two columns in {filename}, but found only one.")
        if data.shape[1] != 2:
            raise ValueError(
                f"Expected two columns in {filename}, but found {data.shape[1]}.")
        # Return the two columns as separate arrays
        return data[:, 0], data[:, 1]

    # Otherwise, return as a 1D array (for redshift bins, for example)
    if data.ndim > 1:
        raise ValueError(
            f"Expected a 1D array from {filename}, but got shape {data.shape}.")
    return data.flatten()  # Return a flattened 1D array


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
            "Invalid input for --redshift_bins. Provide a valid .txt/.fits file,"
            " a single float value, or a comma-separated list of floats.")


def process_power_file(safe_z, user_path=None):
    """
    Loads Gaussian power spectrum file for given redshift.

    Args:
        safe_z (str): Redshift label with '.' replaced by '-'.
        user_path (str or Path, optional): Custom path for power file.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of k and power P(k).

    Raises:
        RuntimeError: If file is not found or cannot be loaded.
    """
    try:
        if user_path:
            power_path = Path(user_path)
        else:
            base_dir = os.path.dirname(__file__)
            p1d_dir = os.path.join(base_dir, '..', 'P_G')
            # power_path = Path(os.path.join(p1d_dir, f'P_G-{safe_z}.txt'))
            power_path = Path(os.path.join(p1d_dir, f'P_G-{safe_z}.txt'))

        if not power_path.exists():
            raise FileNotFoundError(f"Could not find file: {power_path}")

        data = np.loadtxt(power_path)
        k_array = data[:, 0]
        power_array = data[:, 1]

        return k_array, power_array

    except Exception as e:
        raise RuntimeError(
            f"Failed to process power file for z={safe_z}:\n{e}")


def parse_fitting_params(input_str=None, default=(tau0_fid, tau1_fid,
                                                  nu_fid, sigma2_fid)):
    """
    Parses lognormal fitting parameters from string or file, or returns defaults.

    Args:
        input_str (str, optional): Comma-separated string or filename with 4 parameters.
        default (tuple): Default (tau0, tau1, nu, sigma2) parameters.

    Returns:
        tuple: Parsed parameters (tau0, tau1, nu, sigma2).

    Raises:
        ValueError: If parsing fails or incorrect number of parameters.
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
            raise ValueError(
                "Exactly 4 parameters required (tau0, tau1, nu, sigma2)")

        return tuple(values)

    except Exception as e:
        raise ValueError(f"Failed to parse lognormal parameters: {e}")


def lambda_c(z, lambda0=lambda_0):
    """
    Computes redshifted central wavelength.

    Args:
        z (float or np.ndarray): Redshift(s).
        lambda0 (float): Rest-frame wavelength (default: 1216 Å).

    Returns:
        float or np.ndarray: Redshifted wavelength(s).
    """
    lambda_c = (1 + int(z)) * lambda0
    return lambda_c


def generate_wavelength_grid(velocity_grid, z, lambda_min=lambda_min,
                             lambda_max=lambda_max, lambda0=lambda_0):
    """
    Converts velocity grid to observed-frame wavelength grid at redshift z.

    Args:
        velocity_grid (np.ndarray): Velocity grid (km/s).
        z (float): Target redshift.
        lambda_min (float): Minimum wavelength (Å).
        lambda_max (float): Maximum wavelength (Å).
        lambda0 (float): Rest-frame reference wavelength (Å).

    Returns:
        np.ndarray: Wavelength grid (Å).
    """
    wavelength_field = lambda_c(z) * np.exp(velocity_grid / c)
    return wavelength_field


def generate_gaussian_random_field(size=size, seed=None):
    """
    Generates a 1D Gaussian random field.

    Args:
        size (int or tuple, optional): Shape of the output array.
        seed (int, optional): Seed for RNG. If None, uses default
                            random generator.

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

    Args:
        file_k_array (np.ndarray): Baseline k values [1/km].
        file_power_array (np.ndarray): Baseline P(k).
        gaussian_random_field_k (np.ndarray): White noise field in Fourier space.
        dv (float): Velocity spacing [km/s].

    Returns:
        tuple:
            delta_b_tilde (np.ndarray): Scaled k-space field.
            delta_b (np.ndarray): Real-space correlated Gaussian field.
            P_k (np.ndarray): Interpolated power spectrum used.
    """
    N_rfft = gaussian_random_field_k.shape[0]
    N = 2 * (N_rfft - 1)  # real-space size
    k = np.fft.rfftfreq(N, d=dv) * 2 * np.pi  # k [1/km]

    # Interpolate with smooth spline and constant extrapolation
    power_interp = InterpolatedUnivariateSpline(file_k_array,
                                                file_power_array,
                                                k=1, ext=1)
    P_k = power_interp(k)

    # Ensure non-negative and finite
    P_k = np.where((P_k > 0) & np.isfinite(P_k), P_k, 0.0)

    # Scale the white noise in k-space
    delta_b_tilde = gaussian_random_field_k * np.sqrt(P_k / dv)
    # delta_b_tilde = gaussian_random_field_k * np.sqrt(P_k)

    # Inverse FFT to get real-space correlated Gaussian field
    delta_b = np.fft.irfft(delta_b_tilde, n=N) / dv
    # delta_b = np.fft.irfft(delta_b_tilde, n=N) / (N * dv)

    return delta_b_tilde, delta_b, P_k


def a2_z(zp, nu=nu_fid, z0=PD13_PIVOT_Z):
    """
    Computes the redshift-dependent scaling factor a^2(z).

    This function calculates a^2(z) using a power-law scaling
    relative to a pivot redshift.

    Args:
        zp (float or np.ndarray): Redshift(s) at which to evaluate scale factor.
        nu (float, optional): Exponent controlling redshift evolution
            (default: 2.16175).
        z0 (float, optional): Pivot redshift for normalization
            (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: The computed a^2(z) scaling factor.
    """
    return np.power((1. + zp) / (1.+z0), -nu)


def a_z(zp, nu=nu_fid, z0=PD13_PIVOT_Z):
    """
    Computes the redshift-dependent scaling factor a(z).

    This function calculates a(z) as the square root of a^2(z), following a
    power-law scaling relative to a pivot redshift.

    Args:
        zp (float or np.ndarray): Redshift(s) at which to evaluate scale factor.
        nu (float, optional): Exponent controlling redshift evolution
            (default: 2.16175).
        z0 (float, optional): Pivot redshift for normalization
            (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: The computed a(z) scaling factor.
    """
    return np.sqrt(np.power((1. + zp) / (1.+z0), -nu))


def lognormal_transform(delta_z, sigma2_z):
    """
    Applies a lognormal transform to approximate the HI column density as a
    function of redshift.

    This function computes a redshift-dependent factor based on a
    lognormal distribution,
    which is commonly used to approximate the HI column density in
    cosmological models.

    Args:
        delta_z (array): Redshifted delta field.
        sigma2_z (float): Redshifted variance of the Gaussian field.

    Returns:
        float or np.ndarray: Lognormal transform, approximating HI column density.
    """
    n_z = np.exp((2 * (delta_z) - (sigma2_z)))
    return (n_z)


def t_of_z(zp, tau0=tau0_fid, tau1=tau1_fid, z0=PD13_PIVOT_Z):
    """
    Computes the optical depth as a function of redshift.

    This function calculates the optical depth, tau(z), based on a power-law
    scaling relative to a pivot redshift.

    Args:
        zp (float or np.ndarray): Redshift(s) at which to evaluate optical depth.
        tau0 (float, optional): Normalization factor for optical depth (
                                default: 673.77e-3).
        tau1 (float, optional): Exponent controlling redshift evolution
                of optical depth (default: 5.31008).
        z0 (float, optional): Pivot redshift for normalization
                (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: Computed optical depth(s) for given redshift(s).
    """
    return tau0 * np.power((1. + zp) / (1.+z0), tau1)


def x_of_z(t_z, n_z):
    """
    Modifies optical depth for flux calculations by incorporating z-dependence.

    Args:
        t_of_z (float):
        n_z (float or np.ndarray): Lognormal transform, approximating HI
                                    column density.

    Returns:
        float or np.ndarray: The computed x_of_z(z) factor for calculating flux.
    """
    return t_z * n_z


def f_of_z(x_z):
    """
    Calculates the z-dependent flux.

    Args:
        x_of_z (float or np.ndarray): The computed x_of_z(z) factor for
        calculating the flux.

    Returns:
        float or np.ndarray: The computed flux at the given redshift(s).
    """
    return np.exp(-x_z)


# used for GHQ mean flux
def prefactor(variance):
    """
    Compute the prefactor for Gaussian-Hermite quadrature integration.

    Args:
        variance (float): Variance of the Gaussian field.

    Returns:
        float: Normalization factor for the quadrature integral.
    """
    prefactor = 1 / (np.sqrt(variance) * np.sqrt(2 * np.pi))
    return (prefactor)


# used for GHQ mean flux
def xz(z, sigma2=sigma2_fid, tau0=tau0_fid, tau1=tau1_fid,
       nu=nu_fid, z0=PD13_PIVOT_Z):
    """
    Computes the redshift-dependent optical depth factor for GHQ integration.

    Args:
        z (float or np.ndarray): Redshift(s).
        sigma2 (float): Gaussian field variance.
        tau0 (float): Optical depth normalization.
        tau1 (float): Optical depth redshift evolution exponent.
        nu (float): Redshift exponent for the lognormal transform.
        z0 (float, optional): Pivot redshift (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: x(z) used in mean flux calculations.
    """
    return t_of_z(z, tau0, tau1, z0) * np.exp(-a2_z(z, nu, z0) * sigma2)


# used for GHQ mean flux -> pass field variance, not fitting param
def mean_flux(z, variance, z0=PD13_PIVOT_Z):
    """
    Computes the mean transmitted flux ⟨F⟩ using Gaussian-Hermite quadrature.

    Args:
        z (float): Redshift.
        variance (float): Variance of the Gaussian field.
        z0 (float, optional): Pivot redshift (default: PD13_PIVOT_Z).

    Returns:
        float: Mean transmitted flux ⟨F⟩.
    """
    def integrand(x): return np.exp((-(x**2) / (2 * variance)) -
                                    ((xz(z)) * np.exp(2 * (a_z(z)) * x)))
    integral = integrate.quad(integrand, -np.inf, np.inf)[0]
    value = prefactor(variance) * integral
    return (value)


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


def export_transmission(z_safe, v_array, f_array):
    """
    Exports velocity and flux arrays to a uniquely named transmission file.

    Args:
        z_safe (str): Redshift string with periods replaced by dashes.
        v_array (np.ndarray): Velocity array [km/s].
        f_array (np.ndarray): Transmitted flux array.

    Returns:
        str: Path to the directory where the file was saved.

    Raises:
        OSError: If the directory cannot be created or written to.
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

    return (trans_dir)


def delta_F(z, variance, input_flux, z0=PD13_PIVOT_Z):
    """
    Computes fractional flux fluctuations: δ_F = (F - ⟨F⟩) / ⟨F⟩.

    Args:
        z (float): Redshift.
        variance (float): Variance of the Gaussian field.
        input_flux (np.ndarray): Array of transmitted flux values.
        z0 (float, optional): Pivot redshift (default: PD13_PIVOT_Z).

    Returns:
        np.ndarray: δ_F fluctuations.
    """
    f_bar = mean_flux(z, variance, z0)

    flux = input_flux
    delta_f = (flux - f_bar) / (f_bar)
    return (delta_f)


def P_F(delta_f, dv):
    """
    Computes the 1D power spectrum P_F(k) of fractional flux fluctuations.

    Args:
        delta_f (np.ndarray): Fractional flux fluctuations δ_F.
        dv (float): Pixel velocity spacing [km/s].

    Returns:
        np.ndarray: Power spectrum P_F(k).
    """
    L = delta_f.size * dv
    delta_f_tilde = np.fft.rfft(delta_f) * dv
    P_F = np.abs(delta_f_tilde)**2 / L
    return P_F


def evaluatePD13Lorentz(X, A, n, alpha, B, beta, lmd):
    """
    Evaluates the PD13 Lorentzian model for the 1D power spectrum (P1D).

    This function computes the P1D based on the PD13 Lorentzian model
    for given wavenumbers, redshifts, and fitting parameters (A,n,alpha,B,beta).

    Args:
        X (tuple[np.ndarray, np.ndarray]): A tuple containing:
            - k (np.ndarray): Array of wavenumbers.
            - z (np.ndarray or None): Array of redshifts
                (or None for a single evaluation).
        A (float): Amplitude scaling factor.
        n (float): Power-law index for k dependence.
        alpha (float): Logarithmic correction to the power-law.
        B (float): Power-law index for redshift dependence.
        beta (float): Logarithmic correction for redshift evolution.
        lmd (float): Lorentzian damping factor.

    Returns:
        np.ndarray: Evaluated P1D values for the given k and z inputs.
    """
    k, z = X
    q0 = k / PD13_PIVOT_K + 1e-10
    result = (A * np.pi / PD13_PIVOT_K) * np.power(
        q0, 2. + n + alpha * np.log(q0)) / (1. + lmd * k**2)
    if z is not None:
        x0 = (1. + z) / (1. + PD13_PIVOT_Z)
        result *= np.power(q0, beta * np.log(x0)) * np.power(x0, B)
    return result


def fit_PD13Lorentz(delta_f, dv, z):
    """
    Fit the 1D flux power spectrum using the PD13 Lorentzian model.

    This function computes the 1D power spectrum from flux fluctuations,
    bins it over k-space, and fits the PD13 Lorentzian model to the binned
    power using non-linear least squares. The model includes both scale
    and redshift evolution terms, and is evaluated via `evaluatePD13Lorentz`.

    Parameters:
        delta_f (np.ndarray): Array of fractional flux fluctuations δ_F.
        dv (float): Velocity spacing between pixels [km/s].
        z (float): Redshift at which the power spectrum is evaluated.

    Returns:
        k_arr (np.ndarray): Binned k-values [s/km] used in the fit.
        P_k (np.ndarray): Binned power spectrum values corresponding to `k_arr`.
        A (float): Amplitude scaling factor of the PD13 model.
        n (float): Power-law index for the k dependence.
        alpha (float) Logarithmic correction to the power-law index.
        B (float): Power-law index for redshift evolution.
        beta (float): Logarithmic correction for redshift evolution.
        lmd (float): Lorentzian damping factor.

    Raises:
        RuntimeError: If `curve_fit` fails to converge.
    """
    power = P_F(delta_f, dv)
    N = len(delta_f)
    kmodes = np.fft.rfftfreq(n=N, d=dv) * 2 * np.pi
    w_k = (kmodes > 0) & (kmodes < 10e2)
    bins = 10000
    statistic, bin_edges, binnumber = binned_statistic(x=kmodes[w_k],
                                                       values=power[w_k],
                                                       statistic='mean',
                                                       bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # window = (bin_centers > 1e-5) & (bin_centers < 0.10)
    window = (bin_centers > 0) & (bin_centers < 1.0)
    # Remove invalid points
    valid = np.isfinite(statistic) & np.isfinite(bin_centers)
    bin_centers = bin_centers[valid]
    statistic = statistic[valid]

    # Initial guess
    p0 = (0.07, -2.5, -0.1, 3.5, 0.3, 700)

    # Now safe to call curve_fit
    popt_mock, pcov_mock = curve_fit(
        lambda k, A, n, alpha, B, beta, lmd: evaluatePD13Lorentz(
            (k, z), A, n, alpha, B, beta, lmd),
        bin_centers, statistic, p0=p0, maxfev=20000)

    return bin_centers[window], statistic[window], *popt_mock
    # return bin_centers, statistic, *popt_mock


def process_EDR_DATA(z_target):
    """
    Loads and processes EDR QMLE power spectrum data for a given redshift.

    Args:
        z_target (float): Target redshift to filter from the EDR data file.

    Returns:
        tuple:
            - kc (np.ndarray): Center of k-bins.
            - Pk (np.ndarray): Recovered power spectrum.
            - Pk_err (np.ndarray): Associated uncertainties.

    Raises:
        FileNotFoundError: If the data file is missing.
        ValueError: If data format is invalid.
    """
    current_path = Path(__file__).resolve().parent
    edr_data_path = current_path.parent / 'Examples' / \
        'figure8_qmle_desiedrp_results.txt'

    # Read the file, using '|' as a separator and stripping whitespace
    df = pd.read_csv(edr_data_path, sep='|', skiprows=1)
    # Drop extra empty columns from leading/trailing pipes
    df = df.drop(columns=df.columns[[0, -1]])
    df.columns = ['kc', 'z', 'kPpi', 'kepi']   # Rename columns
    df = df.apply(pd.to_numeric)               # Ensure all data is numeric

    subset = df[df['z'] == z_target].copy()

    # Recover P(k) and its uncertainty
    subset['Pk'] = np.pi * subset['kPpi'] / subset['kc']
    subset['Pk_err'] = np.pi * subset['kepi'] / subset['kc']

    return subset['kc'], subset['Pk'], subset['Pk_err']


def get_k_range_desi2025(z):
    """
    Return kmin and kmax (s/km) for DESI DR1 resolution at redshift z.

    Args:
        z (float): Redshift.

    Returns:
        tuple:
            - k_min (float): Minimum k [s/km].
            - k_max (float): Maximum k [s/km].
    """
    c = 299792.458        # km/s
    delta_lambda = 0.8    # Angstrom (DESI)
    lambda_lya = 1215.67  # Angstrom
    R_z = (c * delta_lambda) / ((1 + z) * lambda_lya)
    k_max = 0.5 * np.pi / R_z  # s/km
    k_min = 0.001              # s/km, fixed by continuum limit
    return k_min, k_max


def compute_rms_error(measurement, target, mask=None):
    """
    Computes the root-mean-square (RMS) fractional error between measurement and target.

    Args:
        measurement (np.ndarray): Measured values.
        target (np.ndarray): Target or true values.
        mask (np.ndarray, optional): Boolean mask to apply before computation.

    Returns:
        float: RMS fractional error.
    """
    if mask is not None:
        measurement = measurement[mask]
        target = target[mask]
    frac_diff = (measurement - target) / target
    return np.sqrt(np.mean(frac_diff**2))


def fit_and_plot_power(delta_f=None, z=None, dv=None, dv_array=None, safe_z=None,
                       N_mocks=None, z_target=None, k_arrays=None,
                       power_arrays=None, delta_f_array=None, all_z='n', plot='y'):
    """
    Fit the PD13 Lorentzian model to the 1D flux power spectrum and optionally plot it.

    Two operation modes:
    - **Single-redshift mode (`all_z='n'`)**: Fits the power spectrum at a single redshift
      using `delta_f` and `dv`, and optionally plots comparison against reference models.
    - **Multi-redshift mode (`all_z='y'`)**: Fits and plots the power spectra across all
      redshifts in `z_target`, but does not return fit results.

    Parameters
    ----------
        all_z : {'y', 'n'}, default 'n'
            Whether to operate in multi-redshift mode.
        plot : {'y', 'n'}, default 'y'
            Whether to generate plots.

      --- Used in single-redshift mode ('all_z'='n') ---
        delta_f : np.ndarray, optional
            Normalized flux fluctuations for a single mock spectrum.
        z : float, optional
            Redshift of the spectrum.
        dv : float, optional
            Velocity spacing [km/s].
        safe_z : str, optional
            Safe version of redshift string for filenames.
        N_mocks : int, optional
            Number of mocks used (for legend labels).

      --- Used in multi-redshift mode ('all_z'='y') ---
        dv_array : list of float, optional
            List of dv values for each redshift.
        delta_f_array : list of np.ndarray, optional
            List of flux fluctuation arrays.
        z_target : list or np.ndarray, optional
            Redshift values to process.
        k_arrays : list of np.ndarray, optional
            List of k values per redshift (currently unused).
        power_arrays : list of np.ndarray, optional
            Power spectra per redshift (currently unused).

    Returns
    -------
        bin_centers : np.ndarray
            k-bin centers used for the fit (only if `all_z='n'`).
        statistic : np.ndarray
            Measured power spectrum in each bin (only if `all_z='n'`).
        popt : tuple
            Best-fit parameters from PD13 model (only if `all_z='n'`).

    Saves
    -----
    Power_measured.png :
        Power spectrum plot for all redshifts (if `all_z='y'` and `plot='y'`).
    <safe_z>_power_fit.png :
        Fit vs. data plot for a single redshift (if `all_z='n'` and `plot='y'`).
    """

    if all_z == 'y' and plot == 'y':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                       sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       constrained_layout=True)
        cmap = plt.get_cmap('rainbow')
        norm = plt.Normalize(vmin=min(z_target), vmax=max(z_target))

        for i, z in enumerate(z_target):
            dv = dv_array[i]
            delta_f = delta_f_array[i]
            color = cmap(norm(z))

            # ===== Fit PD13 model to measured power =====
            bin_centers, stat, *popt = fit_PD13Lorentz(delta_f, dv, z)
            w_k = (bin_centers > 1e-5) & (bin_centers < 0.1)

            # Evaluate fits and models
            mock_fit = evaluatePD13Lorentz((bin_centers, z), *popt)
            desi_model = evaluatePD13Lorentz(
                (bin_centers, z), *DESI_EDR_PARAMETERS)

            percent_diff_mock_measure = 100 * (stat - desi_model) / desi_model
            percent_diff_mock_fit = 100 * (mock_fit - desi_model) / desi_model

            # ===== Top: Power spectrum fit comparison =====
            ax1.loglog(bin_centers[w_k], stat[w_k], color=color,
                       alpha=0.5, linewidth=3, label=f'z = {z}')
            ax1.loglog(bin_centers[w_k], desi_model[w_k],
                       color=color, ls='--')

            # ===== Bottom: % difference =====
            ax2.plot(bin_centers[w_k], percent_diff_mock_measure[w_k],
                     lw=1.0, color=color)

        # Final plot styling
        ax1.set_ylabel(r"$P_{\mathrm{1D}}(k)$")
        ax1.legend(ncol=3, fontsize='small', loc='lower left')
        ax1.grid(True, which='both', ls=':', alpha=0.6)
        ax1.yaxis.set_major_formatter(ScalarFormatter())
        ax1.ticklabel_format(style='plain', axis='y')

        ax2.axhline(0, color=CB_color_cycle[6], lw=1, ls='--')
        ax2.set_ylabel(r"% Difference")
        ax2.set_xlabel(r'k $[km/s]^{-1}$')
        ax2.grid(True, ls=':', alpha=0.6)

        print('Saving: Power_measured.png')
        plt.savefig("Power_measured.png")
        plt.close()

    else:
        # Standard single redshift mode
        if z_target is None:
            z_target = np.array([z])
        safe_z = str(z).replace('.', '-')
        measured_power = P_F(delta_f, dv)
        delta_f = delta_f

        bin_centers, stat, *popt = fit_PD13Lorentz(delta_f, dv, z)
        # generic extended k-mask
        # w_k = (bin_centers > 1e-5) & (bin_centers < 0.1)
        # generic condensed mask
        w_k = (bin_centers > 1e-4) & (bin_centers < 0.05)

        edr_k, edr_p, edr_err = process_EDR_DATA(z)

        if plot == 'y':
            # Evaluate fits and models
            mock_fit = evaluatePD13Lorentz((bin_centers, z), *popt)
            desi_model = evaluatePD13Lorentz(
                (bin_centers, z), *DESI_EDR_PARAMETERS)
            naim_2020_fit = evaluatePD13Lorentz(
                (bin_centers, z), *Naim_etal_2020_param)

            percent_diff_mock_measure = 100 * (stat - desi_model) / desi_model
            percent_diff_mock_fit = 100 * (mock_fit - desi_model) / desi_model
            percent_diff_naim_fit = 100 * \
                (naim_2020_fit - desi_model) / desi_model

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                           sharex=True,
                                           gridspec_kw={
                                               'height_ratios': [3, 1]},
                                           constrained_layout=True)

            ax1.loglog(bin_centers[w_k], stat[w_k], color='tab:orange',
                       alpha=0.5, label=f'This Work (N = {N_mocks})')
            ax1.loglog(bin_centers[w_k], desi_model[w_k], ls='--',
                       color='tab:blue', label=r'DESI EDR Fit (PD13)')
            ax1.errorbar(edr_k, edr_p, yerr=edr_err, fmt='o', label='DESI EDR',
                         color='tab:blue')
            ax1.set_ylabel(rf'$P(k)$   (z = {safe_z})')
            ax1.legend(loc='lower left')
            ax1.grid(True)
            # ax1.yaxis.set_major_formatter(ScalarFormatter())
            # ax1.ticklabel_format(style='plain', axis='y')
            ax1.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{int(y)}" if y >= 1 else f"{y:g}")
            )

            ax2.semilogx(bin_centers[w_k], percent_diff_mock_measure[w_k],
                         color='black', alpha=0.7)  # , label='Mock Measure')
            ax2.axhline(0, ls='--', color=CB_color_cycle[6])
            ax2.grid(True)
            # ax2.legend(loc='upper left')
            ax2.set_ylabel('% Difference')
            ax2.set_xlabel(r'k $[km/s]^{-1}$')

            # ax3.semilogx(bin_centers, percent_diff_naim_fit,
            #              color='black', label='Karacayli et al., 2020')
            # ax3.axhline(0, ls='--', color=CB_color_cycle[6])
            # ax3.set_xlabel(r'k $[km/s]^{-1}$')
            # ax3.grid(True)
            # fig.text(0.02, 0.25, "% Difference", va='center',
            #          rotation='vertical', fontsize=16)
            # ax3.set_ylabel("")

            plt.tight_layout()

            # define k-ranges based on Karacayli et al. 2020 / 2023 / 2025
            wk_2020 = (bin_centers > 0.0005) & (
                bin_centers < 0.112)  # DESI Lite
            dynamic_k_min, dynamic_k_max = get_k_range_desi2025(z)
            wk_2023 = (bin_centers > 0.000750) & (
                bin_centers < 0.035)  # EDR
            wk_2025 = (bin_centers > dynamic_k_min) & (
                bin_centers < dynamic_k_max)  # DR1
            wk_extended = (bin_centers > 1e-5) & (bin_centers <
                                                  0.1)  # for robustness
            wk_custom = (bin_centers > 1e-5) & (bin_centers <
                                                0.05)  # for display

            fig.text(
                0.01, -0.02,
                f"""
                    wk_2020: 0.0005 < k < 0.112  [s/km]

                    MH RMS:           {compute_rms_error(stat, desi_model, wk_2020):.4f}
                    MH Mean % diff:   {percent_diff_mock_measure[wk_2020].mean():.2f}%
                    MH Max  % diff:   {np.abs(percent_diff_mock_measure[wk_2020]).max():.2f}%

                    NK20 RMS:         {compute_rms_error(naim_2020_fit, desi_model, wk_2020):.4f}
                    NK20 Mean % diff: {percent_diff_naim_fit[wk_2020].mean():.2f}%
                    NK20 Max % diff:  {np.abs(percent_diff_naim_fit[wk_2020]).max():.2f}%


                    wk_2023: {0.000750} < k < {0.035}  [s/km]

                    MH RMS:           {compute_rms_error(stat, desi_model, wk_2023):.4f}
                    MH Mean % diff:   {percent_diff_mock_measure[wk_2023].mean():.2f}%
                    MH Max  % diff:   {np.abs(percent_diff_mock_measure[wk_2023]).max():.2f}%

                    NK20 RMS:         {compute_rms_error(naim_2020_fit, desi_model, wk_2023):.4f}
                    NK20 Mean % diff: {percent_diff_naim_fit[wk_2023].mean():.2f}%
                    NK20 Max % diff:  {np.abs(percent_diff_naim_fit[wk_2023]).max():.2f}%


                    wk_2025: {dynamic_k_min:.4f} < k < {dynamic_k_max:.4f}  [s/km]

                    MH RMS:           {compute_rms_error(stat, desi_model, wk_2025):.4f}
                    MH Mean % diff:   {percent_diff_mock_measure[wk_2025].mean():.2f}%
                    MH Max  % diff:   {np.abs(percent_diff_mock_measure[wk_2025]).max():.2f}%

                    NK20 RMS:         {compute_rms_error(naim_2020_fit, desi_model, wk_2025):.4f}
                    NK20 Mean % diff: {percent_diff_naim_fit[wk_2025].mean():.2f}%
                    NK20 Max % diff:  {np.abs(percent_diff_naim_fit[wk_2025]).max():.2f}%


                    wk_extended: {1e-5} < k < {0.10}  [s/km]

                    MH RMS:           {compute_rms_error(stat, desi_model, wk_extended):.4f}
                    MH Mean % diff:   {percent_diff_mock_measure[wk_extended].mean():.2f}%
                    MH Max  % diff:   {np.abs(percent_diff_mock_measure[wk_extended]).max():.2f}%

                    NK20 RMS:         {compute_rms_error(naim_2020_fit, desi_model, wk_extended):.4f}
                    NK20 Mean % diff: {percent_diff_naim_fit[wk_extended].mean():.2f}%
                    NK20 Max % diff:  {np.abs(percent_diff_naim_fit[wk_extended]).max():.2f}%

                    """,
                fontsize=9, ha='left', va='top'
            )

            print(f'Saving: {safe_z}_power_fit.png')
            plt.savefig(f'{safe_z}_power_fit.png')
            plt.close()

        return bin_centers, stat, popt


#######################################


def plot_gaussian_field(z, field, space='v', sliced='y'):
    """
    Plots a 1D Gaussian random field in velocity or Fourier (k) space.

    Parameters
    ----------
        z : float or str
            Redshift label used in the filename.
        field : np.ndarray
            The field to plot (velocity or k-space).
        space : {'v', 'k'}, optional
            Plotting mode: 'v' for velocity space, 'k' for k-space.
            Default is 'v'.
        sliced : {'y', 'n'}, optional
            Whether to slice the field using global `min_slice` and `max_slice`.
            Default is 'y'.

    Saves
    -----
        {z}_Gaussian_Field_{space}.png :
            A PNG image of the plotted field.
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
        {z}__gaussian_power.png: Log-log plot of the Gaussian power spectrum.
    """

    filename = f'{z}__gaussian_power.png'
    print(f'Saving {filename}')
    plt.figure()
    plt.loglog(kmodes, field, label=f'z = {z}', color='tab:orange',  ls='--')
    plt.axvspan(0.05, 1.0, alpha=0.2, color='grey')
    plt.xlabel(r'kmodes $[km/s]^{-1}$')
    plt.ylabel(r"$P_G(k)$")
    plt.savefig(filename)


def plot_delta_field(z, kmodes, velocity_grid, field, space='v', sliced='y'):
    """
    Plots the δ_b field in velocity space, k-space, or redshift-labeled space.

    Parameters
    ----------
        z : float or str
            Redshift label used for filename and labels.
        kmodes : np.ndarray
            x-axis for k-space plotting.
        velocity_grid : np.ndarray
            x-axis for velocity-space plotting.
        field : np.ndarray
            The field to plot.
        space : {'k', 'v', 'z'}, optional
            Coordinate space for plotting. Default is 'v'.
        sliced : {'y', 'n'}, optional
            Whether to slice the field and x-axis using `min_slice`
            and `max_slice` globals.

    Saves
    -----
        {z}_delta_field_{space}.png :
            A PNG image of the plotted δ_b field.
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
        sliced (str, optional): Whether to slice the data ('y' or 'n').
                                Default 'y'.

    Saves:
        {z}_nz_field.png: Plot of the lognormal field n(z).
    """
    if sliced.lower() == 'y':
        field = field[min_slice:max_slice]

    filename = f'{z}_nz_field.png'
    print(f'Saving {filename}')
    plt.figure()
    plt.plot(field)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel(rf"n(z = {z})")
    plt.savefig(filename)


def plot_optical_depth(z, field, sliced='y'):
    """
    Plots the optical depth in velocity space.

    Args:
        z (float or str): Redshift label for filename.
        field (np.ndarray): The field to plot (optical depth).
        sliced (str, optional): Whether to slice the data ('y' or 'n').
                                Default 'y'.

    Saves:
        {z}_optical_depth.png: Plot of the optical depth τ(z).
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


def plot_transmission(z, safe_z, velocity_grid, field, variance, tau0, tau1,
                      nu, space='v', sliced='y'):
    """
    Plots the transmission field in velocity or wavelength space.

    Parameters
    ----------
        z : float or str
            Redshift of the transmission field.
        safe_z : str
            File-safe version of z for saving.
        velocity_grid : np.ndarray
            Velocity grid for computing wavelength grid if space='w'.
        field : np.ndarray
            Transmission field to plot.
        variance : float
            Variance for computing mean flux.
        tau0 : float
            Optical depth parameter.
        tau1 : float
            Optical depth redshift scaling.
        nu : float
            Redshift evolution index.
        space : {'v', 'w'}, optional
            'v' for velocity space or 'w' for wavelength space.
            Default is 'v'.
        sliced : {'y', 'n'}, optional
            Whether to slice the field and x-axis using `min_slice`,
            `max_slice`.

    Saves
    -----
    {safe_z}_transmission_field_{space}.png :
        A PNG image of the transmission field with mean flux line.
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

    # mean_flux = mean_flux(z, variance, tau0, tau1, nu)
    mean_f = data.mean()

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
    plt.axhline(y=0, color=CB_color_cycle[6], ls='--')
    plt.axhline(y=1, color=CB_color_cycle[6], ls='--')
    plt.axhline(y=mean_f, color=CB_color_cycle[7], ls='--',
                label=rf'$\overline{{F}}(z) = {mean_flux:.2f}$')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left')
    plt.savefig(filename)


def plot_mean_flux(z_target, mean_flux_array, model_z, model_flux_array):
    """
    Plot measured mean flux against the Turner et al. (2024) model with percent residuals.

    Parameters:
    - z_target (array-like): Redshift values for the measured mean flux.
    - mean_flux_array (array-like): Measured mean flux values.
    - model_z (array-like): Redshift values for the model curve.
    - model_flux_array (array-like): Model mean flux values.
    """
    # Interpolate model to z_target for residuals
    interp_model_at_target = np.interp(z_target, model_z, model_flux_array)
    residuals = 100 * (mean_flux_array -
                       interp_model_at_target) / interp_model_at_target

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   constrained_layout=True)

    # Top panel: mean flux comparison
    ax1.plot(model_z, model_flux_array,
             label='Turner et al., 2024', 
             lw=3, alpha=0.5, color=CB_color_cycle[0])
    ax1.plot(z_target, mean_flux_array, label='Measured',
            linestyle='dotted', lw=2,
            marker='o', markersize=6,
            color=CB_color_cycle[1])
    
    ax1.set_ylabel(r'$\bar F(z)$')
    ax1.legend(loc='lower left')
    ax1.grid()

    # Bottom panel: percent difference
    ax2.axhline(0, color=CB_color_cycle[6], lw=1, ls='--')
    ax2.plot(z_target, residuals, marker='o', 
            color='black', alpha=0.7, lw=2, 
            markersize=6, ls='dotted')
    ax2.set_xlabel('z')
    ax2.set_ylabel('% Difference')
    ax2.grid()

    # Format the mean flux measurements as strings
    flux_text = "\n".join(
        [f"z = {z:.2f},   F(z) = {f:.3f}" for z,
         f in zip(z_target, mean_flux_array)]
    )

    # Add the text to the bottom of the figure
    fig.text(
        0.01, -0.02,
        f"Mean Flux Measurements:\n{flux_text}",
        fontsize=9, ha='left', va='top'
    )

    # Save figure
    print('Saving: Mean_Flux_Measured.png')
    plt.savefig('Mean_Flux_Measured.png')
    plt.close()


def format_redshift_for_filename(z):
    return f"{z:.1f}".replace('.', '-')


#######################################

"""
Main execution script for generating 1D lognormal mock Lyman-alpha forest spectra.

Functionality:
    - Reads target redshift values and optional Gaussian power spectra.
    - Generates mock Gaussian fields and transforms them into flux transmission fields.
    - Computes power spectra, optical depths, and mean flux.
    - Optionally creates and saves diagnostic plots for each redshift.
    - Aggregates and plots the measured vs. modeled mean flux.
    - Aggregates and fits power spectra over all redshifts.

Command-line Arguments:
    --z_target [str]: Path to redshift file (.txt) or single float (required).
    --power_files [str]: Comma-separated list of power spectrum files (optional).
    --fit_params [str]: Comma-separated values or file for tau0, tau1, nu, sigma2.
    --N_mocks [int]: Number of mocks per redshift (default: 1).
    --plot_* [flags]: Optional flags to generate various diagnostic plots.

Outputs:
    - Transmission files for each mock.
    - Power spectrum plots and fits.
    - Comparison plot of measured vs. modeled mean flux.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate random, 1D lognormal mock spectra for Lyman-alpha forest analyis, given redshift (required) and a gaussian correlation function (optional).")
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
    args = parser.parse_args()

    z0 = PD13_PIVOT_Z

    ### Process Input Data ###
    try:
        z_target = parse_redshift_target(args.z_target)
    except Exception as e:
        print(f"Error reading redshift file: {e}")
        return

    # gaussian power (required)
    power_files = args.power_files.split(',') if args.power_files else [
        None] * len(z_target)
    if len(power_files) != len(z_target):
        if args.power_files:
            raise ValueError(
                "Number of power files must match number of redshifts.")
            # Otherwise, user didn't supply files, which is OK

    # fitting params (optional)
    fitting_params = parse_fitting_params(args.fit_params)
    tau0, tau1, nu, sigma2 = fitting_params

    mean_flux_array = []
    power_per_z_array = []
    k_arrays = []
    dv_per_z_array = []

    # generate mocks for each target redshift
    for z, power_file in zip(z_target, power_files):
        idx = np.where(z_target == z)[0]
        redshift_index = idx[0]

        print(f'\nProcessing z = {z}')
        print(f'N Mocks per z: {args.N_mocks}')

        dv = dv_fid
        dv_per_z_array.append(dv)

        time_1 = time.strftime("%H:%M:%S")
        print("Start Time:  ", time_1)
        start_time = time.time()
        safe_z = format_redshift_for_filename(z)

        k_array, power_array = process_power_file(safe_z, power_file)

        temp_mean_flux = []
        delta_f_array = []

        for i in range(args.N_mocks):
            gaussian_random_field_v = generate_gaussian_random_field()
            gaussian_random_field_k = np.fft.rfft(gaussian_random_field_v) * dv

            kmodes = (np.fft.rfftfreq(
                n=gaussian_random_field_v.size, d=dv) * 2 * np.pi)  # + 1e-12

            delta_b_tilde, delta_b_v, P_k = delta_transform_1d(
                k_array, power_array, gaussian_random_field_k, dv)

            variance_1d = sigma2
            delta_b_z = delta_b_v * a_z(z)
            redshifted_variance_1d = variance_1d * a2_z(z)
            variance_1d_field = delta_b_v.var()
            redshifted_variance_1d_field = variance_1d_field * a2_z(z)

            n_z = lognormal_transform(delta_b_z, redshifted_variance_1d)
            t_z = t_of_z(z)
            x_z = x_of_z(t_z, n_z)
            f_z = f_of_z(x_z)

            # save value for mean flux for each transmission file at this z
            temp_mean_flux.append(mean_flux(z, variance_1d_field, z0))

            delta_f = delta_F(z=z,
                              variance=variance_1d_field,
                              input_flux=f_z)
            delta_f_array.append(delta_f)
            measured_power = P_F(delta_f, dv)

            # export transmission file (velocity_grid is globally defined)
            filepath = export_transmission(safe_z, velocity_grid, f_z)
        print(f"Saved transmission file(s): {filepath}")

        mean_flux_per_z = np.mean(temp_mean_flux)
        mean_flux_array.append(mean_flux_per_z)

        delta_f_per_z = np.concatenate(delta_f_array)
        bin_centers, statistic, popt = fit_and_plot_power(
            delta_f=delta_f_per_z, z=z, dv=dv, safe_z=safe_z,
            N_mocks=args.N_mocks, z_target=z_target,
            all_z='n', plot='y')

        if redshift_index == 0:
            len_k_bins = len(statistic)
            len_delta_f = len(delta_f_per_z)

            power_per_z_array = np.zeros((len(z_target), len_k_bins))
            k_arrays = np.zeros((len(z_target), len_k_bins))
            delta_f_per_z_array = np.zeros((len(z_target), len_delta_f))

        power_per_z_array[redshift_index] = statistic
        k_arrays[redshift_index] = bin_centers
        delta_f_per_z_array[redshift_index] = delta_f_per_z

        time_2 = time.strftime("%H:%M:%S")
        print("End Time:    ", time_2)
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Elapsed Time: {minutes} min {seconds} sec\n")

        ### SAVE PLOTS ##
        print("\n###  Saving Figures  ###\n")
        if args.plot_gaussian_field:
            plot_gaussian_field(
                safe_z, gaussian_random_field_v, space='v', sliced='y')

        if args.plot_gaussian_power:
            plot_gaussian_power(safe_z, k_array, power_array)

        if args.plot_delta_k:
            plot_delta_field(safe_z, kmodes, velocity_grid,
                             delta_b_tilde, space='k', sliced='n')
        if args.plot_delta_v:
            plot_delta_field(safe_z, kmodes, velocity_grid,
                             delta_b_v, space='v', sliced='y')
        if args.plot_delta_z:
            plot_delta_field(safe_z, kmodes, velocity_grid,
                             delta_b_z, space='z', sliced='y')
        if args.plot_nz:
            plot_nz(safe_z, n_z, sliced='y')
        if args.plot_optical_depth:
            plot_optical_depth(safe_z, x_z, sliced='y')
        if args.plot_transmission_v:
            plot_transmission(z, safe_z, velocity_grid,
                              f_z, variance_1d, tau0, tau1, nu,
                              space='v', sliced='y')
        if args.plot_transmission_w:
            plot_transmission(z, safe_z, velocity_grid,
                              f_z, variance_1d, tau0, tau1, nu,
                              space='w', sliced='y')

    ### Measure / Plot Mean Flux ###
    mean_flux_array = np.array(mean_flux_array)
    model_z = np.linspace(min(z_target), max(z_target), 500)
    model_flux_array = np.array([turner24_mf(z) for z in model_z])

    plot_mean_flux(z_target, mean_flux_array, model_z, model_flux_array)

    ### Fit / Plot Power for all z ###
    fit_and_plot_power(z_target=z_target, k_arrays=k_arrays,
                       power_arrays=power_per_z_array,
                       delta_f_array=delta_f_per_z_array,
                       dv_array=dv_per_z_array, all_z='y')


if __name__ == "__main__":
    main()
