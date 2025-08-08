#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import sys
import os

from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from astropy.io import fits
from iminuit import Minuit


#######################################


plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['savefig.bbox'] = 'tight'


#######################################


DESI_EDR_PARAMETERS = (
    7.63089e-02, -2.52054e+00, -1.27968e-01,
    3.67469e+00, 2.85951e-01, 7.33473e+02)

Naim_2020_parameters = (
    0.066, -2.685, -0.22,
    3.59, -0.18, 0.53)

#######################################

# set pivot points for k and z using fiducial power estimate
PD13_PIVOT_K = 0.009  # note: k_0
PD13_PIVOT_Z = 3.0    # note: z0 = 3

gh_degree = 25
gausshermite_xi_deg, gausshermite_wi_deg = np.polynomial.hermite.hermgauss(
    int(gh_degree))
YY1, YY2 = np.meshgrid(gausshermite_xi_deg, gausshermite_xi_deg, indexing='ij')
WW1, WW2 = np.meshgrid(gausshermite_wi_deg, gausshermite_wi_deg, indexing='ij')

# set defaults for p1d fit
default_numvpoints = 2**22

default_dv = 1.0

cutoff_v = 10.0     # for z >= 4.0
# cutoff_v = 15.0   # for z < 4.0

default_v_array = np.arange(default_numvpoints) * default_dv
k_arr = 2. * np.pi * \
    np.fft.rfftfreq((2 * default_numvpoints)-1, d=default_dv) + 1e-12

flux_fitting_z_array = np.linspace(1.8, 5.0, 500)


#######################################


def read_data(filename, expect_two_columns=False):
    """
    Reads data from a .txt or .fits file and returns a 1D or 2D array.

    Args:
        filename (str): Name of the input file. Supported formats: .txt, .fits.
        expect_two_columns (bool): If True, ensures the data has exactly two
                                    columns and returns them as separate arrays.

    Returns:
        np.ndarray: A 1D array if `expect_two_columns` is False.
        tuple[np.ndarray, np.ndarray]: Two 1D arrays (columns) if
            `expect_two_columns` is True.

    Raises:
        ValueError: If an unsupported file format is provided.
        ValueError: If `expect_two_columns` is True but the file does not
                    contain exactly two columns.
        ValueError: If `expect_two_columns` is False but the data is not 1D.
    """
    if filename.endswith('.txt'):
        data = np.loadtxt(filename)
    elif filename.endswith('.fits'):
        with fits.open(filename) as hdul:
            data = hdul[1].data  # Assuming the first extension holds the table
    else:
        raise ValueError("Unsupported file format. Use .txt or .fits")

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
            "Invalid input for --redshift_bins. Provide a valid .txt/.fits file, "
            "a single float value, or a comma-separated list of floats."
        )


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


def evaluatePD13Lorentz(X, A, n, alpha, B, beta, lmd):
    """
    Evaluates the PD13 Lorentzian model for the 1D power spectrum (P1D).

    This function computes the P1D based on the PD13 Lorentzian model
    using input wavenumbers and optional redshifts, with six model parameters.

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


def PD13Lorentz_DESI_EDR(zlist):
    """
    Evaluates the PD13 Lorentzian model for DESI EDR at given redshifts.

    This function computes the 1D power spectrum (P1D) for each target redshift
    using DESI Early Data Release (EDR) parameter values.

    Args:
        zlist (np.ndarray): Array of target redshifts to evaluate.

    Returns:
        np.ndarray: A 2D array where each row corresponds to the evaluated P1D
                    for a given redshift in `zlist`,
                    with shape (len(zlist), len(k_arr)).
    """
    # Start with an empty array the size / shape of input k and z arrays
    p1d_edr_fit = np.empty((zlist.size, k_arr.size))

    # Evaluate P1D for each (k,z), using DESI EDR Param. def. above
    for i, z in enumerate(zlist):
        p1d_edr_fit[i] = evaluatePD13Lorentz((k_arr, z), *DESI_EDR_PARAMETERS)

    return p1d_edr_fit


def a2_z(zp, nu=2.82, z0=PD13_PIVOT_Z):
    """
    Computes the redshift-dependent scaling factor a^2(z).

    This function calculates a^2(z) using a power-law scaling
    relative to a pivot redshift.

    Args:
        zp (float or np.ndarray): Redshift(s) at which to evaluate scale factor.
        nu (float, optional): Exponent controlling redshift evolution
            (default: 2.82).
        z0 (float, optional): Pivot redshift for normalization
            (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: The computed a^2(z) scaling factor.
    """
    return np.power((1. + zp) / (1.+z0), -nu)


def a_z(zp, nu=2.82, z0=PD13_PIVOT_Z):
    """
    Computes the redshift-dependent scaling factor a(z).

    This function calculates a(z) as the square root of a^2(z), following a
    power-law scaling relative to a pivot redshift.

    Args:
        zp (float or np.ndarray): Redshift(s) at which to evaluate scale factor.
        nu (float, optional): Exponent controlling redshift evolution
            (default: 2.82).
        z0 (float, optional): Pivot redshift for normalization
            (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: The computed a(z) scaling factor.
    """
    return np.sqrt(np.power((1. + zp) / (1.+z0), -nu))


def t_of_z(zp, tau0=0.55, tau1=5.1, z0=PD13_PIVOT_Z):
    """
    Computes the optical depth as a function of redshift.

    This function calculates the optical depth, tau(z), based on a power-law
    scaling relative to a pivot redshift.

    Args:
        zp (float or np.ndarray): Redshift(s) at which to evaluate optical depth.
        tau0 (float, optional): Normalization factor for optical depth
                                (default: 0.55).
        tau1 (float, optional): Exponent controlling redshift evolution
                of optical depth (default: 5.1).
        z0 (float, optional): Pivot redshift for normalization
                (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: Computed optical depth(s) for the given redshift(s).
    """
    return tau0 * np.power((1. + zp) / (1.+z0), tau1)


def n_z(zp, nu, sigma2, z0=PD13_PIVOT_Z):
    """
    Applies a lognormal transform to approximate the HI column density as a
    function of redshift.

    This function computes a redshift-dependent factor based on a
    lognormal distribution,
    which is commonly used to approximate the HI column density in
    cosmological models.

    Args:
        zp (float or np.ndarray): Redshift(s) at which to evaluate the transform.
        nu (float): Exponent controlling the redshift evolution
                    of the distribution.
        sigma2 (float): Variance of the Gaussian field.
        z0 (float, optional): Pivot redshift for normalization
                            (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: Lognormal transform, approximating HI column density.
    """
    return np.exp(-a2_z(zp, nu, z0) - sigma2)


def x_of_z(zp, tau0, tau1, nu, sigma2, z0=PD13_PIVOT_Z):
    """
    Modifies optical depth for flux calculations by incorporating z-dependence.

    Args:
        zp (float or np.ndarray): Redshift(s) at which to evaluate.
        tau0 (float): The normalization factor for optical depth.
        tau1 (float): Exponent controlling redshift evolution of optical depth.
        nu (float): Exponent controlling redshift evolution of lognormal transform.
        sigma2 (float): Variance of the Gaussian field.
        z0 (float, optional): Pivot redshift for normalization
            (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: The computed x(z) factor for modifying the flux.
    """
    return t_of_z(zp, tau0, tau1, z0) * np.exp(-a2_z(zp, nu, z0) * sigma2)


def Flux_d_z(delta_g, z, tau0, tau1, nu, sigma2, z0=PD13_PIVOT_Z):
    """
    Calculates the z-dependent flux.

    Args:
        delta_g (float): Perturbations in the Gaussian field.
        z (float or np.ndarray): Redshift(s) at which to evaluate the flux.
        tau0 (float): The normalization factor for optical depth.
        tau1 (float): Exponent controlling the redshift evolution of optical depth.
        nu (float): Exponent controlling redshift evolution of lognormal transform.
        sigma2 (float): Variance of the Gaussian field.
        z0 (float, optional): Pivot redshift for normalization
                            (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: The computed flux at the given redshift(s).
    """
    # Exponential growth from Gaussian perturbations
    e1 = np.exp(2 * a2_z(z, nu / 2, z0) * np.sqrt(2 * sigma2) * delta_g)
    # Optical depth modifier from x(z)
    e2 = x_of_z(z, tau0, tau1, nu, sigma2, z0)

    return np.exp(-e2 * e1)


def lognMeanFluxGH(z, tau0, tau1, nu, sigma2, z0=PD13_PIVOT_Z):
    """
    Computes the mean flux using Gaussian-Hermite quadrature.

    Args:
        z (float or np.ndarray): Redshift(s) at which to evaluate the mean flux.
        tau0 (float): The normalization factor for optical depth.
        tau1 (float): The exponent controlling the redshift evolution of
            optical depth.
        nu (float): Exponent controlling the redshift evolution of the
            lognormal transform.
        sigma2 (float): Variance of the Gaussian field.
        z0 (float, optional): Pivot redshift for normalization
            (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: The computed mean flux for the given redshift(s)
                            using Gaussian-Hermite quadrature.
    """
    XIXI, ZZ = np.meshgrid(gausshermite_xi_deg, z)
    Y = Flux_d_z(XIXI, ZZ, tau0, tau1, nu, sigma2, z0)
    result = np.dot(Y, gausshermite_wi_deg)

    return result / np.sqrt(np.pi)


def process_power_file(power_file):
    """
    Processes a user input (.txt) file containing values for P1D and k
    for different redshift(s).

    Args:
        power_file (str or None): Path to the power spectrum file containing
        columns for redshift (z), wave number (k), and power spectrum values (P1D).
        If None, a default model (DESI EDR) is used instead.

    Returns:
        dict: A dictionary of redshift bins with keys 'k' and 'P1D',
            The dictionary structure is
            {z: {'k': [k1, k2, ...],'P1D': [P1D1, P1D2, ...]}}.
        np.ndarray: A sorted array of unique redshift values from the
                input file, or an empty array if the file is not provided or
                the data is invalid.

    Raises:
        Exception: If there is an error reading the power spectrum file.
    """
    if power_file:
        try:
            data = np.loadtxt(power_file)
            if data.shape[1] < 3:  # Ensure at least 3 columns
                print(
                    "Error: Power spectrum file must have at least 3 columns (z, k, P1D).")
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
    Computes the velocity grid properties based on an
    input wave number array (k_array).

    Args:
        k_array (np.ndarray): A 1D array of wave numbers (k),
                            containing least two values.

    Returns:
        dv (float): The velocity spacing, computed based on the k_array spacing.
        numvpoints (int): The total number of velocity points,
                        equal to the length of `k_array`.

    Raises:
        ValueError: If the `k_array` contains fewer than two points,
                    a ValueError is raised.
    """
    numvpoints = len(k_array)

    if numvpoints < 2:
        raise ValueError(
            "k_array must have at least two points to determine spacing.")

    # Check if k-array is uniformly spaced
    dk_values = np.diff(k_array)
    if np.allclose(dk_values, dk_values[0]):
        dk = dk_values[0]
        dv = np.pi / (dk * numvpoints)
        # dv is derived from the inverse FFT relation:
        # dk = 2π / (N * dv) → dv = π / (dk * N) for rfft

    else:
        # More general case: Use range of k values
        print('k-array is not evenly spaced')

    return dv, numvpoints


def process_power_data(z_target, power_file=None):
    """
    Loads or generates power spectrum data and computes velocity grid properties
    for each target redshift.

    If a power spectrum file is provided, the function reads redshift-dependent
    k and P1D data. Otherwise, it uses a default model.

    Args:
        z_target (array-like): List or array of redshifts to process.
        power_file (str, optional): Path to file containing columns [z, k, P1D].
            If None, a default model (PD13Lorentz_DESI_EDR) is used.

    Returns:
        k_array (np.ndarray): Array of k values per redshift (dtype=object).
        P1D_array (np.ndarray): Array of P1D values per redshift (dtype=object).
        dv_array (np.ndarray): Velocity spacing (dv) per redshift.
        numvpoints_array (np.ndarray): Number of velocity points per redshift.

    Raises:
        SystemExit: If data is missing for any redshift in file mode or if k/P1D
                    lengths mismatch.
        ValueError: On processing failure of the power file.
    """
    if power_file:
        print('\n(P,k,z) file provided, using as target P1D')
        grouped_data, zlist = process_power_file(power_file)

        if not grouped_data:
            print("Error: Failed to process power file.")
            return None, None, None, None, None

        P1D_array = []
        k_array = []
        dv_array = []
        numvpoints_array = []

        # Loop through each redshift and extract corresponding (k, P1D) values
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

        # Convert Python lists to NumPy arrays for uniform return format
        P1D_array = np.array(P1D_array, dtype=object)
        k_array = np.array(k_array, dtype=object)
        dv_array = np.array(dv_array)
        numvpoints_array = np.array(numvpoints_array)

    else:
        print("No (P,k,z) file provided, using default model (PD13Lorentz_DESI_EDR)")
        k_array = k_arr
        zlist = z_target
        P1D_array = PD13Lorentz_DESI_EDR(z_target)

        dv_array = np.full(len(zlist), default_dv)
        numvpoints_array = np.full(len(zlist), default_numvpoints)

        for i, z in enumerate(zlist):
            if len(k_array) != len(P1D_array[i]):
                print(
                    f"Error: Mismatch - k array has {len(k_array)} elements, "
                    f"but P1D array has {len(P1D_array[i])} elements.")
                return None, None, None, None, None

    print("Power spectrum data successfully assigned.")
    return k_array, P1D_array, dv_array, numvpoints_array


def scale_input_power(redshift_index, k_array, P1D_array, power_file=None):
    """
    Extracts and scales P1D data for a specific redshift index.

    Uses full or partial k-range depending on whether a custom power file
    was provided or the default model is used.

    Args:
        redshift_index (int): Index of the redshift in the arrays.
        k_array (np.ndarray): Array of k-values per redshift.
        P1D_array (np.ndarray): Array of P1D values per redshift.
        power_file (str, optional): Path to custom power spectrum file.
            If None, the default model is assumed.

    Returns:
        k_array_input (np.ndarray): k-values for the specified redshift.
        p1d_input (np.ndarray): P1D values for the specified redshift.
    """
    if power_file:
        k_array_input = k_array[redshift_index]
        k_array_input = np.array(k_array_input, dtype=float)
        p1d_input = P1D_array[redshift_index]
    else:
        # Skip first k value if using default model
        k_array_input = k_array[1:]
        p1d_input = P1D_array[redshift_index][1:]
    return k_array_input, p1d_input


def process_flux_data(flux_file=None):
    """
    Loads mean flux data from file or defaults to Turner et al. (2024) model.

    Args:
        flux_file (str, optional): Path to flux data file with columns [z, F].
            If None, the default model from Turner et al. (2024) is used.

    Returns:
        flux_z_array (np.ndarray): Redshift values.
        flux_array (np.ndarray): Corresponding mean flux values.
        flux_model (str): 'User Input' or 'Turner et al., 2024'.

    Raises:
        ValueError: If redshift and flux array lengths mismatch.
    """
    if flux_file:
        print('\n(F,z) file provided, using as target mean flux')
        try:
            flux_z_array, flux_array = read_data(
                flux_file, expect_two_columns=True)
            flux_model = 'User Input'
        except Exception as e:
            print(f"Error reading flux file: {e}")
            return
    else:
        print("No (F,z) file provided, using default model (Turner et al., 2024)")
        flux_array = turner24_mf(flux_fitting_z_array)
        flux_z_array = flux_fitting_z_array
        flux_model = 'Turner et al., 2024'

    if len(flux_z_array) != len(flux_array):
        print(
            f"Error: Mismatch - z array has {len(flux_z_array)} elements, "
            f"but flux array has {len(flux_array)} elements.")
        return

    print("Flux data successfully assigned.")
    return flux_z_array, flux_array, flux_model


def fit_mean_flux(flux_array, flux_z_array, z0=PD13_PIVOT_Z):
    """
    Fits the mean flux evolution model parameters to provided data.

    Uses least-squares minimization to estimate optical depth and lognormal
    transform parameters from mean flux measurements.

    Args:
        flux_array (np.ndarray): Observed mean flux values.
        flux_z_array (np.ndarray): Corresponding redshift values.
        z0 (float, optional): Pivot redshift (default: PD13_PIVOT_Z).

    Returns:
        tuple: Optimal parameters of the flux model:
            - tau0 (float): Optical depth normalization.
            - tau1 (float): Redshift evolution exponent for optical depth.
            - nu (float): Lognormal transform redshift scaling exponent.
            - sigma2 (float): Gaussian field variance.
    """
    print('\n###  Fitting Mean Flux Parameters  ###\n')

    flux_fit_precision = 1e-5
    Err_flux_fit = flux_array * flux_fit_precision + 1e-8

    def flux_fit_cost(tau0, tau1, nu, sigma2):
        d = (flux_array - lognMeanFluxGH(flux_z_array,
             tau0, tau1, nu, sigma2, z0)) / Err_flux_fit
        return d.dot(d)

    # Set initial guesses for fitting parameters
    tau0_0, tau1_0, nu_0, sigma2_0 = 0.55, 5.1, 2.82, 2.0

    mini = Minuit(flux_fit_cost, tau0_0, tau1_0, nu_0, sigma2_0)
    mini.errordef = Minuit.LEAST_SQUARES
    mini.migrad()

    tau0, tau1, nu, sigma2 = mini.values

    print(f'tau0 = {tau0}')
    print(f'tau1 = {tau1}')
    print(f'nu = {nu}')
    print(f'sigma2 = {sigma2}')

    return mini.values


def lognXiFfromXiG_pointwise(z, xi_gauss, tau0, tau1, nu, sigma2, z0=PD13_PIVOT_Z):
    """
    Converts Gaussian correlation function xi_G to flux correlation xi_F at a
    given redshift.

    This applies a lognormal transformation based on fitted flux model
    parameters.

    Args:
        z (float): Redshift value.
        xi_gauss (float): Gaussian correlation function value.
        tau0 (float): Optical depth normalization.
        tau1 (float): Optical depth redshift exponent.
        nu (float): Lognormal scaling exponent.
        sigma2 (float): Gaussian field variance.
        z0 (float, optional): Pivot redshift (default: PD13_PIVOT_Z).

    Returns:
        float: Corresponding flux correlation xi_F value.
    """
    xi_sine = np.clip(xi_gauss / sigma2, -1, 1)
    xi_cosine = np.sqrt(1 - xi_sine**2)
    XI_VEC = np.array([xi_sine, xi_cosine])

    YY2_XI_VEC_WEIGHTED = np.dot(
        XI_VEC, np.array([YY1, YY2]).transpose(1, 0, 2))

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


def objective(xi_g, z, xi_f_target, tau0, tau1, nu, sigma2, z0=PD13_PIVOT_Z):
    """
    Objective function for xi_G fitting: computes residuals to target xi_F.

    Args:
        xi_g (array-like): Trial values for Gaussian correlation function.
        z (float): Redshift at which xi_F is evaluated.
        xi_f_target (array-like): Target flux correlation values.
        tau0, tau1, nu, sigma2 (float): Flux model parameters.
        z0 (float, optional): Pivot redshift (default: PD13_PIVOT_Z).

    Returns:
        np.ndarray: Residuals between predicted and target xi_F values.
    """
    xi_f_calculated = np.array([lognXiFfromXiG_pointwise(z, xi_g_i, tau0,
                                                         tau1, nu, sigma2,
                                                         z0) for xi_g_i in xi_g])
    return xi_f_calculated - xi_f_target


def solve_xi_optimized(z_target, redshift_index, size, v_array_downsampled,
                       xi_f_target, tau0, tau1, nu, sigma2, z0=PD13_PIVOT_Z):
    """
    Solves for optimized Gaussian correlation function xi_G that reproduces
    the input flux correlation xi_F via a lognormal transform.

    Uses nonlinear least squares to match xi_F at downsampled velocity points.

    Args:
        z_target (array-like): Array of redshift values.
        redshift_index (int): Index of redshift to solve for.
        size (int): Total number of velocity points.
        v_array_downsampled (np.ndarray): Velocity values used for fitting.
        xi_f_target (np.ndarray): Target flux correlation values at each velocity.
        tau0, tau1, nu, sigma2 (float): Flux model parameters.
        z0 (float, optional): Pivot redshift (default: PD13_PIVOT_Z).

    Returns:
        tuple:
            v_fit (np.ndarray): Velocity values used in fitting.
            xi_g_fit (np.ndarray): Fitted Gaussian correlation values.
            xi_f_optimized (np.ndarray): Optimized flux correlation values.
            xi_g_full (np.ndarray): Interpolated xi_G values over full v-range.
            zero_point (float): xi_G at v=0 (used for sigma² consistency check).
    """
    print('\n(Fitting xi_f)')
    print(f"N-Points:     {size}")

    time_1 = time.strftime("%H:%M:%S")
    print("Start Time:  ", time_1)
    start_time = time.time()

    # cutoff_v = 10.0
    v_fit_mask = v_array_downsampled >= cutoff_v

    v_fit = v_array_downsampled[v_fit_mask]

    xi_f_fit = xi_f_target[v_fit_mask]
    xi_g_initial_guess = np.full(xi_f_fit.size, 0.1)

    # This function solves for xi_G optimized by minimizing the difference
    # between calculated and target xi_F.
    result = least_squares(objective, xi_g_initial_guess,
                           args=(z_target[redshift_index],
                                 xi_f_fit, tau0, tau1, nu, sigma2, z0))
    xi_g_fit = result.x

    xi_g_extrapolator = interp1d(
        v_fit, xi_g_fit,
        kind='linear',
        fill_value='extrapolate',
        bounds_error=False)

    xi_g_full = xi_g_extrapolator(v_array_downsampled)

    zero_point = xi_g_full[0]

    time_2 = time.strftime("%H:%M:%S")
    print("End Time:    ", time_2)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed Time: {minutes} min {seconds} sec\n")

    print(f"sigma2 from Mean Flux fit:      {sigma2}")
    print(f"Calculated sigma2 from Xi_G(0): {xi_g_fit[0]}")
    print(
        f"Difference:                       {np.abs(sigma2 - xi_g_fit[0])}")
    print(f"Extrapolated zero point:        {zero_point}")
    print(f"Difference:                     {sigma2 - zero_point}")

    xi_f_optimized = np.array([lognXiFfromXiG_pointwise(z_target[redshift_index],
                                                        xi_g_i, sigma2, tau0,
                                                        tau1, nu, z0)
                               for xi_g_i in xi_g_full])

    return v_fit, xi_g_fit, xi_f_optimized, xi_g_full, zero_point


def save_xiG_half(v_array_half, xi_g_half, safe_z, save_cf=True):
    """
    Saves the half correlation function xi_G to a text file.

    The output is stored in the '../CF' directory relative to the script, with
    filenames formatted as '{safe_z}_xiG_half_output.txt'.

    Args:
        v_array_half (np.ndarray): Velocity array for the half correlation function.
        xi_g_half (np.ndarray): Gaussian correlation values corresponding to velocities.
        safe_z (str): Safe string identifier for the redshift (used in filename).
        save_cf (bool, optional): If True (default), writes the data to file.
    """
    # Define the path to the output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'CF')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    if save_cf:
        # Stack velocity and correlation into two-column format
        export_data_half = np.column_stack((v_array_half, xi_g_half))

        # Save to text file
        output_path = os.path.join(
            output_dir, f'test{safe_z}_xiG_half_output.txt')
        np.savetxt(output_path, export_data_half, fmt="%.6f", delimiter="\t",
                   header="Velocity\tXi_G_fit")


# mirrors the correlation function to get the full one for P_G calculation
def mirror_xiG(safe_z, save_cf):
    """
    Mirrors a half Gaussian correlation function xi_G to produce a full
    symmetric function.

    Loads the saved half correlation data, interpolates it with a cubic spline,
    and mirrors it around the endpoint to extend to twice the original length.

    Args:
        safe_z (str): Redshift identifier string used to locate the half xi_G file.
        save_cf (str): If 'full', the mirrored full xi_G is saved to a file.
                       If empty string or other value, no file is saved.

    Returns:
        tuple:
            v_extended (np.ndarray): Full symmetric velocity array, extended
                    around midpoint.
            file_xig_extended (np.ndarray): Full symmetric xi_G array, mirrored
                    about midpoint.
    """
    # Set the output directory (../CFs relative to this script)
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'CF')
    os.makedirs(output_dir, exist_ok=True)

    # Load the half CF file
    half_cf_path = os.path.join(
        output_dir, f'test{safe_z}_xiG_half_output.txt')
    data = np.loadtxt(half_cf_path)
    file_v = data[:, 0]         # First column
    file_xiG = data[:, 1]       # Second column

    def truncate_trailing_zeros(x, y):
        # Find the last index where y is nonzero
        last_nonzero_index = np.max(np.nonzero(y))

        # Ensure at least one zero remains at the end
        if last_nonzero_index < len(y) - 1 and y[last_nonzero_index + 1] == 0:
            last_nonzero_index += 1  # Keep the first zero after the last nonzero value

        return x[:last_nonzero_index + 1], y[:last_nonzero_index + 1]

    # v_truncated, xiG_truncated = truncate_trailing_zeros(file_v, file_xiG)
    v_truncated, xiG_truncated = file_v, file_xiG

    # Interpolate linearly (constant dv)
    file_v_fine = np.linspace(v_truncated.min(), v_truncated.max(), 2**20)
    cs = CubicSpline(v_truncated, xiG_truncated)
    file_xi_g_fine = cs(file_v_fine)

    dv_fit_fine = np.diff(file_v_fine)
    v_spacing = np.mean(dv_fit_fine)

    v_extended = np.concatenate(
        [file_v_fine, file_v_fine + file_v_fine[-1] + v_spacing])

    # Mirror the correlation function values
    file_xig_extended = np.concatenate([file_xi_g_fine, file_xi_g_fine[::-1]])
    export_data_full = np.column_stack((v_extended, file_xig_extended))

    if save_cf == 'full':
        full_cf_path = os.path.join(
            output_dir, f'test{safe_z}_xiG_full_output.txt')
        np.savetxt(full_cf_path, export_data_full, fmt="%.6f",
                   delimiter="\t", header="Velocity\tXi_G_fit")

    return v_extended, file_xig_extended


def save_PG(safe_z):
    """
    Loads the full xi_G correlation function, computes its power spectrum via FFT,
    and saves the result to the P_G directory.

    Args:
        safe_z (str): Redshift identifier string used for filename convention.
    """
    # Define paths relative to this script
    base_dir = os.path.dirname(__file__)
    cf_dir = os.path.join(base_dir, '..', 'CF')
    p1d_dir = os.path.join(base_dir, '..', 'P_G')
    os.makedirs(p1d_dir, exist_ok=True)

    # Load full CF data
    path_to_cf = os.path.join(cf_dir, f'test{safe_z}_xiG_full_output.txt')
    data = np.loadtxt(path_to_cf)
    full_v = data[:, 0]
    full_xiG = data[:, 1]
    full_v_spacing = full_v[1] - full_v[0]

    # Compute power spectrum
    file_xig_power = np.fft.rfft(full_xiG) * full_v_spacing
    file_xig_power = np.abs(file_xig_power)  # np.abs vs .real ?
    file_xig_karr = 2 * np.pi * \
        np.fft.rfftfreq(len(full_xiG), d=full_v_spacing)

    # Save power spectrum
    export_filename = os.path.join(p1d_dir, f'testP_G-{safe_z}.txt')
    np.savetxt(export_filename, np.column_stack((file_xig_karr, file_xig_power)),
               header='k [km/s]^-1\tP(k)', fmt='%.6e')


def recover_power(k_arr, xi, v_arr, cf_size):
    """
    Recovers the P1D from the correlation function using FFT.
    (not the same as loading P_G and generating mocks)

    Args:
        k_arr (np.ndarray): Array of k-values corresponding to the power spectrum.
        xi (np.ndarray): Array of correlation function values (either full or half).
        v_arr (np.ndarray): Array of velocity values corresponding to the
                            correlation function.
        cf_size (str): Specifies the type of correlation function. Can be
                        'half' (for half CF) or 'full' (for full CF).

    Returns:
        tuple: A tuple containing:
            - k_arr (np.ndarray): Array of k-values corresponding to
                                    the recovered power.
            - power (np.ndarray): Array of power spectrum values.

    Raises:
        value error: "cf_size must be either 'half' or 'full'"
    """
    dv = np.mean(np.diff(v_arr))

    if cf_size == 'half':
        # Mirror the CF to create full symmetric CF
        v_mirrored = np.concatenate([v_arr, v_arr + v_arr[-1] + dv])
        xi_mirrored = np.concatenate([xi, xi[::-1]])

        power = np.fft.rfft(xi_mirrored) * dv
        k_arr = 2 * np.pi * np.fft.rfftfreq(len(xi_mirrored), d=dv)

        return k_arr, power

    elif cf_size == 'full':
        # Use the full CF directly
        # Factor of 2 to normalize for full range
        power = np.fft.rfft(xi) * dv * 2
        k_arr = 2 * np.pi * np.fft.rfftfreq(len(xi), d=dv)

        return k_arr, power

    else:
        raise ValueError("cf_size must be either 'half' or 'full'")


def interpolate_power_for_fft(new_size, v_spacing, k_data, p_data, log_interp=True):
    """
    Interpolates a power spectrum onto an FFT-compatible k-grid derived from a
    velocity grid.

    Optionally performs interpolation in log-log space for smoother behavior.

    Args:
        new_size (int): Number of velocity samples.
        v_spacing (float): Spacing of the velocity grid (dv).
        k_data (np.ndarray): Original k-values.
        p_data (np.ndarray): Original P(k) values.
        log_interp (bool): If True, interpolate in ln(k) vs ln(P(k)) space.

    Returns:
        tuple:
            - v_array (np.ndarray): Velocity grid used to define FFT k-grid.
            - k_fft (np.ndarray): FFT-compatible k-grid.
            - p_interp (np.ndarray): Interpolated P(k) on the FFT k-grid.
    """
    # Create velocity grid
    v_array = np.arange(new_size) * v_spacing

    # Generate FFT k-grid (size matches real FFT of mirrored velocity)
    fft_k = 2. * np.pi * \
        np.fft.rfftfreq((2 * new_size) - 1, d=v_spacing) + 1e-12

    # Interpolate the power spectrum at these k-values
    valid = (k_data > 0) & (p_data > 0)
    k_data = np.ravel(k_data[valid])
    p_data = np.ravel(p_data[valid])

    # Interpolate in log-log space if requested
    if log_interp:
        ln_k_data = np.log(k_data)
        ln_p_data = np.log(p_data)

        cs = CubicSpline(ln_k_data, ln_p_data, extrapolate=True)
        ln_p_fft = cs(np.log(fft_k))
        p_fft = np.exp(ln_p_fft)
    else:
        cs = CubicSpline(k_data, p_data, extrapolate=True)
        p_fft = cs(fft_k)

    return v_array, fft_k, p_fft


def downsample_array(v_array, xi_array, downsample_size, log_scale=True):
    """
    Downsamples velocity and correlation function arrays, typically for plotting.

    Uses either logarithmic or linear sampling of the velocity magnitude
    to preserve shape and structure in lower-resolution views.

    Args:
        v_array (np.ndarray): Velocity array (can be symmetric or positive half).
        xi_array (np.ndarray): Correlation function values.
        downsample_size (int): Number of output points.
        log_scale (bool, optional): Use logarithmic spacing if True (default).

    Returns:
        tuple:
            - v_array_downsampled (np.ndarray): Downsampled velocity array
                    (positive only).
            - xif_downsampled (np.ndarray): Interpolated xi values on
                    downsampled grid.
            - dv_downsampled (np.ndarray): Spacing between downsampled velocity
                    values.
    """
    velocity_abs = np.abs(v_array[1:])

    if log_scale:
        log_v_min, log_v_max = np.log10(
            1 + velocity_abs.min()), np.log10(1 + velocity_abs.max())
        v_array_downsampled = np.logspace(
            log_v_min, log_v_max, downsample_size) - 1  # Shift back
    else:
        v_array_downsampled = np.linspace(
            velocity_abs.min(), velocity_abs.max(), downsample_size)

    # Interpolate xif values onto the downsampled velocity grid
    cs = CubicSpline(velocity_abs, xi_array[1:])
    xif_downsampled = cs(v_array_downsampled)

    dv_downsampled = np.diff(v_array_downsampled)

    return v_array_downsampled, xif_downsampled, dv_downsampled


def plot_mean_flux(z, target_flux, tau0, tau1, nu, sigma2,
                   flux_model, z0=PD13_PIVOT_Z):
    """
    Plot the best-fit mean flux compared to a target flux model over redshift.

    Generates a two-panel figure:
        - Top: Mean flux from the lognormal model vs. the target flux.
        - Bottom: Percent difference between model and target flux.

    Args:
        z (np.ndarray): Redshift array.
        target_flux (np.ndarray): Target mean flux values to compare against.
        tau0 (float): Normalization factor for optical depth.
        tau1 (float): Exponent controlling redshift evolution of optical depth.
        nu (float): Exponent controlling redshift evolution of the lognormal
                transform.
        sigma2 (float): Variance of the Gaussian field.
        flux_model (str): Name of the target flux model (for labeling).
        z0 (float, optional): Pivot redshift for normalization
                (default: PD13_PIVOT_Z).

    Saves:
        'Mean_Flux_Fit.png' — plot of the best-fit mean flux and its residuals.
    """
    print('Saving: Mean_Flux_Fit.png')

    # Compute best-fit mean flux
    best_fit_flux = lognMeanFluxGH(z, tau0, tau1, nu, sigma2, z0)

    # Compute percent difference
    percent_diff = 100 * (target_flux - best_fit_flux) / best_fit_flux

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   constrained_layout=True)

    # Top panel: mean flux comparison
    ax1.plot(z, target_flux, color='tab:blue',
             lw=6, alpha=0.5, label=f'Model: {flux_model}')
    ax1.plot(z, best_fit_flux, color='black',
             ls='--', label='Fit')
    ax1.set_ylabel(r'$\bar F(z)$')
    ax1.legend(loc='lower left')
    ax1.grid()

    # Add parameter text: optional, used for debugging mostly
    # txt = f'tau0 = {tau0}\ntau1 = {tau1}\nnu = {nu}\nsigma2 = {sigma2}'
    # ax1.text(0.02, 0.1, txt, ha='left', va='center',
    #          fontsize=12, transform=ax1.transAxes)

    # Bottom panel: percent difference
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.plot(z, percent_diff, color='darkred')
    ax2.set_xlabel('z')
    ax2.set_ylabel('% Difference')
    ax2.grid()

    # Save figure
    plt.savefig('Mean_Flux_Fit.png')


def plot_target_power(z, k_array_input, p1d_input, k_array_fine, p1d_fine):
    """
    Plot and save the input and interpolated 1D power spectra at a given redshift.

    Args:
        z (float): Redshift of the target power spectrum.
        k_array_input (np.ndarray): k-values for the original power spectrum.
        p1d_input (np.ndarray): Original P1D values.
        k_array_fine (np.ndarray): Finer k-grid for interpolation.
        p1d_fine (np.ndarray): Interpolated P1D values.

    Output:
        Saves a log-log plot as '{z}_P1D_target.png'.
    """
    print(rf'Saving: {z}_P1D_target.png')
    plt.figure()

    plt.loglog(k_array_input, p1d_input, alpha=0.7,
               label=f'input P1D, z={z}, N={k_array_input.size}')
    plt.loglog(k_array_fine[1:], p1d_fine[1:], color='tab:orange', ls='--',
               label=f'interpolated, z={z}, N={k_array_fine.size}')
    plt.legend()
    plt.xlabel(r'k $[km/s]^{-1}$')
    plt.ylabel(r'$P_{1D}$ ')

    # save figure
    plt.savefig(rf'{z}_P1D_target.png')


def plot_target_xif(z, new_v_array, xif_interp_fit, v_array_downsampled,
                    xif_target_downsampled, dv):
    """
    Plot and save the target and interpolated xi_F correlation functions at a
    given redshift.

    Args:
        z (float): Redshift of the correlation function.
        new_v_array (np.ndarray): Velocity array for interpolated xi_F.
        xif_interp_fit (np.ndarray): Interpolated xi_F values.
        v_array_downsampled (np.ndarray): Downsampled velocity array.
        xif_target_downsampled (np.ndarray): Downsampled xi_F values.
        dv (float): Velocity resolution in km/s.

    Output:
        Saves a semilog plot as '{z}_xi_F_target.png'.
    """
    print(rf'Saving: {z}_xi_F_target.png')
    plt.figure()
    plt.semilogx(new_v_array[1:], xif_interp_fit[1:],
                 label='interpolated, N =' + str(new_v_array.size))
    plt.semilogx(v_array_downsampled, xif_target_downsampled,
                 label='downsampled, N =' + str(v_array_downsampled.size),
                 ls='--', marker='o')
    plt.xlabel('v [km/s]')
    plt.ylabel(r'$\xi_F$')
    plt.title(r'$\xi_F$ Target,  z = '+str(z))
    plt.vlines(dv, 0, 0.20, color='black', ls='--',
               label=f'dv = {dv:.2f} km/s')
    plt.legend()
    plt.tight_layout()

    # save figure
    plt.savefig(rf'{z}_xi_F_target.png')


def plot_xif_fit(z, v_array_downsampled, xi_f_target, xi_f_optimized, dv):
    """
    Plot the target and optimized xi_F correlation functions along with their
    residuals.

    Args:
        z (float): Redshift for the xi_F fit.
        v_array_downsampled (np.ndarray): Downsampled velocity array (km/s).
        xi_f_target (np.ndarray): Target xi_F correlation values.
        xi_f_optimized (np.ndarray): Optimized xi_F correlation values from the fit.
        dv (float): Velocity resolution (km/s), shown as a vertical reference line.

    Saves:
        {z}_xi_F_fit.png: Plot comparing target and optimized xi_F.
        {z}_xi_F_fit_residual.png: Plot of residuals (target minus fit).
    """
    print(rf'Saving: {z}_xi_F_fit.png')
    plt.figure()
    plt.semilogx(v_array_downsampled, xi_f_target, alpha=0.5,
                 label=r'$\xi_F$ Target', color='tab:blue')
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
    # max_y = dif_xi_f[np.argmax(abs_dif_xi_f)]
    # max_x = v_array_downsampled[np.argmax(abs_dif_xi_f)]

    print(rf'Saving: {z}_xi_F_fit_residual.png')
    plt.figure()
    plt.semilogx(v_array_downsampled, dif_xi_f,
                 label=f'z = {z}', color='tab:blue')
    plt.axvline(x=dv, color='black', linestyle='--',
                label=f"dv: {dv:.2f} km/s")
    plt.xlabel('v [km/s]')
    plt.ylabel(r'$\Delta \xi_F$')
    plt.legend()
    plt.title(r"$\xi_F$ Residual (Model - Fit)")
    plt.tight_layout()

    # save figure
    plt.savefig(rf'{z}_xi_F_fit_residual.png')


def plot_xig_fit(z, v_array_downsampled, xi_g_optimized,
                 zero_point, v_extrapolated, xi_g_extrapolated):
    """
    Plot the optimized xi_g correlation function with extrapolated and fixed
    reference points.

    Args:
        z (float): Redshift value for the xi_g fit.
        v_array_downsampled (np.ndarray): Downsampled velocity values (km/s).
        xi_g_optimized (np.ndarray): Optimized xi_g correlation function values.
        zero_point (float): xi_g value at zero velocity (fixed reference).
        v_extrapolated (np.ndarray): Velocity values used for extrapolation.
        xi_g_extrapolated (np.ndarray): Extrapolated xi_g values.

    Saves:
        {z}_xi_G_fit.png: Plot comparing optimized, extrapolated, and
                          fixed-point xi_g.
    """
    print(rf'Saving: {z}_xi_G_fit.png')
    plt.figure()

    plt.plot(v_array_downsampled, xi_g_optimized,
             'o', label=r'$\xi_g$ Fit', color='tab:blue')

    plt.plot(0, zero_point, 'ro', label=f'Fixed Point (0, {zero_point:.3f})')

    plt.plot(v_extrapolated, xi_g_extrapolated, '-',
             label='CS Extrapolation', color='tab:orange')

    plt.xscale('log')
    plt.xlabel('v [km/s]')
    plt.ylabel(r'$\xi_g$')
    plt.legend()
    plt.grid()
    plt.ylim(None, xi_g_extrapolated.max()+0.1)
    plt.tight_layout()

    # save figure
    plt.savefig(rf'{z}_xi_G_fit.png')


def plot_xi_f_recovered(z, v_fine, xif_fine,
                        v_array_downsampled, xif_target_downsampled,
                        v_extrapolated, xi_f_optimized_extrapolated, dv):
    """
    Plot the recovered xi_F correlation function including re-interpolated,
    downsampled target, and extrapolated fits.

    Args:
        z (float): Redshift value for the plot.
        v_fine (np.ndarray): Fine velocity grid for re-interpolated xi_F.
        xif_fine (np.ndarray): Re-interpolated xi_F values on fine velocity grid.
        v_array_downsampled (np.ndarray): Downsampled velocity array.
        xif_target_downsampled (np.ndarray): Downsampled target xi_F values.
        v_extrapolated (np.ndarray): Velocity values used for extrapolation.
        xi_f_optimized_extrapolated (np.ndarray): Extrapolated optimized xi_F values.
        dv (float): Velocity spacing (km/s) shown as a vertical reference line.

    Saves:
        {z}_xi_f_recovery.png: Plot comparing re-interpolated, downsampled,
                               and extrapolated xi_F.
    """
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
                         mirrored_fit_power, w_fit_k, e_p1d, z_id, delta_P_real,
                         z_target):
    """
    Plot the recovered 1D power spectrum along with percent difference residuals.

    Args:
        z (str): Redshift string used in plot titles and filenames.
        k_array_input (np.ndarray): Array of k-values for the power spectrum
                (momentum space).
        p1d_input (np.ndarray): Measured power spectrum values corresponding to
                k_array_input.
        w_k (np.ndarray): Boolean or index array to select a subset of
                k_array_input and p1d_input.
        mirrored_fit_k_arr (np.ndarray): k-values for the fitted power spectrum
                after mirroring.
        mirrored_fit_power (np.ndarray): Fitted power spectrum values
                corresponding to mirrored_fit_k_arr.
        w_fit_k (np.ndarray): Boolean or index array selecting the fitted
                k-values.
        e_p1d (np.ndarray): Array of error values for the power spectrum, indexed
                by redshift.
        z_id (int): Index to select the current redshift from e_p1d.
        delta_P_real (np.ndarray): Residuals array representing
                (model - fit) / model.
        z_target (float): Redshift used in model evaluation (
                (e.g., for external fits).

    Saves:
        {z}_recovered_power.png: Figure containing the power spectrum
                                 and percent residual plots.
    """
    print(rf'Saving: {z}_recovered_power.png')

    naim_2020_fit = evaluatePD13Lorentz(
        (k_array_input[w_k], z_target), *Naim_2020_parameters)
    percent_diff = 100 * delta_P_real
    percent_diff2 = 100 * \
        (naim_2020_fit - p1d_input[w_k].real) / p1d_input[w_k].real

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8), sharex=True,
    #                                     gridspec_kw={'height_ratios': [4, 1, 1]})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [4, 1]})

    # Top subplot: Power spectrum
    # ax1.axvspan(0.05, 0.1, alpha=0.2, color='grey')
    # ax1.loglog(k_array_input[w_k], naim_2020_fit, lw=2,
    #            color='black', label='Karacayli et al. (2020)')
    ax1.loglog(k_array_input[w_k], p1d_input[w_k].real, lw=5,
               color='tab:blue', label='DESI EDR', alpha=0.7)
    ax1.loglog(mirrored_fit_k_arr[w_fit_k], mirrored_fit_power[w_fit_k].real,
               color='tab:orange', ls='--', lw=2,
               label=r'This Work')

    ymin_data = mirrored_fit_power[w_fit_k].real.min()
    ymax_data = mirrored_fit_power[w_fit_k].real.max()

    ymin = max(ymin_data / (10**0.5), 1e-6)
    # adjust 0.5 to 1.0 for full order of mag
    ymax = ymax_data + 100

    # ax1.set_ylim(ymin, ymax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_ylabel(rf'$P(k)$   (z = {z})')
    ax1.grid(True)
    ax1.legend(loc='lower left')

    # Bottom subplot: Percent residuals
    ax2.semilogx(k_array_input, percent_diff, color='darkred')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True)

    # ax3.semilogx(k_array_input[w_k], percent_diff2, color='black')
    # ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    # ax3.grid(True)

    # Compute dynamic y-axis range only for values within x_min and x_max
    x_min, x_max = k_array_input[w_k].min(), k_array_input[w_k].max()
    mask = (k_array_input >= x_min) & (k_array_input <= 0.05)
    percent_diff_in_range = percent_diff[mask]

    # Compute max abs percent difference in the plotting range
    if np.any(~np.isnan(percent_diff_in_range)):
        y_max = np.nanmax(np.abs(percent_diff_in_range))
        buffer = 0.05 * y_max  # Add some padding
        ax2.set_ylim(-y_max - buffer, y_max + buffer)
    else:
        ax2.set_ylim(-10, 10)  # Fallback in case of NaNs

    # ax3.set_xlabel(r'k $[km/s]^{-1}$')
    ax2.set_xlabel(r'k $[km/s]^{-1}$')

    fig.text(0.04, 0.25, "% Difference", va='center',
             rotation='vertical', fontsize=16)
    # ax3.set_ylabel("")  # remove individual y-labels to avoid overlap
    ax2.set_ylabel("")  # remove individual y-labels to avoid overlap

    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    # ax3.set_xlim(x_min, x_max)

    # Clean x ticks on top plot
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.savefig(rf'{z}_recovered_power.png')
    plt.close()


#######################################


def main():
    parser = argparse.ArgumentParser(
        description="Solve for optimal Gaussian correlation function, given redshift (required), P(k,z) (optional), and F(z) (optional)")
    parser.add_argument('--power_file', type=str,
                        help='Path to input (P,k) file (.txt) containing k and P1D arrays')
    parser.add_argument('--flux_file', type=str,
                        help='Path to input (F,z) file (.txt) containing z and mean flux arrays')
    parser.add_argument('--z_target', type=str, required=True,
                        help='Path to input file (.txt) containing target redshift values, OR a single float input')
    parser.add_argument('--plot_mean_flux', action='store_true',
                        help='Generate and save a figure of the Mean Flux (default: False)')
    parser.add_argument('--plot_target_power', action='store_true',
                        help='Generate and save a figure of the target P1D (default: False)')
    parser.add_argument('--plot_target_xif', action='store_true',
                        help='Generate and save a figure of the target Xi_F (default: False)')
    parser.add_argument('--plot_xig_fit', action='store_true',
                        help='Generate and save a figure of the Xi_G fit (default: False)')
    parser.add_argument('--plot_xif_fit', action='store_true',
                        help='Generate and save a figure of the Xi_F fit (default: False)')
    parser.add_argument('--plot_xif_recovered', action='store_true',
                        help='Generate and save a figure of the Xi_F, recovered from Xi_G best fit (default: False)')
    parser.add_argument('--plot_recovered_power', action='store_true',
                        help='Generate and save a figure of the P1D, recovered from Xi_F best fit (default: False)')

    args = parser.parse_args()

    # Pivot redshift for normalization in optical depth fits
    z0 = PD13_PIVOT_Z

    ### Process Input Data ###
    # Parse target redshift(s) either from file or single float input
    try:
        z_target = parse_redshift_target(args.z_target)
    except Exception as e:
        print(f"Error reading redshift file: {e}")
        return

    # Read optional power spectrum input data (k, P1D, velocity spacing arrays)
    k_array, P1D_array, dv_array, numvpoints_array = process_power_data(
        z_target, args.power_file)

    # Read optional mean flux data (z, flux, model name)
    flux_z_array, flux_array, flux_model = process_flux_data(args.flux_file)

    ### FIT MEAN FLUX  ###
    # Perform parameter fitting for mean flux model (tau0, tau1, nu, sigma2)
    tau0, tau1, nu, sigma2 = fit_mean_flux(flux_array, flux_z_array, z0)

    ###  FIT POWER SPECTRUM  ###
    print("\n\n###  Fitting Power  ###")

    for z in z_target:
        # Find index of current redshift for array indexing
        idx = np.where(z_target == z)[0]
        redshift_index = idx[0]
        safe_z = str(z_target[redshift_index]).replace(
            '.', '-')  # For naming figures

        dv = dv_array[redshift_index]
        numvpoints = numvpoints_array[redshift_index]

        print(f"\nProcessing redshift: {z}")
        print(
            f'Mean Flux (z = {z}): {lognMeanFluxGH(z, tau0, tau1, nu, sigma2, z0)[0]}')

        # Scale and select power spectrum input data at current redshift
        k_array_input, p1d_input = scale_input_power(
            redshift_index, k_array, P1D_array, args.power_file)

        # Interpolate power spectrum for FFT with high resolution
        interp_size = 2**20
        new_v_array, k_array_fine, p1d_fine = interpolate_power_for_fft(
            interp_size, default_dv, k_array_input, p1d_input, log_interp=True)
        # False for linear

        # Calculate target xi_F by inverse FFT of interpolated power
        xif_interp_fit = (np.fft.irfft(p1d_fine))[:interp_size] / dv

        # Downsample xi_F for fitting (logarithmic spacing)
        # downsample_size = 2**10   # higher resolution
        downsample_size = 2**6   # lower resolution, for testing
        v_array_downsampled, xif_target_downsampled, dv_downsampled = downsample_array(
            new_v_array, xif_interp_fit, downsample_size, log_scale=True)

        # Solve for optimized Gaussian correlation functions xi_G and xi_F
        v_fit, xi_g_optimized, xi_f_optimized, xi_g_full, zero_point = solve_xi_optimized(z_target,
                                                                                          redshift_index,
                                                                                          downsample_size,
                                                                                          v_array_downsampled,
                                                                                          xif_target_downsampled,
                                                                                          tau0, tau1, nu,
                                                                                          sigma2, z0)

        # Save xi_G half correlation function for later use
        save_xiG_half(v_array_downsampled, xi_g_full, safe_z)

        # Mirror xi_G and export full correlation function
        v_mirrored, xiG_mirrored = mirror_xiG(safe_z, save_cf='full')

        # Save power spectrum from xi_G
        save_PG(safe_z)

        # Recover xi_F from xi_G via lognormal transformation (pointwise)
        xi_f_optimized_extrapolated = np.array([lognXiFfromXiG_pointwise(z_target[redshift_index],
                                                xi_g_i, tau0, tau1, nu, sigma2, z0)
                                                for xi_g_i in xi_g_full])

        # Interpolate recovered xi_F linearly onto fine velocity grid
        v_fine = np.linspace(v_array_downsampled.min(),
                             v_array_downsampled.max(), interp_size)
        cs = CubicSpline(v_array_downsampled, xi_f_optimized_extrapolated)
        xif_fine = cs(v_fine)

        # Recover P_F (k) from solved xi_F
        fit_k_arr, fit_power = recover_power(
            k_array_fine, xif_fine, v_fine, cf_size='half')

        # Set windows for plotting
        # wk_2023 = (k > 0.000750) & (k < 0.035) # EDR measurement
        # w_k = (k_array_input > 1e-4) & (k_array_input < 0.05)
        # w_fit_k = (fit_k_arr > 1e-4) & (fit_k_arr < 0.05)
        # w_k = (k_array_input > 0.000750) & (k_array_input < 0.045)
        # w_fit_k = (fit_k_arr > 0.000750) & (fit_k_arr < 0.045)

        wk_display = (k_array_input > 1e-4) & (k_array_input < 0.05)
        wk_fit_display = (fit_k_arr > 1e-4) & (fit_k_arr < 0.05)

        # Estimate errors on P1D measurement for residuals (simple model)
        p1d_precision = 1e-1
        ptrue = p1d_input[wk_display]
        e_p1d = p1d_precision * ptrue + 1e-8

        # Compute fractional residuals between input power and recovered fit
        fit_power_interp_2 = np.interp(k_array_input, fit_k_arr, fit_power)
        delta_P = np.real(
            (p1d_input.real - fit_power_interp_2.real) / p1d_input.real)
        delta_P_real = delta_P.real

        ### SAVE PLOTS ##
        print("\n###  Saving Figures  ###\n")
        if args.plot_mean_flux:
            plot_mean_flux(flux_z_array, flux_array,
                           tau0, tau1, nu, sigma2, flux_model)

        if args.plot_target_power:
            plot_target_power(safe_z, k_array_input,
                              p1d_input, k_array_fine, p1d_fine)

        if args.plot_target_xif:
            plot_target_xif(safe_z, new_v_array, xif_interp_fit,
                            v_array_downsampled, xif_target_downsampled, dv)

        if args.plot_xif_fit:
            plot_xif_fit(safe_z,  v_array_downsampled,
                         xif_target_downsampled, xi_f_optimized, dv)

        if args.plot_xig_fit:
            plot_xig_fit(safe_z, v_fit, xi_g_optimized,
                         zero_point, v_array_downsampled, xi_g_full)
            # , new_points_only)

        if args.plot_xif_recovered:
            plot_xi_f_recovered(safe_z, v_fine, xif_fine, v_array_downsampled,
                                xif_target_downsampled, v_array_downsampled,
                                xi_f_optimized_extrapolated, dv)

        if args.plot_recovered_power:
            plot_recovered_power(safe_z, k_array_input, p1d_input, wk_display,
                                 fit_k_arr, fit_power, wk_fit_display, e_p1d,
                                 redshift_index, delta_P_real, z)


if __name__ == "__main__":
    main()
