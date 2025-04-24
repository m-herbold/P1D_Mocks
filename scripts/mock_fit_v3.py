#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from iminuit import Minuit
from scipy.optimize import least_squares
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
plt.rcParams['savefig.bbox'] = 'tight'


#######################################


DESI_EDR_PARAMETERS = (
    7.63089e-02, -2.52054e+00, -1.27968e-01,
    3.67469e+00, 2.85951e-01, 7.33473e+02)

# set pivot points for k and z using fiducial power estimate
PD13_PIVOT_K = 0.009  # note: k_0
PD13_PIVOT_Z = 3.0    # note: z0 = 3

gh_degree = 25
gausshermite_xi_deg, gausshermite_wi_deg = np.polynomial.hermite.hermgauss(
    int(gh_degree))
YY1, YY2 = np.meshgrid(gausshermite_xi_deg, gausshermite_xi_deg, indexing='ij')
WW1, WW2 = np.meshgrid(gausshermite_wi_deg, gausshermite_wi_deg, indexing='ij')

# set defaults for p1d fit
default_numvpoints = 2**20
default_dv = 12.0 # safeguard
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

    Returns:python
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
        tau0 (float, optional): Normalization factor for optical depth (default: 0.55).
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
        nu (float): Exponent controlling redshift evolution of the lognormal transform.
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
        nu (float): Exponent controlling redshift evolution of the lognormal transform.
        sigma2 (float): Variance of the Gaussian field.
        z0 (float, optional): Pivot redshift for normalization
                            (default: PD13_PIVOT_Z).

    Returns:
        float or np.ndarray: The computed flux at the given redshift(s).
    """
    e1 = np.exp(2 * a2_z(z, nu / 2, z0) * np.sqrt(2 * sigma2) * delta_g)
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
        dict: A dictionary with redshift as keys and corresponding lists of
            `k` and `P1D` values as values. The dictionary structure is
            {z: {'k': [k1, k2, ...], 'P1D': [P1D1, P1D2, ...]}}.
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

    # Determine dv
    # k_min, k_max = np.min(k_array), np.max(k_array)

    # Check if k-array is uniformly spaced
    dk_values = np.diff(k_array)
    if np.allclose(dk_values, dk_values[0]):
        dk = dk_values[0]
        dv = np.pi / (dk * numvpoints)
    else:
        # More general case: Use range of k values
        print('k-array is not evenly spaced')

    return dv, numvpoints


def process_power_data(z_target, power_file=None):
    """
    Processes the input power spectrum data and computes velocity properties
    for the target redshift(s).

    Args:
        z_target (array-like): A list or array of target redshifts for which
                            power spectrum data should be processed.
        power_file (str, optional): Path to a power spectrum file containing
                                    columns for redshift (z), wave number (k),
                                    and power spectrum values (P1D). If `None`,
                                    a default model (PD13Lorentz_DESI_EDR) is used.

    Returns:
        k_array (np.ndarray): A 1D array of wave numbers for each redshift in `z_target`.
        P1D_array (np.ndarray): A 1D array of power spectrum values (P1D)
                                corresponding to `k_array` for each redshift
                                in `z_target`.
        dv_array (np.ndarray): A 1D array of velocity spacings computed
                                for each redshift.
        numvpoints_array (np.ndarray): A 1D array of the number of velocity
                                points for each redshift.

    Raises:
        SystemExit: If no data is found for any target redshift in the power
                    spectrum file or if there is a mismatch between the k and
                    P1D array lengths.
        ValueError: If there is an error processing the power spectrum file.
    """
    # sort minimum v by redshift (lower redshift requires larger minimum dv 
    # to ensure sigma2 > \xi_G(v=0)
    def get_dv_for_redshift(z):
        if z <= 2.0:
            return 12
        elif 2.1 <= z <= 2.5:
            return 10
        elif z >= 2.6:
            return 1
        else:
            return default_dv  # fallback, just in case
    
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

        # Convert lists to arrays
        P1D_array = np.array(P1D_array, dtype=object)
        k_array = np.array(k_array, dtype=object)
        dv_array = np.array(dv_array)
        numvpoints_array = np.array(numvpoints_array)

    else:
        print("No (P,k,z) file provided, using default model (PD13Lorentz_DESI_EDR)")
        k_array = k_arr
        zlist = z_target
        P1D_array = PD13Lorentz_DESI_EDR(z_target)

        dv_array = np.array([get_dv_for_redshift(z) for z in zlist])
        numvpoints_array = np.array([
            int(np.round(2 * np.pi / (k_array[1] - k_array[0]) / dv)) if len(k_array) > 1 else default_numvpoints
            for dv in dv_array])

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
    Scales the input P1D data to match the default PD13 model behavior for a given z.

    Args:
        redshift_index (int): Index corresponding to the target redshift in
                                the P1D data.
        k_array (np.ndarray): Array of wave numbers (k) for each redshift.
        P1D_array (np.ndarray): Array of power spectrum values (P1D)
                                for each redshift.
        power_file (str, optional): Path to a user-provided P1D file.
                                If not provided, the function
                                uses the default PD13 model data.

    Returns:
        k_array_input (np.ndarray): Array of k-values corresponding to the
                                    selected redshift index.
        p1d_input (np.ndarray): Array of power spectrum values (P1D)
                                corresponding to the selected redshift index.
    """
    if power_file:
        k_array_input = k_array[redshift_index]
        k_array_input = np.array(k_array_input, dtype=float)
        p1d_input = P1D_array[redshift_index]
    else:
        k_array_input = k_array[1:]
        p1d_input = P1D_array[redshift_index][1:]
    return k_array_input, p1d_input


def process_flux_data(flux_file=None):
    """
    Processes the mean flux data, either from a user-provided file or from
    the default model.

    Args:
        flux_file (str, optional): Path to a user-provided flux data file.
                                    The file must have two columns: redshift (z)
                                    and mean flux (F).

    Returns:
        flux_z_array (np.ndarray): Array of redshift values corresponding to
                                    the flux data.
        flux_array (np.ndarray): Array of flux values corresponding to the
                                    redshift values.
        flux_model (str): The source of the flux data, either 'User Input'
                                    (if a file was provided) or 'Turner et al., 2024'
                                    (if the default model is used).

    Raises:
        ValueError: If the number of elements in the redshift array
                    does not match the number of flux values.
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
            "but flux array has {len(flux_array)} elements.")
        return

    print("Flux data successfully assigned.")
    return flux_z_array, flux_array, flux_model


def fit_mean_flux(flux_array, flux_z_array, z0=PD13_PIVOT_Z):
    """
    Fits the mean flux parameters to the given flux and redshift values.

    Args:
        flux_array (np.ndarray): Array of flux values corresponding to the
                                redshift array.
        flux_z_array (np.ndarray): Array of redshift values (z) corresponding
                                to the flux array.
        z0 (float, optional): Pivot redshift for normalization
                                (default: PD13_PIVOT_Z).

    Returns:
        tuple: A tuple containing the optimal values for the flux model parameters:
            - tau0 (float): The best-fit value for the optical depth parameter.
            - tau1 (float): The best-fit value for the power-law exponent parameter.
            - nu (float): The best-fit value for the spectral index.
            - sigma2 (float): The best-fit value for the variance parameter.
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
    Calculates the flux correlation function from the Gaussian correlation function.

    Args:
        z (float): Redshift value.
        xi_gauss (float): Gaussian correlation function value.
        tau0 (float):  Normalization factor for optical depth.
        tau1 (float): Exponent controlling the redshift evolution of optical depth.
        nu (float): Exponent controlling redshift evolution of the lognormal transform.
        sigma2 (float): Variance of the Gaussian field.
        z0 (float, optional): Pivot redshift for normalization
                            (default: PD13_PIVOT_Z).

    Returns:
        float: Flux correlation function corresponding to the input xi_gauss.
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
    Computes the residuals between the target and calculated flux correlation functions.

    Args:
        xi_g (array-like): Current guess for the Gaussian correlation function values.
        z (float): Redshift value at which xi_f is calculated.
        xi_f_target (array-like): Target flux correlation function values.
        tau0 (float): The normalization factor for optical depth.
        tau1 (float): The exponent controlling the redshift evolution of optical depth.
        nu (float): Exponent controlling the redshift evolution of the lognormal transform.
        sigma2 (float): Variance of the Gaussian field.
        z0 (float, optional): Pivot redshift for normalization (default: PD13_PIVOT_Z).

    Returns:
        array-like: Residuals (difference) between the calculated and target
                    flux correlation function values.
    """
    xi_f_calculated = np.array([lognXiFfromXiG_pointwise(z, xi_g_i, tau0,
                                                         tau1, nu, sigma2,
                                                         z0) for xi_g_i in xi_g])
    return xi_f_calculated - xi_f_target


def solve_xi_optimized(z_target, redshift_index, size, xi_f_target,
                       tau0, tau1, nu, sigma2, z0=PD13_PIVOT_Z):
    """
    Solves for the optimized xi_G and xi_F values.

    This function sovles for xi_G and xi_F optimized by minimizing the difference
    between the calculated xi_F and the target xi_F using least squares.

    Args:
        z_target (array-like): Array of redshift values for which to calculate xi_f.
        redshift_index (int): Index of the redshift value in the target array
                                for a given calculation.
        size (int): Number of points for the optimization.
        xi_f_target (array-like): Target flux correlation function (xi_f) values.
        tau0 (float): Normalization factor for optical depth.
        tau1 (float): Exponent controlling the redshift evolution of optical depth.
        nu (float): Exponent controlling the redshift evolution of the
                    lognormal transform.
        sigma2 (float): Variance of the Gaussian field.
        z0 (float, optional): Pivot redshift for normalization
                    (default: PD13_PIVOT_Z).

    Returns:
        tuple: A tuple containing:
            - xi_g_optimized (np.ndarray): Optimized xi_g values corresponding
                                            to the target xi_f.
            - xi_f_optimized (np.ndarray): Optimized flux correlation
                                            function (xi_f) values.
    """
    print('\n(Fitting xi_f)')
    print(f"N-Points:     {size}")

    time_1 = time.strftime("%H:%M:%S")
    print("Start Time:  ", time_1)
    start_time = time.time()

    xi_g_initial_guess = np.full(xi_f_target.size, 0.1)

    result = least_squares(objective, xi_g_initial_guess,
                           args=(z_target[redshift_index],
                                 xi_f_target, tau0, tau1, nu, sigma2, z0))
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
    print(
        f"Difference:                     {np.abs(sigma2-xi_g_optimized[0])}")

    xi_f_optimized = np.array([lognXiFfromXiG_pointwise(z_target[redshift_index],
                                                        xi_g_i, sigma2, tau0,
                                                        tau1, nu, z0)
                               for xi_g_i in xi_g_optimized])
    return xi_g_optimized, xi_f_optimized


def extrapolate_xiG(v_array, xi_G, safe_z, save_cf):
    """
    Extrapolates the xi_G values to v = 0 and optionally saves the extended data.

    Args:
        v_array (np.ndarray): Array of velocity grid points.
        xi_G (np.ndarray): Array of xi_G values corresponding to the velocity grid.
        safe_z (str): Identifier for the redshift to be used in the output
                        file name.
        save_cf (str): Option to save the extrapolated data, can be 'half'
                        to save or an empty string to skip.

    Returns:
        tuple: A tuple containing:
            - v_extrapolated (np.ndarray): Extended velocity grid, including
                                            zero velocity.
            - xi_g_extrapolated (np.ndarray): Extrapolated xi_G values, including
                                            the value at zero velocity.
            - zero_point (float): Extrapolated value of xi_G at zero velocity.
            - new_points_only (np.ndarray): Array of shape (N_new, 2), containing
                                            only the new (v, xi_G) points.
    """
    # Extrapolate xi_G to zero
    linear_extrapolation = interp1d(
        v_array, xi_G, kind='linear', fill_value="extrapolate")
    zero_point = linear_extrapolation(0)
    print(f"zero_point: {zero_point}\n")

    # Extend to include v = 0 and higher resolution sampling
    v_extrapolated = np.logspace(np.log10(1), np.log10(max(v_array)), num=500)
    v_extrapolated = np.insert(v_extrapolated, 0, 0)

    # Build spline from original + zero
    xi_g_new_vals = np.insert(xi_G, 0, zero_point)
    v_extended = np.insert(v_array, 0, 0)
    spline = CubicSpline(v_extended, xi_g_new_vals, bc_type='natural')
    xi_g_extrapolated = spline(v_extrapolated)

    # # Save to file if requested
    # if save_cf == 'half':
    #     export_data_half = np.column_stack((v_extrapolated, xi_g_extrapolated))
    #     np.savetxt(rf'{safe_z}_xiG_half_output.txt', export_data_half, fmt="%.6f",
    #                delimiter="\t", header="Velocity\tXi_G_fit")

    # Define the path to the output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'CF')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Save to file if requested
    if save_cf == 'half':
        export_data_half = np.column_stack((v_extrapolated, xi_g_extrapolated))
        output_path = os.path.join(output_dir, f'{safe_z}_xiG_half_output.txt')
        np.savetxt(output_path, export_data_half, fmt="%.6f",
                   delimiter="\t", header="Velocity\tXi_G_fit")

    # Extract only new points (not in original array)
    # Include v = 0 and v > max(original)
    new_mask = (v_extrapolated == 0) | (v_extrapolated > v_array.max())
    new_points_only = np.column_stack(
        (v_extrapolated[new_mask], xi_g_extrapolated[new_mask]))

    return v_extrapolated, xi_g_extrapolated, zero_point, new_points_only


def mirror_xiG(v_extrapolated, xi_g_extrapolated, safe_z, save_cf):
    """
    Mirrors the xi_G correlation function to create the full correlation function.

    Args:
        v_extrapolated (np.ndarray): Array of extended velocity grid points
                                    including v = 0.
        xi_g_extrapolated (np.ndarray): Array of extrapolated xi_G values
                                        corresponding to the extended velocity grid.
        safe_z (str): Identifier for the redshift, used in naming the output file.
        save_cf (str): Option to save the mirrored data, can be 'full' to save
                                    or an empty string to skip.

    Returns:
        tuple: A tuple containing:
            - v_extended (np.ndarray): Extended velocity grid, mirrored
                                        about zero.
            - file_xig_extended (np.ndarray): Full xi_G correlation function,
                                        mirrored about zero.
    """
    # Set the output directory (../CFs relative to this script)
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'CF')
    os.makedirs(output_dir, exist_ok=True)

    # Load the half CF file
    half_cf_path = os.path.join(output_dir, f'{safe_z}_xiG_half_output.txt')
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

    v_truncated, xiG_truncated = truncate_trailing_zeros(file_v, file_xiG)

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
            output_dir, f'{safe_z}_xiG_full_output.txt')
        np.savetxt(full_cf_path, export_data_full, fmt="%.6f",
                   delimiter="\t", header="Velocity\tXi_G_fit")

    return v_extended, file_xig_extended


def save_PG(safe_z):
    """
    Loads the full xi_G correlation function, computes the power spectrum,
    and saves the result to the P1D directory.

    Args:
        safe_z (str): Identifier for the redshift, used in naming files.
    """
    # Define paths relative to this script
    base_dir = os.path.dirname(__file__)
    cf_dir = os.path.join(base_dir, '..', 'CF')
    p1d_dir = os.path.join(base_dir, '..', 'P_G')
    os.makedirs(p1d_dir, exist_ok=True)

    # Load full CF data
    path_to_cf = os.path.join(cf_dir, f'{safe_z}_xiG_full_output.txt')
    data = np.loadtxt(path_to_cf)
    full_v = data[:, 0]
    full_xiG = data[:, 1]
    full_v_spacing = full_v[1] - full_v[0]

    # Compute power spectrum
    file_xig_power = np.fft.rfft(full_xiG) * full_v_spacing
    file_xig_karr = 2 * np.pi * \
        np.fft.rfftfreq(len(full_xiG), d=full_v_spacing)

    # Save power spectrum
    export_filename = os.path.join(p1d_dir, f'P_G-{safe_z}.txt')
    np.savetxt(export_filename, np.column_stack((file_xig_karr, file_xig_power.real)),
               header='k [km/s]^-1\tP(k)', fmt='%.6e')


def recover_power(k_arr, xi, v_arr, cf_size):
    """
    Recovers the P1D from the correlation function using FFT.

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
    """
    dv = np.mean(np.diff(v_arr))

    if cf_size == 'half':  # mirror first, assume half cf
        v_mirrored = np.concatenate([v_arr, v_arr + v_arr[-1] + dv])
        xi_mirrored = np.concatenate([xi, xi[::-1]])
        mirrored_fit_power = np.fft.rfft(xi_mirrored) * dv
        mirrored_fit_k_arr = 2 * np.pi * \
            np.fft.rfftfreq(len(xi_mirrored/2), d=dv)

        return mirrored_fit_k_arr, mirrored_fit_power

    elif cf_size == 'full':  # do not mirror, assume full cf already
        fit_power = np.fft.rfft(xi) * dv * 2
        fit_k_arr = 2 * np.pi * np.fft.rfftfreq(len(xi), d=dv)

        return fit_k_arr, fit_power

    else:
        pass


def interpolate_arrays(new_size, v_spacing, k_data, p_data):
    """
    Interpolates the P1D onto a new velocity and k grid with higher resolution.

    Args:
        new_size (int): The size of the new grid.
        v_spacing (float): The spacing between velocity points.
        k_data (np.ndarray): Array of k-values to interpolate.
        p_data (np.ndarray): Array of power values to interpolate.

    Returns:
        tuple: A tuple containing:
            - new_v_data (np.ndarray): New velocity grid.
            - new_k_data (np.ndarray): New k grid.
            - new_p_data (np.ndarray): Interpolated power data.
    """
    new_v_data = np.arange(new_size) * v_spacing
    new_k_data = np.linspace(k_data.min(), k_data.max(), new_size)

    cs = CubicSpline(np.ravel(k_data), np.ravel(p_data))
    new_p_data = cs(new_k_data)

    return new_v_data, new_k_data, new_p_data


def downsample_array(v_array, xi_array, downsample_size, log_scale=True):
    """
    Downsamples the given velocity and correlation function arrays.

    Args:
        v_array (np.ndarray): Array of velocity values.
        xi_array (np.ndarray): Array of correlation function values.
        downsample_size (int): The desired size of the downsampled array.
        log_scale (bool, optional): Whether to use a logarithmic scale for
                                sampling the velocity values (default is True).

    Returns:
        tuple: A tuple containing:
            - v_array_downsampled (np.ndarray): Downsampled velocity array.
            - xif_downsampled (np.ndarray): Downsampled correlation function array.
            - dv_downsampled (np.ndarray): Velocity spacing for the downsampled array.
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
    Plots the mean flux fit compared to the target, with a subplot for % difference.

    Args:
        z (np.ndarray): Array of redshift values.
        target_flux (np.ndarray): Array of target flux values to compare against.
        tau0 (float): Normalization factor for optical depth.
        tau1 (float): Exponent controlling the redshift evolution of optical depth.
        nu (float): Exponent controlling redshift evolution of the lognormal transform.
        sigma2 (float): Variance of the Gaussian field.
        flux_model (str): The model name used for the target flux.
        z0 (float, optional): Pivot redshift for normalization (default: PD13_PIVOT_Z).

    Saves:
        Mean_Flux_Fit.png: A plot of the best-fit mean flux and target flux.
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
    ax1.plot(z, best_fit_flux, color='tab:blue',
             lw=6, alpha=0.5, label='Best Fit')
    ax1.plot(z, target_flux, color='black',
             ls='--', label=f'Model: {flux_model}')
    ax1.set_ylabel(r'$\bar F(z)$')
    ax1.legend(loc='upper right')
    ax1.grid()

    # Add parameter text
    txt = f'tau0 = {tau0}\ntau1 = {tau1}\nnu = {nu}\nsigma2 = {sigma2}'
    ax1.text(0.02, 0.1, txt, ha='left', va='center',
             fontsize=12, transform=ax1.transAxes)

    # Bottom panel: percent difference
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.plot(z, percent_diff, color='darkred')
    ax2.set_xlabel('z')
    ax2.set_ylabel('% Diff')
    ax2.grid()

    # Save figure
    plt.savefig('Mean_Flux_Fit.png')


def plot_target_power(z, k_array_input, p1d_input, k_array_fine, p1d_fine):
    """
    Plots the target 1D power spectrum and its interpolated version.

    Args:
        z (float): Redshift value for the target power spectrum.
        k_array_input (np.ndarray): Array of k-values for the input
                                    power spectrum.
        p1d_input (np.ndarray): Array of input power spectrum values.
        k_array_fine (np.ndarray): Array of k-values for the interpolated
                                    power spectrum.
        p1d_fine (np.ndarray): Array of interpolated power spectrum values.

    Saves:
        {z}_P1D_target.png: A plot comparing the input and interpolated
                            power spectra.
    """
    print(rf'Saving: {z}_P1D_target.png')
    plt.figure()

    plt.loglog(k_array_input, p1d_input, alpha=0.7,
               label=f'input P1D, z={z}, N={k_array_input.size}')
    plt.loglog(k_array_fine, p1d_fine, color='tab:orange',
               label=f'interpolated, z={z}, N={k_array_fine.size}', ls='--')
    plt.legend()
    plt.xlabel(r'k $[km/s]^{-1}$')
    plt.ylabel(r'$P_{1D}$ ')
    plt.savefig(rf'{z}_P1D_target.png')


def plot_target_xif(z, new_v_array, xif_interp_fit, v_array_downsampled,
                    xif_target_downsampled, dv):
    """
    Plots the target and interpolated target xi_F correlation functions.

    Args:
        z (float): Redshift value for the target xi_F.
        new_v_array (np.ndarray): Array of velocities for the interpolated xi_F.
        xif_interp_fit (np.ndarray): Interpolated xi_F values.
        v_array_downsampled (np.ndarray): Array of downsampled velocities.
        xif_target_downsampled (np.ndarray): Array of downsampled target xi_F values.
        dv (float): Velocity spacing used for the plot.

    Saves:
        {z}_xi_F_target.png: A plot comparing the interpolated and downsampled xi_F.
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
    plt.savefig(rf'{z}_xi_F_target.png')


def plot_xif_fit(z, v_array_downsampled, xi_f_target, xi_f_optimized, dv):
    """
    Plots the target and optimized xi_F fits along with the residuals.

    Args:
        z (float): Redshift value for the xi_F fit.
        v_array_downsampled (np.ndarray): Array of downsampled velocities.
        xi_f_target (np.ndarray): Target xi_F values.
        xi_f_optimized (np.ndarray): Optimized xi_F values.
        dv (float): Velocity spacing used for the plot.

    Saves:
        {z}_xi_F_fit.png: A plot comparing the target and optimized xi_F.
        {z}_xi_F_fit_residual.png: A plot of the residuals between target
                                    and optimized xi_F.
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
    plt.savefig(rf'{z}_xi_F_fit_residual.png')


def plot_xig_fit(z, v_array_downsampled, xi_g_optimized,
                 zero_point, v_extrapolated, xi_g_extrapolated, new_points_only):
    """
    Plots the optimized xi_g fit along with the extrapolated and fixed point values.

    Args:
        z (float): Redshift value for the xi_g fit.
        v_array_downsampled (np.ndarray): Array of downsampled velocities.
        xi_g_optimized (np.ndarray): Optimized xi_g values.
        zero_point (float): Value of xi_g at zero velocity.
        v_extrapolated (np.ndarray): Extrapolated velocity values.
        xi_g_extrapolated (np.ndarray): Extrapolated xi_g values.

    Saves:
        {z}_xi_G_fit.png: A plot comparing the optimized, extrapolated,
                            and fixed xi_g values.
    """
    print(rf'Saving: {z}_xi_G_fit.png')
    plt.figure()

    plt.plot(v_array_downsampled, xi_g_optimized,
             'o', label=r'$\xi_g$ Fit', color='tab:blue')

    plt.plot(0, zero_point, 'ro', label=f'Fixed Point (0, {zero_point:.3f})')

    plt.plot(v_extrapolated, xi_g_extrapolated, '-',
             label='CS Extrapolation', color='tab:orange')

    # plt.plot(new_points_only[:, 0], new_points_only[:, 1],
    #          'o', markerfacecolor='none', markeredgecolor='tab:orange',
    #          markersize=6)  # , label='Extrapolated Points')

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
    """
    Plots the recovered xi_F fit along with the
    downsampled target and extrapolated values.

    Args:
        z (float): Redshift value for the xi_F recovery plot.
        v_fine (np.ndarray): Fine velocity array for re-interpolated xi_F.
        xif_fine (np.ndarray): Re-interpolated xi_F values.
        v_array_downsampled (np.ndarray): Array of downsampled velocities.
        xif_target_downsampled (np.ndarray): Downsampled target xi_F values.
        v_extrapolated (np.ndarray): Extrapolated velocity values.
        xi_f_optimized_extrapolated (np.ndarray): Extrapolated optimized xi_F.
        dv (float): Velocity spacing used for the plot.

    Saves:
        {z}_xi_F_recovery.png: A plot comparing the re-interpolated,
                            downsampled target, and extrapolated xi_F values.
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
                         mirrored_fit_power, w_fit_k, e_p1d, z_id, delta_P_real):
    """
    Plots the recovered power spectrum and percent difference residuals.

    Args:
        z (float): Redshift value for the power spectrum plot.
        k_array_input (np.ndarray): Array of k values (momentum space).
        p1d_input (np.ndarray): Power spectrum values corresponding to
                                k_array_input.
        w_k (np.ndarray): Index array used to filter k_array_input and p1d_input.
        mirrored_fit_k_arr (np.ndarray): Fitted k values after mirroring the
                                power spectrum.
        mirrored_fit_power (np.ndarray): Fitted power values corresponding to
                                mirrored_fit_k_arr.
        w_fit_k (np.ndarray): Index array for selecting the fitted power spectrum.
        e_p1d (np.ndarray): Error values for the power spectrum.
        z_id (int): Index for selecting the redshift value in e_p1d.
        delta_P_real (np.ndarray): Residuals of the power spectrum
                                (difference between model and fit divided by model).

    Saves:
        {z}_recovered_power.png: A plot of the power spectrum
                                (model vs fit) and the percent residuals.
    """
    print(rf'Saving: {z}_recovered_power.png')

    temp_k = k_array_input[w_k]
    temp_p = p1d_input[w_k]
    temp_e = np.full_like(temp_k, e_p1d[z_id])

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    # Top subplot: Power spectrum
    ax1 = fig.add_subplot(gs[0])
    ax1.axvspan(0.05, 0.1, alpha=0.2, color='grey')

    alpha_shade = 0.3
    # Plot shaded error band instead of individual error bars
    ax1.fill_between(temp_k, temp_p - temp_e, temp_p + temp_e,
                     color='tab:blue', alpha=alpha_shade, label=' ± precision')
    ax1.loglog(k_array_input[w_k], p1d_input[w_k].real,
               color='tab:blue', label='Model')
    ax1.loglog(mirrored_fit_k_arr[w_fit_k], mirrored_fit_power[w_fit_k].real,
               color='tab:orange', ls='--', label='Best Fit')
    ax1.set_ylim(1e-1, mirrored_fit_power[w_fit_k].real.max() + 100)
    ax1.set_ylabel(rf'$P(k)$   (z = {z})')
    ax1.legend(loc='lower left')

    # Bottom subplot: Percent residuals
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    percent_diff = 100 * delta_P_real  # Convert to percent
    ax2.semilogx(k_array_input, percent_diff, color='darkred')
    ax2.axvspan(0.05, 1.0, alpha=0.2, color='grey')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)

    # Compute dynamic y-axis range only for values within x_min and x_max
    x_min, x_max = k_array_input[w_k].min(), 0.1
    mask = (k_array_input >= x_min) & (k_array_input <= 0.05)
    percent_diff_in_range = percent_diff[mask]
    
    # Compute max abs percent difference in the plotting range
    if np.any(~np.isnan(percent_diff_in_range)):  # Ensure there are valid values
        y_max = np.nanmax(np.abs(percent_diff_in_range))
        buffer = 0.05 * y_max  # Add some padding
        ax2.set_ylim(-y_max - buffer, y_max + buffer)
    else:
        ax2.set_ylim(-10, 10)  # Fallback in case of NaNs
        
    ax2.set_xlabel(r'k $[km/s]^{-1}$')
    ax2.set_ylabel(r"% Difference")
    ax2.grid()

    # Set shared x limits
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    # Clean x ticks on top plot
    plt.setp(ax1.get_xticklabels(), visible=False)

    # plt.savefig(rf'{z}_recovered_power.png')
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

    z0 = PD13_PIVOT_Z

    ### Process Input Data ###
    # z_target (required)
    try:
        z_target = parse_redshift_target(args.z_target)
    except Exception as e:
        print(f"Error reading redshift file: {e}")
        return
    # power file (optional)
    k_array, P1D_array, dv_array, numvpoints_array = process_power_data(
        z_target, args.power_file)
    # flux file  (optional)
    flux_z_array, flux_array, flux_model = process_flux_data(args.flux_file)

    ### FIT MEAN FLUX  ###
    tau0, tau1, nu, sigma2 = fit_mean_flux(flux_array, flux_z_array, z0)

    ###  FIT POWER  ###
    print("\n\n###  Fitting Power  ###")

    for z in z_target:
        idx = np.where(z_target == z)[0]
        redshift_index = idx[0]
        safe_z = str(z_target[redshift_index]).replace(
            '.', '-')  # For naming figures

        dv = dv_array[redshift_index]
        print(f'dv = {dv}')
        numvpoints = numvpoints_array[redshift_index]

        print(f"\nProcessing redshift: {z}")
        print(
            f'Mean Flux (z = {z}): {lognMeanFluxGH(z, tau0, tau1, nu, sigma2, z0)[0]}')

        k_array_input, p1d_input = scale_input_power(
            redshift_index, k_array, P1D_array, args.power_file)

        # Interpolate input P(k) to a set size
        interp_size = 2**20
        new_v_array, k_array_fine, p1d_fine = interpolate_arrays(
            interp_size, dv, k_array_input, p1d_input)

        # Calculate target xi_f from target p_f
        xif_interp_fit = (np.fft.irfft(p1d_fine))[:interp_size] / dv

        # Downsample xi_f (logarithmically)
        downsample_size = 2**10
        v_array_downsampled, xif_target_downsampled, dv_downsampled = downsample_array(
            new_v_array, xif_interp_fit, downsample_size, log_scale=True)

        # Solve for xi_G and xi_F optimized
        xi_g_optimized, xi_f_optimized = solve_xi_optimized(z_target,
                                                            redshift_index,
                                                            downsample_size,
                                                            xif_target_downsampled,
                                                            tau0, tau1, nu,
                                                            sigma2, z0)

        # Extrapolate xi_G to zero (saves half and full cf)
        v_extrapolated, xi_g_extrapolated, zero_point, new_points_only = extrapolate_xiG(v_array_downsampled,
                                                                                         xi_g_optimized, safe_z,
                                                                                         save_cf='half')

        # Mirror and export xi_G full
        v_mirrored, xiG_mirrored = mirror_xiG(v_extrapolated, xi_g_extrapolated,
                                              safe_z, save_cf='full')

        # Export P_G
        save_PG(safe_z)

        # Recover xi_f
        xi_f_optimized_extrapolated = np.array([lognXiFfromXiG_pointwise(z_target[redshift_index],
                                                xi_g_i, tau0, tau1, nu, sigma2, z0)
                                                for xi_g_i in xi_g_extrapolated])
        # Interpolate linearly
        v_fine = np.linspace(v_extrapolated.min(),
                             v_extrapolated.max(), interp_size)
        cs = CubicSpline(v_extrapolated, xi_f_optimized_extrapolated)
        xif_fine = cs(v_fine)

        # Recover P_F (k)
        fit_k_arr, fit_power = recover_power(
            k_array_fine, xif_fine, v_fine, cf_size='half')

        # Set windows for plotting
        w_k = (k_array_input > 1e-4) & (k_array_input < 0.05)
        w_fit_k = (fit_k_arr > 1e-4) & (fit_k_arr < 0.05)

        p1d_precision = 1e-1
        ptrue = p1d_input[w_k]
        e_p1d = p1d_precision * ptrue + 1e-8

        # Compute residual
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
            plot_xig_fit(safe_z, v_array_downsampled, xi_g_optimized,
                         zero_point, v_extrapolated, xi_g_extrapolated, new_points_only)

        if args.plot_xif_recovered:
            plot_xi_f_recovered(safe_z, v_fine, xif_fine, v_array_downsampled,
                                xif_target_downsampled, v_extrapolated,
                                xi_f_optimized_extrapolated, dv)

        if args.plot_recovered_power:
            plot_recovered_power(safe_z, k_array_input, p1d_input, w_k,
                                 fit_k_arr, fit_power, w_fit_k, e_p1d,
                                 redshift_index, delta_P_real)


if __name__ == "__main__":
    main()
