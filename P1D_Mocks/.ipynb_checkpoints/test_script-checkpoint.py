# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm

import scipy as sp
from scipy.fft import fft, ifft, rfft, irfft
from scipy.stats import binned_statistic
from scipy import integrate

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
# %matplotlib inline

from numpy.polynomial.hermite import hermgauss
import argparse  # Added for argument parsing

###############################################################################

# constants
c = 299792.458  # speed of light in km/s

###############################################################################

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Simulate a Gaussian random field in velocity and wavelength space")

    # User-defined variables with default values
    parser.add_argument('--seed', type=int, default=10, help='Seed for random number generation (default: 10)')
    parser.add_argument('--size', type=int, default=2**20, help='Size of the velocity/wavelength grid (default: 2^20)')
    parser.add_argument('--z', type=float, default=2.8, help='Redshift value (default: 2.8)')
    parser.add_argument('--z0', type=float, default=3.0, help='Reference redshift value (default: 3.0)')
    parser.add_argument('--plot_flux', action='store_true', help='Generate and save flux plot (default: False)')
    parser.add_argument('--plot_pdf_f', action='store_true', help='Generate and save flux PDF plot (default: False)')
    parser.add_argument('--plot_pdf_d', action='store_true', help='Generate and save flux decrement PDF plot (default: False)')

    args = parser.parse_args()
    return args

# Get user inputs or defaults
args = parse_args()

seed = args.seed
size = args.size
z = args.z
z0 = args.z0
flux_plot = args.plot_flux
pdf_f_plot = args.plot_pdf_f
pdf_d_plot = args.plot_pdf_d

###############################################################################

# wavelength grid
def wavelength_field(z, grid_size, lambda_min=3600, lambda_max=9800, lambda_0=1216):
    velocity_grid = np.arange(grid_size) - grid_size / 2
    lambda_c = (1 + z) * lambda_0
    wavelength_field = lambda_c * (np.exp(velocity_grid / c))
    return wavelength_field

# gaussian random grid
def gaussian_field_v(grid_size=size, seed=None, plot=False):
    # Generate Gaussian random field in velocity-space
    gaussian_random_field = np.random.default_rng(seed).normal(size=grid_size)

    if plot:
        plt.plot(gaussian_random_field)
        plt.title('Gaussian Random Field, Velocity Space')
        plt.xlabel('Points')
        plt.ylabel('Velocity (km/s)')
        plt.savefig('gaussian_field_v_plot.png')  # Save the plot
        plt.close()

    return gaussian_random_field

def gaussian_field_k(grid_size=size, seed=None, plot=False):
    # Generate Gaussian random field in velocity-space
    gaussian_random_field = np.random.default_rng(seed).normal(size=grid_size)

    # Compute the FFT to get the field in k-space
    gaussian_field_k = np.fft.rfft(gaussian_random_field)

    if plot:
        plt.plot(np.real(gaussian_field_k))
        plt.title('RFFT(1D field), k-space (Real part)')
        plt.xlabel('k-modes')
        plt.ylabel('Amplitude')
        plt.savefig('gaussian_field_k_plot.png')  # Save the plot
        plt.close()

    return gaussian_field_k

v_field = gaussian_field_v(size, seed)
k_field = gaussian_field_k(size, seed)

###############################################################################

# make grid of k-modes
def make_kmodes(grid_size=size, spacing=1):
    kmodes = (np.fft.rfftfreq(n=grid_size, d=spacing) * 2 * np.pi) + 1e-12  # factor of 2pi
    return kmodes

kmodes = make_kmodes(size, spacing=1)

###############################################################################

# rebin k-modes
def rebin_kmodes(kmodes, array, kmode_spacing=1, stat='mean', bin_num=40):
    # returns an averaged / smoothed array by rebinning kmodes over a number of bins
    statistic, bin_edges, binnumber = binned_statistic(x=kmodes, values=array, statistic=stat, bins=bin_num)
    return statistic

# Measure Power
def measure_power(array):
    power = np.abs(array)**2
    return power

def measure_scaled_power(array, n, length_scale=1):
    # scale power as P(k)/L, where n=number of points or size and L=n*len_scale
    power = np.abs(array)**2
    scaled_power = power / (n * length_scale)
    return scaled_power

# Input Power Spectrum model
def power_spectrum(k, k0=0.001, k1=0.04, n=0.5, alpha=0.26, gamma=1.8, plot=False):
    pk = ((k / k0)**(n - alpha * np.log(k / k0))) / (1 + (k / k1)**gamma)
    pk_i = np.arange(len(pk))

    mask = np.isfinite(pk)
    pk_filtered = np.interp(pk_i, pk_i[mask], pk[mask])

    if plot:
        plt.loglog(k[1:], power_spectrum(k[1:]), color='tab:blue')
        plt.title(r'$P(k) = \frac{(k / k_0)^{n - \alpha \ln(k/k_0)}}{1 + (k/k_1)^{\gamma}}$')
        plt.xlabel('k-modes')
        plt.ylabel('P(k)')
        plt.savefig('power_spectrum_plot.png')  # Save the plot
        plt.close()

    return pk_filtered

# Delta transform
def delta_transform(k_field, kmodes, dv=1, k0=0.001, k1=0.04, n=0.5, alpha=0.26, gamma=1.8):
    delta_b_tilde = k_field * np.sqrt(power_spectrum(kmodes, k0, k1, n, alpha, gamma) / dv)  # multiply k-space field by p(k)
    delta_b = np.fft.irfft(delta_b_tilde)  # inverse rfft back to v-space
    return delta_b

delta_b_v = delta_transform(k_field, kmodes)
variance_1d = delta_b_v.var()  # sigma^2

###############################################################################

# redshift evolution factor
def redshift_evolution(z, z0, a=58.6, b=-2.82):
    a_z = np.sqrt(a * (((1 + z) / (1 + z0))**(b)))
    return a_z

delta_b_z = delta_b_v * redshift_evolution(z, z0)
redshifted_variance_1d = variance_1d * redshift_evolution(z, z0)**2

###############################################################################

# lognormal transformation
def lognormal_transform(delta_z, variance):
    n_z = np.exp((2 * (delta_z) - (variance)))
    return n_z

n_z = lognormal_transform(delta_b_z, redshifted_variance_1d)

###############################################################################

# optical depth transform
def optical_depth_transform(n_z, z=2.8, z0=3, c=0.55, d=5.1):
    tau_z = c * (((1 + z) / (1 + z0))**d) * n_z
    return tau_z

tau_z = optical_depth_transform(n_z, z, z0)

###############################################################################

def flux(tau):
    F_z = np.exp(-(tau))
    return F_z

f_z = flux(tau_z)

# option to save flux plot
def plot_flux(z):
    # Generate the wavelength grid
    lambda_field = wavelength_field(z, size)

    if flux_plot:  # Use global flux_plot directly
        plt.figure(figsize=(12, 6))
        plt.plot(lambda_field[:10000], f_z[:10000], label='z = ' + str(z))
        plt.title(r'$ F(z) = e^{-\tau(z)}$')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')
        plt.hlines(y=f_z[:10000].mean(), xmin=lambda_field[:10000].min(), xmax=lambda_field[:10000].max(), color='black', ls='--', label='Mean Flux')
        plt.legend()
        plt.savefig(f'flux_plot_z_{z}.png')  # Save the plot
        plt.close()

if flux_plot:
    plot_flux(z)

# calculate mean flux using quad integration
def mean_flux(z, variance, z0=3):
    prefactor = 1 / (np.sqrt(variance) * np.sqrt(2 * np.pi))
    x_z = 0.55 * (((1 + z) / (1 + z0))**(5.1)) * np.exp(-((0.55 / variance) * (((1 + z) / (1 + z0))**(5.1))**2))

    # performing integral
    result = sp.integrate.quad(lambda x: np.exp(-(x_z * np.exp(-x))), -np.inf, np.inf)

    return result

# calculate mean flux using Gauss-Hermite quadrature integration
def gh_mean_flux(z, variance, z0=3, n=10):
    # generate n weights and points for n-th order Gauss-Hermite polynomial
    gh_roots, gh_weights = hermgauss(n)
    gh_points = np.sqrt(2 * variance) * gh_roots
    integral_approximation = np.sum([np.exp(-(0.55 * ((1 + z) / (1 + z0))**(5.1) * np.exp(-x))) for x in gh_points] * gh_weights)

    gh_prefactor = 1 / np.sqrt(2 * np.pi * variance)
    integral_approximation = integral_approximation * gh_prefactor

    return integral_approximation

f_bar_gh = gh_mean_flux(z, variance_1d, z0, n=10)

###############################################################################

# PDF of flux and flux decrement
def pdf_d(z, variance, gaussian_k_field, input_kmodes, z0=3, color=None):
    delta_z = delta_transform(k_field=gaussian_k_field, kmodes=input_kmodes, dv=1) * redshift_evolution(z, z0)
    redshifted_variance = variance * redshift_evolution(z, z0)**2
    f_z_vals = flux(optical_depth_transform(lognormal_transform(delta_z, redshifted_variance)))

    if pdf_d_plot:
        plt.hist((1 - f_z_vals), bins=100, alpha=0.5, label='z=' + str(z), color=color, density=True, stacked=True)
        plt.xlabel('Flux Decrement: D = 1 - F(z)')
        plt.ylabel('PDF weight (D)')
        plt.legend()
        plt.savefig(f'pdf_d_plot_z_{z}.png')  # Save the plot
        plt.close()

def pdf_f(z, variance, gaussian_k_field, input_kmodes, z0=3, color=None):
    delta_z = delta_transform(k_field=gaussian_k_field, kmodes=input_kmodes, dv=1) * redshift_evolution(z, z0)
    redshifted_variance = variance * redshift_evolution(z, z0)**2
    f_z_vals = flux(optical_depth_transform(lognormal_transform(delta_z, redshifted_variance)))

    if pdf_f_plot:
        plt.hist((f_z_vals), bins=100, alpha=0.5, label='z=' + str(z), color=color, density=True, stacked=True)
        plt.xlabel('Flux: F(z)')
        plt.ylabel('PDF weight (F)')
        plt.legend()
        plt.savefig(f'pdf_f_plot_z_{z}.png')  # Save the plot
        plt.close()

pdf_d(z, variance_1d, k_field, kmodes, z0)
pdf_f(z, variance_1d, k_field, kmodes, z0)

###############################################################################
