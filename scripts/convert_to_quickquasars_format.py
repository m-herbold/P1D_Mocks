#!/usr/bin/env python3
"""
Conversion script to transform transmission files from mock_gen.py
to the format expected by quickquasars.

Handles conversion from velocity-space (km/s) to wavelength-space (Angstroms)
for each individual quasar spectrum at its specific redshift.
"""

import numpy as np
import argparse
import os
from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from scipy.interpolate import interp1d

# Constants
LIGHT_SPEED = 299792.458  # km/s
LYA_WAVELENGTH = 1215.67  # Angstroms (rest-frame Lyman-alpha)


def convert_transmission_to_quickquasars(
    input_files, output_file,
    ra_values, dec_values, redshift_values,
    mockid_start=1000,
    wavelength_min=3600.0,
    wavelength_max=9800.0,
    dwave_out=None  # None = keep original resolution
):
    """
    Convert transmission FITS files from mock_gen.py to quickquasars format.

    Key operations:
    1. Read velocity grid and transmission from each file
    2. Convert velocity → wavelength at each quasar's redshift
    3. Optionally resample all spectra onto a common wavelength grid
    4. Write in minimal format expected by quickquasars

    Parameters
    ----------
    input_files : list of str
        List of transmission file paths from mock_gen.py
    output_file : str
        Output FITS file path
    ra_values, dec_values, redshift_values : array-like
        Sky coordinates and redshifts for each spectrum
    mockid_start : int
        Starting MOCKID value
    wavelength_min, wavelength_max : float
        Wavelength range to include (Angstroms)
    dwave_out : float, optional
        Output wavelength spacing (Angstroms). If None, uses the native
        resolution from the velocity grid.
    """

    n_spectra = len(input_files)
    print(f"Converting {n_spectra} transmission files...")

    # =========================================================================
    # STEP 1: Read first file to understand the velocity grid
    # =========================================================================
    with fits.open(input_files[0]) as hdul:
        trans_table = Table.read(hdul['TRANSMISSION'])
        velocity_grid = np.array(trans_table['VELOCITY'])  # km/s

        # Get velocity spacing
        primary_header = hdul[0].header
        if 'DV' in primary_header:
            dv = primary_header['DV']
        else:
            dv = velocity_grid[1] - velocity_grid[0]

        print(
            f"Velocity grid: {velocity_grid[0]:.1f} to {velocity_grid[-1]:.1f} km/s")
        print(f"Velocity spacing: dv = {dv} km/s")
        print(f"Number of velocity points: {len(velocity_grid)}")

    # =========================================================================
    # STEP 2: Create common output wavelength grid
    # =========================================================================
    if dwave_out is not None:
        # User specified output spacing - create linear grid
        n_wave_out = int((wavelength_max - wavelength_min) / dwave_out) + 1
        wavelength_out = np.linspace(
            wavelength_min, wavelength_max, n_wave_out)
        print(f"\nOutput wavelength grid (resampled):")
        print(f"  Range: {wavelength_out[0]:.2f} - {wavelength_out[-1]:.2f} Å")
        print(f"  Spacing: {dwave_out} Å")
        print(f"  Number of points: {n_wave_out}")
    else:
        # Use native resolution from velocity grid
        # Convert at mean redshift to estimate number of points needed
        z_mean = np.mean(redshift_values)
        wave_center = LYA_WAVELENGTH * (1 + z_mean)
        wavelength_native = wave_center * np.exp(velocity_grid / LIGHT_SPEED)

        # Filter to wavelength range
        mask = (wavelength_native >= wavelength_min) & (
            wavelength_native <= wavelength_max)
        wavelength_out = wavelength_native[mask]
        n_wave_out = len(wavelength_out)

        print(f"\nOutput wavelength grid (native resolution):")
        print(f"  Range: {wavelength_out[0]:.2f} - {wavelength_out[-1]:.2f} Å")
        print(f"  Number of points: {n_wave_out}")
        print(
            f"  Estimated spacing: ~{np.median(np.diff(wavelength_out)):.4f} Å")

    # Initialize output transmission array
    transmission_out = np.zeros((n_spectra, n_wave_out), dtype=np.float32)

    # =========================================================================
    # STEP 3: Process each spectrum individually
    # =========================================================================
    print(f"\nProcessing {n_spectra} spectra...")

    for i, (fname, z) in enumerate(zip(input_files, redshift_values)):
        # Read transmission from file
        with fits.open(fname) as hdul:
            trans_table = Table.read(hdul['TRANSMISSION'])
            transmission_data = np.array(trans_table['FLUX'])

        # Convert velocity → wavelength at THIS quasar's redshift
        # wave = LYA_WAVELENGTH * (1 + z) * exp(v / c)
        wave_center = LYA_WAVELENGTH * (1 + z)
        wavelength_hires = wave_center * np.exp(velocity_grid / LIGHT_SPEED)

        # Check if this spectrum's wavelength range covers our output grid
        if wavelength_hires[-1] < wavelength_min or wavelength_hires[0] > wavelength_max:
            print(
                f"  Warning: Spectrum {i} at z={z:.2f} outside wavelength range")
            # Fill with ones (no absorption) outside range
            transmission_out[i, :] = 1.0
            continue

        # Interpolate onto common wavelength grid
        # Use linear interpolation, fill with 1.0 outside range
        interp_func = interp1d(
            wavelength_hires, transmission_data,
            kind='linear',
            bounds_error=False,
            fill_value=1.0
        )
        transmission_out[i, :] = interp_func(wavelength_out)

        lya_obs = LYA_WAVELENGTH * (1 + z)
        # should this be > or >=?
        transmission_out[i, wavelength_out > lya_obs] = 1.0

        if (i + 1) % 10 == 0 or i == n_spectra - 1:
            print(f"  Processed {i+1}/{n_spectra} spectra")

    # =========================================================================
    # STEP 4: Validate transmission values
    # =========================================================================
    print(f"\nTransmission statistics:")
    print(f"  Shape: {transmission_out.shape}")
    # print(f"  Mean: {transmission_out.mean():.6f}")
    print(
        f"  Range: {transmission_out.min():.6f} - {transmission_out.max():.6f}")

    if transmission_out.min() < 0 or transmission_out.max() > 1.01:
        print(f"  ⚠️  WARNING: Transmission outside [0, 1] range!")

    # =========================================================================
    # STEP 5: Create FITS HDUs (MINIMAL - only what's needed)
    # =========================================================================

    # Primary HDU - completely empty, like Ohio mocks
    primary_hdu = fits.PrimaryHDU()
    # Don't add ANY header keywords - keep it minimal

    # METADATA table - only essential columns
    metadata = Table()
    metadata['RA'] = np.array(ra_values, dtype=np.float64)
    metadata['DEC'] = np.array(dec_values, dtype=np.float64)
    metadata['Z'] = np.array(redshift_values, dtype=np.float64)
    metadata['MOCKID'] = np.arange(
        mockid_start, mockid_start + n_spectra, dtype=np.int64)

    metadata_hdu = fits.table_to_hdu(metadata)
    metadata_hdu.name = 'METADATA'
    # Don't add ANY extra header keywords

    # WAVELENGTH HDU (1D array)
    wavelength_hdu = fits.ImageHDU(wavelength_out.astype(np.float32))
    wavelength_hdu.name = 'WAVELENGTH'

    # TRANSMISSION HDU (2D: n_quasars x n_wavelength)
    transmission_hdu = fits.ImageHDU(transmission_out)
    transmission_hdu.name = 'TRANSMISSION'

    # Assemble and write
    hdul = fits.HDUList(
        [primary_hdu, metadata_hdu, wavelength_hdu, transmission_hdu])

    print(f"\nWriting output to {output_file}")
    hdul.writeto(output_file, overwrite=True)
    print("✓ Done!")

    # Print final structure for verification
    print("\nOutput file structure:")
    print(f"  HDU 0: PRIMARY (empty)")
    print(
        f"  HDU 1: METADATA ({len(metadata)} entries, columns: {list(metadata.colnames)})")
    print(f"  HDU 2: WAVELENGTH (1D array, {len(wavelength_out)} points)")
    print(f"  HDU 3: TRANSMISSION (2D array, shape {transmission_out.shape})")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert mock_gen.py transmission files to quickquasars format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='Input transmission file(s) from mock_gen.py')
    parser.add_argument('--output', type=str, required=True,
                        help='Output FITS file for quickquasars')
    parser.add_argument('--ra', type=float, nargs='+', required=True,
                        help='Right Ascension value(s) [degrees]')
    parser.add_argument('--dec', type=float, nargs='+', required=True,
                        help='Declination value(s) [degrees]')
    parser.add_argument('--redshift', type=float, nargs='+', required=True,
                        help='Redshift value(s)')
    parser.add_argument('--mockid', type=int, default=1000,
                        help='Starting MOCKID value')
    parser.add_argument('--wavelength-min', type=float, default=3600.0,
                        help='Minimum output wavelength [Angstrom]')
    parser.add_argument('--wavelength-max', type=float, default=9800.0,
                        help='Maximum output wavelength [Angstrom]')
    parser.add_argument('--dwave', type=float, default=None,
                        help='Output wavelength spacing [Angstrom]. If not specified, '
                             'keeps native resolution from velocity grid.')

    args = parser.parse_args()

    # Validate inputs
    n_files = len(args.input)

    # Handle single vs multiple values for RA, DEC, Z
    ra_values = [args.ra[0]] * n_files if len(args.ra) == 1 else args.ra
    dec_values = [args.dec[0]] * n_files if len(args.dec) == 1 else args.dec
    redshift_values = [args.redshift[0]] * \
        n_files if len(args.redshift) == 1 else args.redshift

    if len(ra_values) != n_files or len(dec_values) != n_files or len(redshift_values) != n_files:
        raise ValueError(
            f"Number of RA/DEC/redshift values must match number of input files ({n_files})")

    # Convert
    convert_transmission_to_quickquasars(
        args.input, args.output,
        ra_values, dec_values, redshift_values,
        mockid_start=args.mockid,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        dwave_out=args.dwave
    )


if __name__ == "__main__":
    main()
