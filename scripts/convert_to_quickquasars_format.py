#!/usr/bin/env python3
"""
Conversion script to transform transmission files from mock_gen.py
to the format expected by quickquasars. 

Usage:  
    python convert_to_quickquasars_format.py --input transmission_2-2_*. fits \
        --output transmission-16-12345. fits --ra 150.0 --dec 2.0 \
        --redshift 2.2 --mockid 1000 --nside 16 --pixel 12345
"""

import numpy as np
import argparse
import os
from astropy.io import fits
from astropy.table import Table
from pathlib import Path


def convert_transmission_to_quickquasars(input_files, output_file, 
                                         ra_values, dec_values, 
                                         redshift_values, mockid_start,
                                         nside=16, pixel=None,
                                         healpix_nest=True):
    """
    Convert transmission FITS files from mock_gen.py to quickquasars format.
    
    Parameters:
    -----------
    input_files : list of str
        List of input transmission file paths (. fits format from mock_gen.py)
    output_file : str
        Output FITS file path
    ra_values : array-like
        Right Ascension values for each spectrum (degrees)
    dec_values : array-like
        Declination values for each spectrum (degrees)
    redshift_values : array-like
        Redshift values for each spectrum
    mockid_start : int
        Starting MOCKID value (will increment for each spectrum)
    nside : int
        HEALPix nside parameter
    pixel : int, optional
        HEALPix pixel number
    healpix_nest : bool
        Whether HEALPix scheme is nested (True) or ring (False)
    """
    
    n_spectra = len(input_files)
    
    # Read first file to get dimensions and check format
    print(f"Reading {n_spectra} transmission files...")
    with fits.open(input_files[0]) as hdul:
        # Read from TRANSMISSION table
        trans_table = Table. read(hdul['TRANSMISSION'])
        velocity_grid = trans_table['VELOCITY']  # km/s
        transmission_first = trans_table['FLUX']
        
        # Get metadata from primary header if available
        primary_header = hdul[0]. header
        if 'DV' in primary_header: 
            dv = primary_header['DV']
            print(f"Found velocity spacing: dv = {dv} km/s")
        if 'LAM0' in primary_header:
            lambda_lya = primary_header['LAM0']
            print(f"Found Lya wavelength: λ_0 = {lambda_lya} Å")
        else:
            lambda_lya = 1215.67  # Default
    
    # Convert velocity to wavelength for each redshift
    c = 299792.458  # km/s
    
    # We'll use the mean redshift to calculate a reference wavelength grid
    # In practice, quickquasars expects a single wavelength grid for all spectra
    z_ref = np.mean(redshift_values)
    lambda_central = lambda_lya * (1 + z_ref)
    wavelength = lambda_central * (1 + velocity_grid / c)
    
    n_wave = len(wavelength)
    
    # Initialize transmission array
    transmission_array = np.zeros((n_spectra, n_wave), dtype=np.float32)
    
    # Read all transmission files
    for i, fname in enumerate(input_files):
        with fits.open(fname) as hdul:
            trans_table = Table.read(hdul['TRANSMISSION'])
            transmission_array[i, :] = trans_table['FLUX']
    
    print(f"Transmission array shape: {transmission_array.shape}")
    print(f"Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} Å")
    print(f"Mean transmission: {transmission_array.mean():.4f}")
    
    # Create METADATA table
    metadata = Table()
    metadata['MOCKID'] = np.arange(mockid_start, mockid_start + n_spectra, dtype=np.int64)
    metadata['RA'] = np.array(ra_values, dtype=np.float64)
    metadata['DEC'] = np. array(dec_values, dtype=np.float64)
    metadata['Z'] = np.array(redshift_values, dtype=np.float64)
    metadata['Z_noRSD'] = np.array(redshift_values, dtype=np.float64)  # same if no RSD applied
    
    # Create HDU list
    primary_hdu = fits.PrimaryHDU()
    
    # Add HEALPix header information
    primary_hdu.header['NSIDE'] = nside
    primary_hdu.header['HPXNSIDE'] = nside
    if pixel is not None:
        primary_hdu.header['PIXNUM'] = pixel
        primary_hdu.header['HPXPIXEL'] = pixel
    primary_hdu.header['HPXNEST'] = healpix_nest
    primary_hdu.header['NESTED'] = healpix_nest
    primary_hdu.header['SCHEME'] = 'NEST' if healpix_nest else 'RING'
    primary_hdu.header['COMMENT'] = 'Converted from mock_gen.py output'
    primary_hdu.header['COMMENT'] = f'Original files: {len(input_files)} transmission files'
    
    # Create METADATA HDU
    metadata_hdu = fits.table_to_hdu(metadata)
    metadata_hdu.name = 'METADATA'
    metadata_hdu.header['HPXNSIDE'] = nside
    metadata_hdu.header['NSIDE'] = nside
    if pixel is not None:
        metadata_hdu.header['HPXPIXEL'] = pixel
        metadata_hdu.header['PIXNUM'] = pixel
    metadata_hdu.header['HPXNEST'] = healpix_nest
    metadata_hdu.header['NESTED'] = healpix_nest
    metadata_hdu.header['SCHEME'] = 'NEST' if healpix_nest else 'RING'
    
    # Create WAVELENGTH HDU (1D array)
    wavelength_hdu = fits.ImageHDU(wavelength.astype(np.float32))
    wavelength_hdu.name = 'WAVELENGTH'
    wavelength_hdu.header['BUNIT'] = 'Angstrom'
    wavelength_hdu.header['COMMENT'] = 'Wavelength array converted from velocity grid'
    
    # Create TRANSMISSION HDU (2D array:  n_quasars x n_wavelengths)
    transmission_hdu = fits.ImageHDU(transmission_array)
    transmission_hdu.name = 'TRANSMISSION'
    transmission_hdu.header['EXTNAME'] = 'TRANSMISSION'
    transmission_hdu. header['COMMENT'] = 'Lyman-alpha forest transmission'
    
    # Assemble HDU list
    hdul = fits.HDUList([primary_hdu, metadata_hdu, wavelength_hdu, transmission_hdu])
    
    # Write to file
    print(f"\nWriting output to {output_file}")
    hdul.writeto(output_file, overwrite=True)
    print("Done!")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert transmission files from mock_gen. py to quickquasars format"
    )
    
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='Input transmission file(s) in .fits format from mock_gen.py')
    parser.add_argument('--output', type=str, required=True,
                        help='Output FITS file for quickquasars')
    parser.add_argument('--ra', type=float, nargs='+', required=True,
                        help='Right Ascension value(s) in degrees')
    parser.add_argument('--dec', type=float, nargs='+', required=True,
                        help='Declination value(s) in degrees')
    parser.add_argument('--redshift', type=float, nargs='+', required=True,
                        help='Redshift value(s)')
    parser.add_argument('--mockid', type=int, default=1000,
                        help='Starting MOCKID value (default: 1000)')
    parser.add_argument('--nside', type=int, default=16,
                        help='HEALPix nside parameter (default: 16)')
    parser.add_argument('--pixel', type=int, default=None,
                        help='HEALPix pixel number')
    parser.add_argument('--nest', action='store_true', default=True,
                        help='Use nested HEALPix scheme (default: True)')
    
    args = parser.parse_args()
    
    # Validate inputs
    n_files = len(args.input)
    
    # Handle single vs multiple values for RA, DEC, Z
    if len(args.ra) == 1:
        ra_values = [args.ra[0]] * n_files
    else:
        ra_values = args.ra
        
    if len(args.dec) == 1:
        dec_values = [args.dec[0]] * n_files
    else:
        dec_values = args.dec
        
    if len(args.redshift) == 1:
        redshift_values = [args.redshift[0]] * n_files
    else:
        redshift_values = args.redshift
    
    if len(ra_values) != n_files or len(dec_values) != n_files or len(redshift_values) != n_files:
        raise ValueError(f"Number of RA/DEC/redshift values must match number of input files ({n_files})")
    
    # Convert
    convert_transmission_to_quickquasars(
        args.input, args.output,
        ra_values, dec_values, redshift_values,
        args. mockid, args.nside, args.pixel, args.nest
    )


if __name__ == "__main__":
    main()