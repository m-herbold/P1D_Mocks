#!/usr/bin/env python3

import numpy as np
import fitsio
import time
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.optimize import minimize
from scipy.optimize import least_squares
from iminuit.cost import LeastSquares
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import binned_statistic 
import os
import argparse
import glob
import sys
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# from convert_to_quickquasars_format_v2 import convert_transmission_to_quickquasars
from convert_to_quickquasars_format_v3 import convert_transmission_to_quickquasars

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00', 
                  '#000000', '#FFFFFF']


## helper functions ##

def z_to_bin_index(z, zmin=2.0, dz=0.2):
    """
    Convert redshift to nearest bin index using stable rounding.
    """
    return np.floor((z - zmin) / dz + 0.5).astype(int)


def sample_ref_catalog(catalog, zmin, zmax, dz, n_per_bin, seed):
    with fitsio.FITS(catalog) as f:
        cat = f[1].read(columns=[
            "TARGETID",
            "Z",
            "TARGET_RA",
            "TARGET_DEC", 
            "HPXPIXEL"
        ])

        nbins = int(round((zmax - zmin) / dz))

        z_bins = zmin + dz * np.arange(nbins + 1)
        z_bins = np.round(z_bins, 1)
        
        z_idx = z_to_bin_index(cat["Z"], zmin=zmin, dz=dz)
        valid = (z_idx >= 0) & (z_idx <= nbins)
        
        cat = cat[valid]
        z_idx = z_idx[valid]

        N_per_bin = n_per_bin + 10 # add buffer
        rows = []
        rows_zidx = [] 
        rng = np.random.default_rng()

        for i in range(nbins + 1):
            m = z_idx == i
            if m.sum() < N_per_bin:
                continue
        
            chosen = rng.choice(np.where(m)[0], size=int(N_per_bin), replace=False)
            rows.append(cat[chosen])
            rows_zidx.append(z_idx[chosen]) 
        
        rows = np.concatenate(rows)
        rows_zidx = np.concatenate(rows_zidx)
        
        sample = pd.DataFrame({
            "targetid": rows["TARGETID"],
            "ra": rows["TARGET_RA"],
            "dec": rows["TARGET_DEC"],
            "healpix": rows["HPXPIXEL"],
            "z": rows["Z"],
            "z_bin_index": rows_zidx,
            "z_bin": zmin + dz * rows_zidx 
        })
        sample["z_bin"] = sample["z_bin"].round(1)
        return sample


def fake_catalog(zmin, zmax, dz, n_per_bin, seed):
    rng = np.random.default_rng(seed=seed)
    nbins = int(round((zmax - zmin) / dz))
    z_bins = zmin + dz * np.arange(nbins + 1)
    z_bins = np.round(z_bins, 1)
    
    rows = []
    for i in range(nbins + 1):
        z_bin_val = round(zmin + dz * i, 1)
        n = n_per_bin + 10  # add buffer

        z_vals = rng.uniform(z_bin_val, z_bin_val + dz, size=n)
        z_vals = np.clip(z_vals, zmin, zmax)
    
        rows.append(pd.DataFrame({
            "targetid": rng.integers(1_000_000, 9_999_999, size=n),
            "ra":       rng.uniform(0.0, 360.0, size=n),
            "dec":      rng.uniform(-90.0, 90.0, size=n),
            "healpix":  rng.integers(0, 786432, size=n),  # nside=256 max pixel
            "z":        z_vals,
            "z_bin_index": np.full(n, i, dtype=int),
            "z_bin":    np.full(n, z_bin_val),
        }))
    
    sample = pd.concat(rows, ignore_index=True)
    sample["z_bin"] = sample["z_bin"].round(1)
    return sample


def pop_qso_for_redshift(z, qso_pool):
    if z not in qso_pool or len(qso_pool[z]) == 0:
        raise RuntimeError(f"No remaining QSOs for redshift bin z={z}")
    return qso_pool[z].pop()


def initialize_qso_pool(sample, seed=123):
    """
    (Re)initialize the QSO pool from the sample DataFrame.
    Call this before each conversion to get a fresh pool.
    """
    from collections import defaultdict

    qso_pool = defaultdict(list)
    for _, row in sample.iterrows():
        qso_pool[round(float(row["z_bin"]), 1)].append(row)  # round keys on ingestion

    # Shuffle each bin
    rng = np.random.default_rng(seed=seed)
    for zb in qso_pool:
        rng.shuffle(qso_pool[zb])

    return qso_pool
    

def sample_qso_for_redshift(z, qso_pool, seed, rng=None):
    """
    Sample a QSO without removing it from the pool.
    Allows repeated use of the same pool.
    """
    if z not in qso_pool or len(qso_pool[z]) == 0:
        raise RuntimeError(f"No QSOs available for redshift bin z={z}")

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=seed)
    
    # Sample WITHOUT replacement from the pool
    idx = rng.choice(len(qso_pool[z]))
    return qso_pool[z][idx]


def sample_unique_qsos(z_bins, qso_pool, n_per_bin, seed=None):
    """
    Sample unique QSOs for all redshift bins at once.
    """
    rng = np.random.default_rng(seed=seed)
    sampled_qsos = []
    
    for z_bin in z_bins:
        if z_bin not in qso_pool:
            print(f"⚠ No QSOs for z={z_bin}")
            continue
            
        available = qso_pool[z_bin]
        n_sample = min(n_per_bin, len(available))
        
        # Sample indices without replacement
        indices = rng.choice(len(available), size=n_sample, replace=False)
        sampled_qsos.extend([available[i] for i in indices])
    
    return sampled_qsos

    
def build_redshift_template(catalog_path, zmin, zmax, dz, 
                            z_col: str = "Z", normalize: bool = True):
    """
    Scan a quasar catalog and build a redshift distribution template.

    Parameters
    ----------
    catalog_path : str
        Path to the FITS catalog file.
    zmin, zmax : float
        Redshift range to consider.
    dz : float
        Bin width in redshift.
    z_col : str
        Name of the redshift column in the catalog.
    normalize : bool
        If True, return fractional counts (PDF-like); otherwise return raw counts.

    Returns
    -------
    dict with keys:
        'bin_edges'   : array of bin edges (length nbins+1)
        'bin_centers' : array of bin centers (length nbins)
        'counts'      : raw integer counts per bin
        'weights'     : normalized weights (fractions) per bin
        'zmin', 'zmax', 'dz', 'nbins'
    """
    # Read only the redshift column — fast even for large catalogs
    with fitsio.FITS(catalog_path) as f:
        z = f[1].read(columns=[z_col])[z_col]

    # Bin edges and centers
    bin_edges   = np.arange(zmin, zmax + dz * 0.5, dz)   # robust against float drift
    bin_edges   = np.round(bin_edges, 10)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    nbins       = len(bin_centers)

    # Count quasars per bin
    counts, _ = np.histogram(z, bins=bin_edges)

    # Normalized weights (probability of drawing from each bin)
    total    = counts.sum()
    weights  = counts / total if (normalize and total > 0) else counts.astype(float)

    template = {
        "bin_edges":   bin_edges,
        "bin_centers": bin_centers,
        "counts":      counts,
        "weights":     weights,
        "zmin": zmin, "zmax": zmax, "dz": dz, "nbins": nbins,
        "total_quasars": int(total),
    }
    print(f"[template] Built from {total:,} quasars in {nbins} bins "
          f"({zmin:.1f} ≤ z < {zmax:.1f}, dz={dz})\n")
    return template


# def save_template(template: dict, out_path: str) -> None:
#     """Save template to a .npz file for reuse."""
#     np.savez(out_path, **{k: np.asarray(v) for k, v in template.items()})
#     print(f"[template] Saved to {out_path}")


# def load_template(npz_path: str) -> dict:
#     """Load a previously saved template."""
#     data = np.load(npz_path)
#     template = {k: data[k].item() if data[k].ndim == 0 else data[k]
#                 for k in data.files}
#     return template


def plot_redshift_template(template: dict, outdir = None):
    """
    Quick diagnostic plot for a redshift template dictionary.
    Shows both raw counts and normalized weights side by side.
    """
    bin_centers = template["bin_centers"]
    counts      = template["counts"]
    weights     = template["weights"]
    dz          = template["dz"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # fig.suptitle(fontsize=13)

    # --- Left: raw counts ---
    axes[0].bar(bin_centers, counts, width=dz * 0.85, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Redshift z")
    axes[0].set_ylabel("Number of quasars")
    axes[0].set_title(f"Raw counts  (total = {template['total_quasars']:,})")
    axes[0].set_xticks(np.round(template["bin_edges"], 2))
    axes[0].tick_params(axis="x", rotation=45)

    # --- Right: normalized weights ---
    axes[1].bar(bin_centers, weights, width=dz * 0.85, color="tomato", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Redshift z")
    axes[1].set_ylabel("Fraction of quasars")
    axes[1].set_title("Normalized weights (sums to 1)")
    axes[1].set_xticks(np.round(template["bin_edges"], 2))
    axes[1].tick_params(axis="x", rotation=45)

    # Annotate each bar with its value
    for ax, vals in zip(axes, [counts, weights]):
        for x, v in zip(bin_centers, vals):
            fmt = f"{v:,.0f}" if vals is counts else f"{v:.3f}"
            ax.text(x, v * 1.01, fmt, ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    
    if outdir:
        plot_file = os.path.join(outdir, "redshift_template.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file} \n ")


def plot_template_vs_mock(template: dict,
                          downsampled_df: pd.DataFrame, 
                          outdir=None):
    """
    Overlay the EDR template redshift distribution against the
    downsampled mock catalog to verify the shape match.
    """
    bin_edges   = template["bin_edges"]
    bin_centers = template["bin_centers"]
    dz          = template["dz"]

    # Compute normalized mock distribution using the same bins
    mock_counts, _ = np.histogram(downsampled_df["z"], bins=bin_edges)
    mock_weights   = mock_counts / mock_counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    # fig.suptitle(fontsize=13)

    # --- Left: overlaid distributions ---
    axes[0].bar(bin_centers, template["weights"], width=dz * 0.85,
                label=f"EDR template  (N={template['total_quasars']:,})",
                color="steelblue", alpha=0.6, edgecolor="white")
    axes[0].bar(bin_centers, mock_weights, width=dz * 0.45,
                label=f"Downsampled mock  (N={len(downsampled_df):,})",
                color="tomato", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Redshift z")
    axes[0].set_ylabel("Fraction of quasars")
    axes[0].set_title("Normalized distributions")
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(np.round(bin_edges, 2))
    axes[0].tick_params(axis="x", rotation=45)

    # --- Right: residuals (mock - template) ---
    residuals = mock_weights - template["weights"]
    colors    = ["steelblue" if r >= 0 else "tomato" for r in residuals]
    axes[1].bar(bin_centers, residuals, width=dz * 0.85,
                color=colors, alpha=0.8, edgecolor="white")
    axes[1].axhline(0, color="k", lw=1.0, ls="--")
    axes[1].set_xlabel("Redshift z")
    axes[1].set_ylabel("Δ fraction (mock − template)")
    axes[1].set_title("Residuals")
    axes[1].set_xticks(np.round(bin_edges, 2))
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if outdir:
        plot_file = os.path.join(outdir, "redshift_template_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file} \n")


def apply_redshift_template(mock_path: str, 
                            template: dict, seed: int, 
                            target_total: int | None = None) -> pd.DataFrame:
    """
    Downsample a mock quasar catalog so its redshift distribution
    matches the template (real-data) distribution.

    Strategy
    --------
    For each redshift bin i:
      - N_mock[i]     : number of mocks available in that bin
      - w_template[i] : desired fractional weight from real data

    Find the largest N such that N * w_template[i] <= N_mock[i] for all bins:
      N = min_i( N_mock[i] / w_template[i] )

    Then draw exactly round(N * w_template[i]) quasars from each bin.

    Parameters
    ----------
    mock_path : str
        Path to the mock FITS file (transmission-*.fits format).
    template : dict
        Output of build_redshift_template().
    seed : int
        RNG seed for reproducibility.
    target_total : int or None
        If set, cap the total selected quasars at this value.

    Returns
    -------
    pd.DataFrame with columns: mockid, ra, dec, z, z_bin_index, z_bin
    """
    # Read METADATA extension from mock
    with fitsio.FITS(mock_path) as f:
        meta = f["METADATA"].read()

    z_mock = meta["Z"]

    bin_edges = template["bin_edges"]
    weights   = template["weights"]   # normalized fractions, shape (nbins,)
    nbins     = template["nbins"]

    # Assign each mock quasar to a template bin (-1 or nbins = outside range)
    bin_idx  = np.searchsorted(bin_edges, z_mock, side="right") - 1
    in_range = (bin_idx >= 0) & (bin_idx < nbins)

    print(f"[apply]  Mock catalog    : {len(z_mock):,} quasars total")
    print(f"[apply]  In z range      : {in_range.sum():,} quasars "
          f"({template['zmin']:.1f} ≤ z < {template['zmax']:.1f})")

    # Count available mocks per bin
    N_mock = np.array([
        int((in_range & (bin_idx == i)).sum()) for i in range(nbins)
    ])

    print(f"[apply]  Mocks per bin   : min={N_mock.min()}, max={N_mock.max()}, "
          f"empty={( N_mock == 0).sum()}")

    # Max total drawable while preserving template shape
    valid        = weights > 0
    N_max_by_bin = np.where(valid, N_mock / weights, np.inf)
    N_total      = int(np.floor(N_max_by_bin[valid].min()))

    if target_total is not None:
        N_total = min(N_total, target_total)

    # Per-bin draw counts, clamped to available mocks
    n_draw = np.round(N_total * weights).astype(int)
    n_draw = np.minimum(n_draw, N_mock)

    print(f"[apply]  Max drawable    : {N_total:,} (shape-matched)")
    print(f"[apply]  Will select     : {n_draw.sum():,} quasars")

    # Sample from each bin
    rng  = np.random.default_rng(seed)
    rows = []
    for i in range(nbins):
        if n_draw[i] == 0:
            continue
        indices = np.where(in_range & (bin_idx == i))[0]
        chosen  = rng.choice(indices, size=n_draw[i], replace=False)
        rows.append(meta[chosen])

    if not rows:
        raise RuntimeError("No rows selected — check redshift overlap between mock and template.")

    selected = np.concatenate(rows)

    # Build output DataFrame
    df = pd.DataFrame({
        "mockid": selected["MOCKID"],
        "ra":     selected["RA"],
        "dec":    selected["DEC"],
        "z":      selected["Z"],
    })
    df["z_bin_index"] = np.searchsorted(bin_edges, df["z"].values, side="right") - 1
    df["z_bin"]       = (template["zmin"] + template["dz"] * df["z_bin_index"]).round(2)

    return df


def get_template_file(spec_path: str) -> str:
    """
    Return a single transmission FITS file from spec_path to use as a format template.
    Searches recursively through redshift subdirectories.
    """
    matches = glob.glob(os.path.join(spec_path, "**", "transmission_*.fits"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No transmission FITS files found under: {spec_path}")
    return matches[0]


def save_downsampled_mock(
    mock_catalog_path: str,
    downsampled_df: pd.DataFrame,
    nside, pixel, ext,
    outdir = None):
    """
    Save the downsampled mock catalog in the same FITS format as the input file.
    
    Preserves all original extensions (WAVELENGTH, TRANSMISSION, etc.) but
    replaces the METADATA extension with only the downsampled rows.
    The output file is saved in the same directory as the input, with a
    suffix appended to the filename before the .fits extension.

    Parameters
    ----------
    mock_catalog_path : str
        Path to the original mock FITS file.
    downsampled_df : pd.DataFrame
        Output of apply_redshift_template(). Must contain: mockid, ra, dec, z.
    suffix : str
        String appended to the filename stem (default: "_sampled").

    Returns
    -------
    str : path to the saved output file.
    """
    # Build output path:  /path/to/transmission-16-0_2.fits
    #                  -> /path/to/transmission-16-0_2_sampled.fits
    input_path  = Path(mock_catalog_path)

    file_name = f"/transmission_{nside}_{pixel}_{ext}.fits"
    output_path = outdir+file_name
    
    # Indices of selected rows in the original METADATA extension
    selected_mockids = downsampled_df["mockid"].values

    with fitsio.FITS(mock_catalog_path, "r") as f_in, \
         fitsio.FITS(str(output_path), "rw", clobber=True) as f_out:

        # --- Extension 0: primary IMAGE_HDU (copy as-is) ---
        primary_data   = f_in[0].read()
        primary_header = f_in[0].read_header()
        f_out.write(primary_data, header=primary_header)

        # --- Extension 1: METADATA (write only downsampled rows) ---
        meta_all = f_in["METADATA"].read()
        meta_header = f_in["METADATA"].read_header()

        # Match selected MOCKIDs back to original row indices
        mockid_to_idx = {mid: i for i, mid in enumerate(meta_all["MOCKID"])}
        selected_idx  = np.array([mockid_to_idx[mid] for mid in selected_mockids])
        selected_idx.sort()   # preserve original ordering
        meta_selected = meta_all[selected_idx]

        f_out.write(meta_selected, extname="METADATA", header=meta_header)

        # --- Extensions 2+: WAVELENGTH, TRANSMISSION, etc. ---
        # These are per-quasar image arrays — rows must match METADATA
        for ext_num in range(2, len(f_in)):
            ext        = f_in[ext_num]
            ext_info   = ext.get_info()
            ext_name   = ext_info.get("extname", f"EXT{ext_num}")
            ext_header = ext.read_header()
            ext_data   = ext.read()

            # If this extension has one row per quasar, slice it accordingly
            if ext_data.ndim > 0 and ext_data.shape[0] == len(meta_all):
                ext_data = ext_data[selected_idx]

            f_out.write(ext_data, extname=ext_name, header=ext_header)

    print(f"[save]  Wrote {len(selected_idx):,} quasars to: {output_path}\n")
    return str(output_path)


# def simple_diagnostic_plot(output_file, outdir=None):
#     # DIAGNOSTIC PLOT: Mean Transmission vs Redshift
#     fits_file = output_file

#     # Read the data
#     with fits.open(fits_file) as hdul:
#         metadata = Table.read(hdul['METADATA'])
#         transmission = hdul['TRANSMISSION'].data
    
#     # Calculate mean transmission for each quasar
#     mean_trans_per_qso = np.mean(transmission, axis=1)
    
#     # Get statistics per redshift bin
#     unique_z = np.unique(metadata['Z'])
#     z_bins = []
#     mean_trans = []
#     std_trans = []
#     n_qsos = []

#     for z in unique_z:
#         mask = metadata['Z'] == z
#         trans_at_z = mean_trans_per_qso[mask]
        
#         z_bins.append(z)
#         mean_trans.append(np.mean(trans_at_z))
#         std_trans.append(np.std(trans_at_z))
#         n_qsos.append(np.sum(mask))
    
#     z_bins = np.array(z_bins)
#     mean_trans = np.array(mean_trans)
#     std_trans = np.array(std_trans)

#     # Build stats string for fig.text()
#     lines = []
#     lines.append(f"{'Redshift':<10} {'N_QSOs':<10} {'Mean':<12} {'Std':<12}")
#     lines.append("-" * 46)
#     for z, n, m, s in zip(z_bins, n_qsos, mean_trans, std_trans):
#         lines.append(f"{z:<10.1f} {n:<10d} {m:<12.6f} {s:<12.6f}")
#     lines.append("-" * 46)
#     lines.append(f"{'TOTAL':<10} {len(metadata):<10d} {mean_trans_per_qso.mean():<12.6f}")
#     stats_str = "\n".join(lines)

#     # Create figure with extra space at bottom for stats
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
#     fig.subplots_adjust(bottom=0.28) 
    
#     # Top: Mean transmission vs redshift
#     ax1.scatter(metadata['Z'], mean_trans_per_qso, 
#                alpha=0.3, s=30, color='#F18F01', label='Individual QSOs')
    
#     ax1.set_xlabel('Redshift', fontsize=12, fontweight='bold')
#     ax1.set_ylabel('Mean Transmission', fontsize=12, fontweight='bold')
#     ax1.set_title('Lyman-α Forest Transmission vs Redshift', fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
#     ax1.set_ylim([0.4, 0.9])
    
#     # Bottom: Number of QSOs per bin
#     ax2.bar(z_bins, n_qsos, width=0.15, color='#06A77D', 
#             edgecolor='black', alpha=0.7)
#     ax2.set_xlabel('Redshift', fontsize=12, fontweight='bold')
#     ax2.set_ylabel('N QSOs', fontsize=12, fontweight='bold')
#     ax2.set_title('QSO Count per Redshift Bin', fontsize=12, fontweight='bold')
#     ax2.grid(True, alpha=0.3, axis='y')
#     for z, n in zip(z_bins, n_qsos):
#         ax2.text(z, n + 0.1, str(n), ha='center', fontsize=9, fontweight='bold')

#     fig.text(
#         0.5, 0.01, 
#         stats_str,
#         ha='center', va='bottom',
#         fontsize=9,
#         fontfamily='monospace',         # critical for column alignment
#         bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8, edgecolor='gray')
#     )

#     plt.tight_layout(rect=[0, 0.28, 1, 1])  # match subplots_adjust bottom

#     if outdir:
#         plot_file = os.path.join(outdir, "transmission_diagnostics.png")
#         plt.savefig(plot_file, dpi=300, bbox_inches='tight')
#         print(f"Plot saved to: {plot_file} \n")


def simple_diagnostic_plot(output_file, outdir=None):
    # DIAGNOSTIC PLOT: Mean Transmission vs Redshift (blue side of Lya only)
    LYA_WAVELENGTH = 1215.67  # Angstroms

    with fits.open(output_file) as hdul:
        metadata = Table.read(hdul['METADATA'])
        wavelength = hdul['WAVELENGTH'].data    # 1D array
        transmission = hdul['TRANSMISSION'].data  # 2D: (n_qso, n_wave)

    # Mean transmission per quasar using only pixels blueward of Lya
    mean_trans_per_qso = np.full(len(metadata), np.nan)
    for i, z in enumerate(metadata['Z']):
        lya_obs = LYA_WAVELENGTH * (1 + z)
        blue_mask = wavelength < lya_obs
        if blue_mask.sum() > 0:
            mean_trans_per_qso[i] = np.mean(transmission[i, blue_mask])

    # Get statistics per redshift bin
    unique_z = np.unique(metadata['Z'])
    z_bins = []
    mean_trans = []
    std_trans = []
    n_qsos = []

    for z in unique_z:
        mask = metadata['Z'] == z
        trans_at_z = mean_trans_per_qso[mask]
        trans_at_z = trans_at_z[np.isfinite(trans_at_z)]

        z_bins.append(z)
        mean_trans.append(np.mean(trans_at_z) if len(trans_at_z) > 0 else np.nan)
        std_trans.append(np.std(trans_at_z) if len(trans_at_z) > 0 else np.nan)
        n_qsos.append(np.sum(mask))

    z_bins = np.array(z_bins)
    mean_trans = np.array(mean_trans)
    std_trans = np.array(std_trans)

    # Build stats string for fig.text()
    valid_mean = mean_trans_per_qso[np.isfinite(mean_trans_per_qso)]
    lines = []
    lines.append(f"{'Redshift':<10} {'N_QSOs':<10} {'Mean':<12} {'Std':<12}")
    lines.append("-" * 46)
    for z, n, m, s in zip(z_bins, n_qsos, mean_trans, std_trans):
        lines.append(f"{z:<10.1f} {n:<10d} {m:<12.6f} {s:<12.6f}")
    lines.append("-" * 46)
    lines.append(f"{'TOTAL':<10} {len(metadata):<10d} {valid_mean.mean():<12.6f}")
    stats_str = "\n".join(lines)

    # Create figure with extra space at bottom for stats
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.subplots_adjust(bottom=0.28)

    # Top: Mean transmission vs redshift
    ax1.scatter(metadata['Z'], mean_trans_per_qso,
                alpha=0.3, s=30, color='#F18F01', label='Individual QSOs')
    ax1.set_xlabel('Redshift', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Transmission', fontsize=12, fontweight='bold')
    ax1.set_title(r'Lyman-α Forest Transmission vs Redshift'
                  '\n' r'(blue side of Ly$\alpha$ only)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0.4, 0.9])

    # Bottom: Number of QSOs per bin
    ax2.bar(z_bins, n_qsos, width=0.15, color='#06A77D',
            edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Redshift', fontsize=12, fontweight='bold')
    ax2.set_ylabel('N QSOs', fontsize=12, fontweight='bold')
    ax2.set_title('QSO Count per Redshift Bin', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for z, n in zip(z_bins, n_qsos):
        ax2.text(z, n + 0.1, str(n), ha='center', fontsize=9, fontweight='bold')

    fig.text(
        0.5, 0.01,
        stats_str,
        ha='center', va='bottom',
        fontsize=9,
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8, edgecolor='gray')
    )

    plt.tight_layout(rect=[0, 0.28, 1, 1])

    if outdir:
        plot_file = os.path.join(outdir, "transmission_diagnostics.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file} \n")

## main ## 
def main():
    ap = argparse.ArgumentParser(
        description="Convert raw P1D mocks into QQ input format."
    )
    ap.add_argument("--spec_path", required=True, help="Path to transmisison files")
    ap.add_argument("--z_template", required=False, help="Apply redshift distribution downsampling to match catalog (if provided)")
    ap.add_argument("--catalog", required=False, help="Catalog Reference for Metadata")
    ap.add_argument("--zmin", required=True, help="Minimum redshift")
    ap.add_argument("--zmax", required=True, help="Maximum redshift")
    ap.add_argument("--dz", required=True, help="Redshift bin separation")
    ap.add_argument("--seed", required=False, help="Optional seed for repeatability")
    ap.add_argument("--N", required=False, help="Number of QSO per redshift bin to sample")
    ap.add_argument("--nside", required=False, help="Nside (optional, default 16)")
    ap.add_argument("--pixel", required=False, help="Pixel (optional, default 1)")
    ap.add_argument("--wmin", required=False, help="Minimum output wavelength (default 3600)")
    ap.add_argument("--wmax", required=False, help="Maximum output wavelength (default 9800)")
    ap.add_argument("--dwave", required=False, help="Output wavelenth spacing (default 0.2, mimcks Ohio P1D mocks)")
    ap.add_argument("--diag_plot", required=False, help="Optionally saves diagnostic plots")
    ap.add_argument("--outdir", type=str, required=True, help="Directory to write output files")
    
    args = ap.parse_args()

    #### Assign + process args ####
    spec_path = args.spec_path
    z_template = args.z_template
    zmin = float(args.zmin)
    zmax = float(args.zmax)
    dz = float(args.dz)
    seed = int(args.seed)

    if args.N:
        n_per_bin = float(args.N)
    else:
        all_trans_files = sorted(glob.glob(os.path.join(spec_path, "**", "transmission_*.fits"), recursive=True))
        n_per_bin = all_trans_files

    if args.nside:
        nside= float(args.nside)
    else:
        nside=16
        
    if args.pixel:
        pixel=float(args.pixel)
    else: 
        pixel=1
        
    if args.wmin:
        wmin=float(args.wmin)
    else: 
        wmin=3600
        
    if args.wmax:
        wmax=float(args.wmax)
    else: 
        wmax=9800
        
    if args.dwave:
        dwave=float(args.dwave)
    else: 
        dwave=0.2

    file_index = 1 # total: 3072 allowed by quickquasars w/ seed

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)  # once, before the loop
        outdir = args.outdir
    else:
        raise ValueError("--outdir must be provided")   
    
    print('\n\n')
    
    ### CATALOG ###
    
    ### if a ref catalog is provided, then sample for metadata 
    if args.catalog:
        ref_catalog = args.catalog
        print (f'Using catalog: {ref_catalog}.\n')
        sample = sample_ref_catalog(ref_catalog, zmin, zmax, dz, n_per_bin, seed)
    else: 
        print ('No catalog provided, metadata will be randomized.\n')
        sample = fake_catalog((zmin, zmax, dz, n_per_bin, seed))

    ### CONVERSION ###

    ### convert to qq format
    print('\nConverting to quickquasars format.\n')
    redshift_bins = np.arange(zmin, zmax, dz)

    qso_pool = initialize_qso_pool(sample, seed)
    random.seed(seed)
    all_input_files = []
    
    # Round bins to match pool keys
    redshift_bins = np.round(np.arange(zmin, zmax + dz, dz), 1)

    # Check pool inventory
    print("QSO Pool Inventory:")
    total_available = 0
    for z_bin in redshift_bins:
        count = len(qso_pool.get(round(float(z_bin), 1), []))
        total_available += count
        status = "✓" if count >= n_per_bin else "✗"
        print(f"  z={z_bin}: {count} QSOs available {status}")
    
    print(f"\nTotal QSOs available: {total_available}")
    print(f"Total QSOs needed: {len(redshift_bins) * n_per_bin}")
    if total_available < len(redshift_bins) * n_per_bin:
        print("\n⚠ WARNING: Not enough QSOs in pool!")
        print("   Reduce n_per_bin or expand the sample")
    
    # validate pool size
    for zb, entries in qso_pool.items():
        if n_per_bin > len(entries):
            raise ValueError(
                f"n_per_bin ({n_per_bin}) exceeds pool size in z_bin {zb} ({len(entries)} entries)"
            )
    
    random.seed(seed)
    
    all_input_files = []
    all_ra = []
    all_dec = []
    all_redshift = []
    all_mockid = []
    
    print(f"\nCollecting {n_per_bin} files per redshift bin...")
    for z_bin in redshift_bins:
        z_key = round(float(z_bin), 1)  # round lookup key to match pool
        if z_key not in qso_pool or len(qso_pool[z_key]) < n_per_bin:
            print(f"  ⚠ z={z_bin}: Only {len(qso_pool.get(z_key, []))} QSOs available, skipping")
            continue
  
        # Convert redshift to folder format
        z_int = int(z_bin)
        z_frac = int(round((z_bin - z_int) * 10))
        redshift_folder = f"{z_int}-{z_frac}"
        
        # Find transmission files
        pattern = f"{spec_path}{redshift_folder}/transmission_*.fits"
        available_files = glob.glob(pattern)
        
        if not available_files:
            print(f"  ⚠ No transmission files in {redshift_folder}/")
            continue
        
        # Sample files
        n_sample = min(n_per_bin, len(available_files))
        selected_files = random.sample(available_files, int(n_sample))
        
        # Assign QSOs to files
        for input_file in selected_files:
            qso = pop_qso_for_redshift(z_bin, qso_pool)
            
            all_input_files.append(input_file)
            all_ra.append(qso["ra"])
            all_dec.append(qso["dec"])
            all_redshift.append(z_bin)
            all_mockid.append(int(qso["targetid"]))
        
        print(f"  ✓ z={z_bin}: {n_sample} files, {n_sample} QSOs assigned")
    
    print(f"\nTotal: {len(all_input_files)} quasars")
    
    # -------------------- CONVERT --------------------
    if len(all_input_files) == 0:
        print("\n✗ No files to convert!")
    else:
        # timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        # output_file = f"transmission-{nside}-{pixel}_{timestamp}.fits"

        output_file = os.path.join(args.outdir, 
                                   f"transmission-{nside}-{pixel}_{file_index}.fits")
        file_index += 1  # increment for next iteration
        
        print(f"\nConverting to: {output_file}\n")
        
        try:
            output = convert_transmission_to_quickquasars(
                input_files=all_input_files,
                output_file=output_file,
                ra_values=all_ra,
                dec_values=all_dec,
                redshift_values=all_redshift,
                mockid_start=1000,
                wavelength_min=wmin,
                wavelength_max=wmax,
                dwave_out = dwave, 
            )
            
            print(f"\n✓ Success! Output: {output}")
            
            # Quick shape check
            with fits.open(output) as hdul:
                print(f"\nOutput shape: {hdul['TRANSMISSION'].data.shape}")
                
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    # Optional diagnostic plot
    if args.diag_plot:
        simple_diagnostic_plot(output, outdir)

    ### REDSHIFT TEMPLATE ###
    
    ### if z-template provided, downsample in z
    if args.z_template and not args.catalog:
        raise ValueError("--z_template requires --catalog to also be provided.")
    if args.z_template and args.catalog:
        print(f"Bulding redshift template from provided catalog.")
        template = build_redshift_template(args.catalog, zmin, zmax+dz, dz)
        if args.diag_plot:
            plot_redshift_template(template, outdir)
            
    print("Applying redshift template.\n")
    mock_sample = apply_redshift_template(output, template, seed)
    
    # print(mock_sample.head(10))
    print(f"\nz range in output: {mock_sample['z'].min():.3f} – {mock_sample['z'].max():.3f}")
    print(f"Total selected: {len(mock_sample):,}\n")

    ## save file ##
    save_downsampled_mock(output_file, mock_sample, nside, pixel, file_index, outdir)

    if args.diag_plot:
        plot_template_vs_mock(template, 
                              mock_sample, 
                              outdir)


if __name__ == "__main__":
    main()
