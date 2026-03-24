#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.stats import binned_statistic

LYA_WAVELENGTH = 1215.67  # Angstrom


# -----------------------
# IO helpers
# -----------------------
def read_fibermap(spectra_path):
    with fits.open(spectra_path, memmap=True) as hdul:
        if "FIBERMAP" not in hdul:
            raise ValueError(f"{spectra_path} missing FIBERMAP HDU")
        fm = hdul["FIBERMAP"].data

        # canonical columns in DESI fibermap
        tid = fm["TARGETID"].astype(np.int64)

        # prefer TARGET_RA/DEC for object coords; fallback to FIBER_RA/DEC if needed
        if "TARGET_RA" in fm.columns.names and "TARGET_DEC" in fm.columns.names:
            ra = fm["TARGET_RA"].astype(float)
            dec = fm["TARGET_DEC"].astype(float)
            ra_name, dec_name = "TARGET_RA", "TARGET_DEC"
        elif "FIBER_RA" in fm.columns.names and "FIBER_DEC" in fm.columns.names:
            ra = fm["FIBER_RA"].astype(float)
            dec = fm["FIBER_DEC"].astype(float)
            ra_name, dec_name = "FIBER_RA", "FIBER_DEC"
        else:
            raise ValueError(
                "FIBERMAP missing RA/DEC columns (expected TARGET_RA/TARGET_DEC or FIBER_RA/FIBER_DEC)")

        return fm, tid, ra, dec, ra_name, dec_name


def read_truth_z(truth_path):
    with fits.open(truth_path, memmap=True) as hdul:
        if "TRUTH" not in hdul:
            raise ValueError(f"{truth_path} missing TRUTH HDU")
        t = hdul["TRUTH"].data

        if "TARGETID" not in t.columns.names:
            raise ValueError("TRUTH table missing TARGETID column")

        tid = t["TARGETID"].astype(np.int64)

        if "Z" in t.columns.names:
            z = t["Z"].astype(float)
            z_name = "Z"
        elif "TRUEZ" in t.columns.names:
            z = t["TRUEZ"].astype(float)
            z_name = "TRUEZ"
        else:
            raise ValueError("TRUTH table missing Z/TRUEZ columns")

        return t, tid, z, z_name


def match_by_targetid(left_targetid, right_targetid):
    """
    Returns indices (i_left, i_right) such that left_targetid[i_left] == right_targetid[i_right]
    Robust to ordering; only keeps intersection.
    """
    r_sort = np.argsort(right_targetid)
    r_tid_sorted = right_targetid[r_sort]

    pos = np.searchsorted(r_tid_sorted, left_targetid)
    ok = (pos >= 0) & (pos < len(r_tid_sorted)) & (
        r_tid_sorted[pos] == left_targetid)

    i_left = np.where(ok)[0]
    i_right = r_sort[pos[ok]]
    return i_left, i_right


def read_band_images(spectra_path, band):
    """
    For your file layout:
      B_WAVELENGTH, B_FLUX, B_IVAR, B_MASK etc.
    """
    with fits.open(spectra_path, memmap=True) as hdul:
        w = hdul[f"{band}_WAVELENGTH"].data.astype(float)
        f = hdul[f"{band}_FLUX"].data.astype(float)
        iv = hdul[f"{band}_IVAR"].data.astype(float)
        m = hdul[f"{band}_MASK"].data
    return w, f, iv, m


# -----------------------
# plot helpers
# -----------------------
def _savefig(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def mean_flux_per_spectrum(flux, wave, redshifts, ivar=None, mask=None):
    """
    Compute mean flux per spectrum using only pixels blueward of each
    quasar's observed Lyman-alpha wavelength (the IGM forest region).

    Masking also excludes:
      - non-finite flux
      - ivar <= 0 (if provided)
      - mask != 0  (if provided)

    Parameters
    ----------
    flux : (n_spec, n_wave) array
    wave : (n_wave,) array  -- wavelength grid for this band
    redshifts : (n_spec,)   -- redshift of each quasar
    ivar : (n_spec, n_wave) array, optional
    mask : (n_spec, n_wave) array, optional

    Returns
    -------
    out : (n_spec,) array of mean flux values (NaN where no valid pixels)
    """
    f = np.array(flux, copy=False)
    good = np.isfinite(f)
    if ivar is not None:
        good &= np.isfinite(ivar) & (ivar > 0)
    if mask is not None:
        good &= (mask == 0)

    out = np.full(f.shape[0], np.nan, dtype=float)
    for i in range(f.shape[0]):
        z = redshifts[i]
        if not np.isfinite(z):
            continue
        lya_obs = LYA_WAVELENGTH * (1.0 + z)
        blue_mask = wave < lya_obs          # pixels blueward of Lya for this quasar
        gi = good[i] & blue_mask
        if np.any(gi):
            out[i] = np.mean(f[i, gi])
    return out


# -----------------------
# plotting functions
# -----------------------
def plot_overview(fm, ra, dec, ra_name, dec_name, z, z_name, outprefix):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # redshift hist
    ax = axes[0, 0]
    bins = np.linspace(np.nanmin(z), np.nanmax(
        z), 30) if np.isfinite(z).any() else 30
    ax.hist(z[np.isfinite(z)], bins=bins, alpha=0.8,
            color="steelblue", edgecolor="black")
    ax.set_xlabel(f"Redshift ({z_name})")
    ax.set_ylabel("Number of QSOs")
    ax.set_title("Redshift distribution (from TRUTH)")
    ax.grid(True, alpha=0.3)

    # sky
    ax = axes[0, 1]
    ax.scatter(ra, dec, s=12, alpha=0.6, color="darkorange")
    ax.set_xlabel(f"{ra_name} [deg]")
    ax.set_ylabel(f"{dec_name} [deg]")
    ax.set_title("Sky positions (from FIBERMAP)")
    ax.grid(True, alpha=0.3)

    # counts per z bin
    ax = axes[1, 0]
    zfinite = z[np.isfinite(z)]
    if len(zfinite) > 0:
        zbins = np.linspace(zfinite.min(), zfinite.max(), 15)
        counts, _ = np.histogram(zfinite, bins=zbins)
        zcent = 0.5 * (zbins[:-1] + zbins[1:])
        ax.bar(zcent, counts, width=0.8 *
               (zbins[1] - zbins[0]), color="slateblue", alpha=0.8)
        ax.set_xlabel(f"Redshift ({z_name})")
        ax.set_ylabel("Count")
        ax.set_title("Counts per redshift bin")
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.1, 0.5, "No finite redshifts to bin.", transform=ax.transAxes)

    # summary
    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        "SPECTRA SUMMARY\n\n"
        f"N spectra (FIBERMAP rows): {len(fm)}\n"
        f"RA/DEC columns: {ra_name}, {dec_name}\n\n"
        "TRUTH SUMMARY (matched by TARGETID)\n\n"
        f"Matched objects: {len(z)}\n"
        f"Finite z: {np.isfinite(z).sum()}\n"
    )
    if np.isfinite(z).any():
        txt += f"z range: {np.nanmin(z):.3f} .. {np.nanmax(z):.3f}\n"
        txt += f"mean z:  {np.nanmean(z):.3f}\n"
    ax.text(0.05, 0.95, txt, va="top", family="monospace", fontsize=10)

    _savefig(f"{outprefix}_1_overview.png")


def plot_wavelength_grid(wave_by_band, outprefix):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for b, w in wave_by_band.items():
        ax.plot(w, label=f"{b}_WAVELENGTH", lw=1)
    ax.set_xlabel("Pixel index")
    ax.set_ylabel("Wavelength [Å]")
    ax.set_title("Wavelength arrays (from spectra file)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    for b, w in wave_by_band.items():
        dw = np.diff(w)
        ax.plot(dw, label=f"{b} dλ", lw=1)
    ax.set_xlabel("Pixel index")
    ax.set_ylabel("Δλ [Å]")
    ax.set_title("Wavelength spacing")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    for b, w in wave_by_band.items():
        dw = np.diff(w)
        ax.hist(dw, bins=60, alpha=0.5, label=b, density=True)
    ax.set_xlabel("Δλ [Å]")
    ax.set_ylabel("Density")
    ax.set_title("Δλ distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")
    lines = ["WAVELENGTH GRID STATS\n"]
    for b, w in wave_by_band.items():
        dw = np.diff(w)
        lines += [
            f"{b}: N={len(w)} range={w.min():.1f}..{w.max():.1f} Å",
            f"   mean dλ={dw.mean():.4f} median dλ={np.median(dw):.4f} std dλ={dw.std():.6f}",
        ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top",
            family="monospace", fontsize=10)

    _savefig(f"{outprefix}_2_wavelength_grid.png")


def plot_mean_flux_vs_z(z, z_name, wave_by_band, band_flux, band_ivar, band_mask, outprefix):
    """
    Plot mean flux vs redshift for each band, using only pixels blueward of
    each quasar's observed Lyman-alpha wavelength.
    """
    bands = list(band_flux.keys())
    fig, axes = plt.subplots(2, len(bands), figsize=(
        6 * len(bands), 9), squeeze=False)

    zfinite = z[np.isfinite(z)]
    if len(zfinite) == 0:
        raise ValueError(
            "No finite redshifts available for mean-flux-vs-z plots.")

    zbins = np.linspace(zfinite.min(), zfinite.max(), 20)
    zcent = 0.5 * (zbins[:-1] + zbins[1:])

    for j, b in enumerate(bands):
        # Pass the band's wavelength array and per-quasar redshifts so the
        # helper can apply a per-spectrum blue-side mask
        mf = mean_flux_per_spectrum(
            band_flux[b], wave_by_band[b], z,
            ivar=band_ivar[b], mask=band_mask[b]
        )

        ax = axes[0, j]
        ax.scatter(z, mf, s=10, alpha=0.4)
        ax.set_xlabel(f"Redshift ({z_name})")
        ax.set_ylabel(f"Mean flux ({b})")
        ax.set_title(
            f"Mean flux vs z ({b})\n" r"(blue side of Ly$\alpha$ only)")
        ax.grid(True, alpha=0.3)

        ax = axes[1, j]
        finite_mf = np.isfinite(mf)
        if finite_mf.sum() == 0:
            ax.text(0.5, 0.5, f"No valid blue-side pixels\nin band {b}",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            bs = binned_statistic(
                z[finite_mf], mf[finite_mf], statistic="mean", bins=zbins)
            valid = np.isfinite(bs.statistic)
            ax.plot(zcent[valid], bs.statistic[valid], "o-", lw=2)
        ax.set_xlabel(f"Redshift ({z_name})")
        ax.set_ylabel(f"Binned mean flux ({b})")
        ax.set_title(
            f"Binned mean flux ({b})\n" r"(blue side of Ly$\alpha$ only)")
        ax.grid(True, alpha=0.3)

    _savefig(f"{outprefix}_3_mean_flux_vs_z.png")


def plot_random_spectra(z, z_name, wave_by_band, band_flux, band_ivar, band_mask, outprefix, nspec=3, seed=42):
    rng = np.random.default_rng(seed)
    n = len(z)
    nspec = min(nspec, n)
    idxs = rng.choice(n, size=nspec, replace=False)

    bands = list(band_flux.keys())
    fig, axes = plt.subplots(nspec, len(bands), figsize=(
        6 * len(bands), 3.5 * nspec), squeeze=False)

    for i, idx in enumerate(idxs):
        for j, b in enumerate(bands):
            ax = axes[i, j]
            w = wave_by_band[b]
            f = band_flux[b][idx]
            iv = band_ivar[b][idx]
            m = band_mask[b][idx]

            # mask bad pixels for plotting continuity
            good = np.isfinite(f) & np.isfinite(iv) & (iv > 0) & (m == 0)
            fp = np.where(good, f, np.nan)

            ax.plot(w, fp, lw=0.8, color="black")

            # simple noise visualization
            if np.any(good):
                sig = np.full_like(f, np.nan, dtype=float)
                sig[good] = 1.0 / np.sqrt(iv[good])
                ax.fill_between(w, fp - sig, fp + sig,
                                color="tab:blue", alpha=0.15, linewidth=0)

            if np.isfinite(z[idx]):
                lya_obs = LYA_WAVELENGTH * (1.0 + z[idx])
                ax.axvline(lya_obs, color="red", ls=":", alpha=0.8,
                           label=f"Lyα (z={z[idx]:.2f})")

            ax.set_xlabel("Wavelength [Å]")
            ax.set_ylabel("Flux")
            ax.set_title(f"idx={idx} band={b}  z({z_name})={z[idx]:.3f}")
            ax.grid(True, alpha=0.25)

            if i == 0 and j == 0 and np.isfinite(z[idx]):
                ax.legend(loc="best", fontsize=9)

    _savefig(f"{outprefix}_4_random_spectra.png")


# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Diagnostics for quickquasars spectra/truth outputs; uses FIBERMAP for RA/DEC and TRUTH for Z."
    )
    ap.add_argument("--spectra", required=True, help="Path to spectra-*.fits")
    ap.add_argument("--truth", required=True,
                    help="Path to truth-*.fits (for redshift axis)")
    ap.add_argument("--outprefix", default="qqdiag", help="Output plot prefix")
    ap.add_argument("--nspec", type=int, default=3,
                    help="Number of random spectra to plot")
    args = ap.parse_args()

    # plot style
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["legend.fontsize"] = 10

    # read fibermap
    fm, fm_tid, ra, dec, ra_name, dec_name = read_fibermap(args.spectra)

    # read truth Z
    truth, truth_tid, truth_z, z_name = read_truth_z(args.truth)

    # match by TARGETID (so all arrays align with spectra rows)
    i_fm, i_truth = match_by_targetid(fm_tid, truth_tid)
    if len(i_fm) == 0:
        raise ValueError("No TARGETID overlap between FIBERMAP and TRUTH.")

    # select matched rows
    fm = fm[i_fm]
    ra = ra[i_fm]
    dec = dec[i_fm]
    z = truth_z[i_truth]
    fm_tid = fm_tid[i_fm]

    # read band images, then select matched rows too
    wave_by_band = {}
    band_flux = {}
    band_ivar = {}
    band_mask = {}
    for b in ["B", "R", "Z"]:
        w, f, iv, m = read_band_images(args.spectra, b)
        wave_by_band[b] = w
        band_flux[b] = f[i_fm, :]
        band_ivar[b] = iv[i_fm, :]
        band_mask[b] = m[i_fm, :]

    # plots
    plot_overview(fm, ra, dec, ra_name, dec_name, z, z_name, args.outprefix)
    plot_wavelength_grid(wave_by_band, args.outprefix)
    plot_mean_flux_vs_z(z, z_name, wave_by_band, band_flux,
                        band_ivar, band_mask, args.outprefix)
    plot_random_spectra(z, z_name, wave_by_band, band_flux, band_ivar, band_mask, args.outprefix,
                        nspec=args.nspec)

    print(f"Wrote plots: {args.outprefix}_*.png")
    print(
        f"Matched {len(z)} objects by TARGETID between spectra FIBERMAP and truth TRUTH.")


if __name__ == "__main__":
    main()
