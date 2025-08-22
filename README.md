# P1D_Mocks

This is a collection of code and notebooks for producing P1D mocks for the Lyman Alpha Forest.

## Functionality

This repo provides a suite of scripts for fitting the one dimensional flux power spectrum (P1D) and mean flux to observations of the Lyman-alpha forest, and generating mock spectra with these properties for relevant studies. The code supports creating customizable synthetic datasets and tuning their statistical properties (P1D and mean flux) to different models or obserational datasets.

## Installation & Cloning

1. Clone the repository:
   ```bash
   git clone https://github.com/m-herbold/P1D_Mocks.git
   cd P1D_Mocks
   ```

2. (Recommended) Set up a virtual environment:
   ```bash
   conda create -n env_name
   
   conda activate env_name
   ```

3. Install dependencies (see below).

## File Structure & Contents
```
├── CF 
│   ├── correlation functions               # ouput by fit, used to calcualte P_G
│   │   ├── e.g. {redshift}_xiG_full_output.txt
│   │   ├── e.g. {redshift}_xiG_half_output.txt
├── Examples
│   ├── fit_power_by_redshift (plots)
│   ├── measured_power_by_redshift (plots)
│   ├── mock_fit_input_files                # example alternative inputs for fit
│   └── xi_g_fit_by_redshift (plots)
│   └── figure8_qmle_desiedrp_results.txt   # DESI EDR P1D measurement
├── Notebooks                               # interractive notebooks
│   ├── fitting_mocks_interractive.ipynb
│   ├── mock_generation_interactive.ipynb
└── P_G
│   ├── Gaussian Power                      # ouput by fit, used to generate mocks
│   │   ├── e.g. P_G-{redshift}.txt
└── scripts
│   ├── mock_fit.py                         # main fitting code
│   ├── mock_gen.py                         # main mock generation code
└── transmission_files                      # output by mock_gen.py
│   ├── e.g. {redshift}
│   │   ├── e.g. transmission_{redshift}_{date}_{ID}txt
```

## Key Scripts

### mock_fit.py

Python script designed to fitting of the 1D Lyman-alpha forest power spectrum (`P1D`) and associated mean flux evolution as a function of redshift. 

#### Inputs

The script accepts several types of inputs via command-line arguments:

- **Redshift Targets (`--z_target` [required])**
  - A single redshift value (e.g., `3.0`)
  - A comma-separated list of redshifts (e.g., `"2.0,2.2,2.4"`)
  - A text file with one or more redshift values (one per line)

- **Power Spectrum Data (`--power_file` [optional])**
  - Path to a `.txt` file containing at least three columns: redshift (`z`), wavenumber (`k` in [km/s]⁻¹), and 1D power spectrum (`P1D`).
  - If not provided, a built-in DESI EDR model is used.

- **Mean Flux Data (`--flux_file` [optional])**
  - Path to a `.txt` file containing two columns: redshift (`z`) and mean flux (`F`).
  - If not provided, a model from Turner et al. (2024) is used.

- **Plotting Flags [optional]:**  
  Use the following flags to generate plots after fitting:
    - `--plot_mean_flux`
    - `--plot_target_power`
    - `--plot_target_xif`
    - `--plot_xig_fit`
    - `--plot_xif_fit`
    - `--plot_xif_recovered`
    - `--plot_recovered_power`

#### Outputs

The script generates both data files and plots, depending on the chosen options:

##### Data Files

- **Half and Full Gaussian Correlation Functions (`xi_G`):**
  - Saved under `../CF/{z}_xiG_half_output.txt` and `../CF/{z}_xiG_full_output.txt`, where `{z}` is the redshift.
  - Format: Two columns (velocity [km/s], xi_G).
  - Used for calculating Gaussian power spectrum

- **Gaussian Power Spectrum:**
  - Saved under `../P_G/P_G-{z}.txt` for each redshift.
  - Format: Two columns (k [km/s]⁻¹, P(k)).
  - Used for mock generation 

##### Plots

Depending on the flags, generates a number of plots.


#### Usage

Basic Example
```bash
python /.../scripts/mock_fit_v4.py --z_target 3.0 --plot_mean_flux --plot_recovered_power
```

Using Custom Data Files
```bash
python /.../scripts/mock_fit_v4.py \
  --z_target redshifts.txt \
  --power_file input_power.txt \
  --flux_file input_flux.txt \
  --plot_mean_flux --plot_recovered_power
```

- Replace `redshifts.txt` with a file containing redshift values.
- Replace `input_power.txt` with a file containing [z, k, P1D] columns.
- Replace `input_flux.txt` with a file containing [z, F] columns.

#### Requirements

- Python 3.x
- Packages: numpy, matplotlib, scipy, astropy, iminuit, argparse

#### Directory Structure

The script saves outputs in directories **relative to the script's location**:

- Power spectrum: `../P_G/`
- Correlation functions: `../CF/`
- Plots: Current working directory

Ensure these directories exist, or the script will create them as needed.

#### Citation

If using this script or its results for published work, please cite:

- **Turner et al. (2024)**
- **Karacayli et al. (2020)**
- **DESI Early Data Release (EDR)**



### mock_gen.py

Python script for generating 1D lognormal mock Lyman-alpha forest spectra at user-specified redshifts. 

The script generates mock transmitted flux fields by:
  - Creating Gaussian random fields with a user-supplied or default power spectrum.
  - Applying lognormal and physical transformations to model the Lyman-alpha forest.
  - Outputting synthetic spectra and diagnostic plots for further analysis.

#### Inputs

##### Required

- `--z_target`:  
  - File path to a `.txt` file containing redshift values (one per line),  
  - or a single float (e.g., `2.2`),  
  - or a comma-separated list of redshifts (e.g., `2.0,2.2,2.4`).

##### Optional

- `--power_files`:  
  - Comma-separated list of power spectrum files (one per redshift),  
  - or leave blank to use the default power spectrum for each redshift.

- `--fit_params`:  
  - Comma-separated string or filename containing four lognormal fitting parameters: `tau0, tau1, nu, sigma2`.  
  - If omitted, uses defaults from the script.

- `--N_mocks`:  
  - Number of mocks to generate per redshift (default: 1).

- Plotting flags (all optional, default off):
  - `--plot_gaussian_field`
  - `--plot_gaussian_power`
  - `--plot_delta_k`
  - `--plot_delta_v`
  - `--plot_delta_z`
  - `--plot_nz`
  - `--plot_optical_depth`
  - `--plot_transmission_v`
  - `--plot_transmission_w`

#### Outputs

- **Transmission Files:**  
  For each mock and redshift, a file is saved in `../transmission_files/{z}` containing two columns: velocity [km/s] and transmitted flux.

- **Diagnostic Plots:**  
  Saved to the current working directory.

- **Console Output:**  
  Information on progress, timing, and key statistics for each redshift.

#### Example Usage

Basic, defaults to DESI-like mean flux and P1D:
``` bash
python /.../scripts/mock_gen_v2.py \
    --z_target 2.0,2.2,2.4 \
    --N_mocks 100 \
    --plot_transmission_w
```

Using user-specified redshift, power, and mean flux parameter information:
```bash
python /.../scripts/mock_gen_v2.py \
    --z_target redshifts.txt \
    --power_files "P_G-2-2.txt,P_G-2-4.txt,P_G-2-6.txt" \
    --N_mocks 100 \
    --fit_params 0.67,5.31,2.16,1.50 \
    --plot_transmission_w
```

#### Requirements

- Python 3.x
- Packages: numpy, matplotlib, pandas, scipy

#### Directory Structure

  - `scripts/mock_gen.py` : Main script.
  - `P_G/` : Default location for power spectrum files.
  - `transmission_files/{z}/` : Output transmission files.
  - `Examples/figure8_qmle_desiedrp_results.txt` : Sample DESI EDR results for direct comparison.

#### Notes

  - All output files are saved in the working directory or in the `transmission_files/` folder.
  - For custom power spectra or parameter files, ensure each file matches the number/order of redshifts.
