#Findings Summary
Though usually represented within one latent space, some psychedelic neural trajectories change their spatial and/or temporal geometry across environmental contexts, and while CEBRA readily produces one low-dimensional geometry for these context-dependent—also interpreted as participant-shared—neural trajectories, it consistently fails to establish their independence from acquisition order in fixed-order recordings; accordingly, we conclude that the reported psilocybin-visit advantage in Stoliker and colleagues’ study is an unreliable indicator of participant-shared alignment between brain activity and context, because stricter participant-held-out evaluation shows baseline mean balanced accuracy exceeding the near-chance psilocybin-visit mean by 0.140, fixed-order analysis treats four position-specific scan runs as environmental contexts—a fundamental ambiguity liable to distort conclusions—and participant-level sensitivity analysis yields 576 CEBRA metric–behaviour association tests with negligible multiplicity-adjusted support plus nine ostensibly network-attributed fixed-classifier losses whose estimates differ from corresponding refitted-classifier ones, leading us to recommend an observational account, qualified by these evidential limitations, as a more limited and less mechanistically specific alternative to the proposed therapeutic interpretation of the relation between participant-shared brain–context alignment at the psilocybin visit and mindset change measured one day later.

For details, see our analysis preprint.

# Reanalysis code

This package contains executable analysis code supporting the secondary statistical reanalysis reported in "Temporal confounding in study of psychedelic brain-context alignment".

## Analysis modules

| Directory | Contents |
|---|---|
| `behavior` | MEQ30 scoring and next-day mindset summaries. |
| `fmri` | Extraction of 332-region time series, strict motion-censored parcel connectivity, fixed-network modularity, and vertex-wise surface global functional connectivity. |
| `context` | Absolute acquisition-time and within-run phase decoding under random-frame, purged-block, participant-held-out, and cross-visit validation. |
| `cebra` | Cohort manifests, CEBRA-Time validation, latent-geometry metrics, multi-gap circular-shift inference, behavioural associations, and network-replacement analyses. |
| `tavrnn` | Corrected Graph-GRU/TAVRNN fitting, feature and density sensitivity analyses, participant-level inference, behavioural associations, normalized node distances, and classical multidimensional scaling. |
| `dcm` | Six-region spectral DCM preparation and estimation, task-specific PEB/BMR/BMA analyses, and compact posterior summaries. |
| `eeg` | Welch spectra, 8-12-Hz alpha power, eyes-closed-minus-movie spectral contrasts, and sequence-length-corrected LZ76 analyses. |

## Data

The public, de-identified fMRI, EEG, and behavioural records are available from OpenNeuro dataset ds006110, version 1.2.0: https://openneuro.org/datasets/ds006110/versions/1.2.0

The TAVRNN reference implementation is available at commit `e0f365aa60dacabbc5fc2d22ecab9e46080c6823`: https://github.com/TAVRNN/TAVRNN_Repo_Paper/tree/e0f365aa60dacabbc5fc2d22ecab9e46080c6823

SPM12 revision 7771 is available from the UCL SPM distribution: https://www.fil.ion.ucl.ac.uk/spm/software/spm12/

## Environment

The CPU analyses use NumPy 2.2.6, SciPy 1.15.3, pandas 2.2.3, PyArrow 19.0.1, NiBabel 5.3.2, scikit-learn 1.6.1, statsmodels 0.14.4, h5py 3.13.0, and Numba 0.61.2. CEBRA analyses use CEBRA 0.6.1 and PyTorch. TAVRNN analyses use PyTorch 2.1 or later. Spectral DCM and PEB use SPM12 revision 7771 with MATLAB Runtime R2019b Update 9 (v97).

## Entry points

Each Python entry point exposes its complete command-line interface through `--help`.

```text
python behavior/prepare_behavioral_scores.py --help

python fmri/extract_volume_roi.py --help
python fmri/analyze_motion_censored_parcel_fc.py --help
python fmri/extract_surface_gfc.py --help
python fmri/analyze_surface_gfc.py --help

python context/analyze_temporal_context_decoding.py --help

python cebra/build_cebra_manifest.py --help
python cebra/run_cebra_jobs.py --help
python cebra/summarize_cebra.py --help
python cebra/analyze_latent_geometry.py --help
python cebra/analyze_cebra_gap_sensitivity.py --help
python cebra/analyze_cebra_behavior_robust.py --help
python cebra/run_cebra_network_replacement.py --help
python cebra/summarize_cebra_network_replacement.py --help
python cebra/analyze_network_replacement_statistics.py --help

python tavrnn/run_tavrnn_reanalysis.py --help
python tavrnn/node_geometry.py --help

python dcm/extract_dcm_spheres.py --help
python dcm/generate_dcm_mat_inputs.py --help
python dcm/prepare_dcm_peb_manifest.py --help
python dcm/run_standalone_dcm_peb.py --help
python dcm/summarize_dcm_peb.py --help

python eeg/extract_eeg_features.py --help
python eeg/analyze_eeg.py --help
python eeg/analyze_spectral_gap.py --help
```

The analysis entry points write numerical arrays, model estimates, participant-level tables, group summaries, inferential statistics, and machine-readable run metadata under caller-selected output directories.
