# Changelog

## 2026-11: ER-Decon II built from source

- **ER-Decon II built from source.** The deconvolution engine, ER-Decon II (`core2_decon`; Arigovindan et al., PNAS 2013), is now built from https://github.com/Biochemistry-Biophysics-Cheng-Lab/ER-DECON-II into the conda environment (README, "Installation"). The Priism-built `core2_decon`/`core2_decon_static` and the bundled Priism, X11/Motif, FFTW and OpenMP libraries (`ardecon/bin`, `ardecon/lib`, `etc/conda`) are removed. Results are those of 0.9: on the tutorial map, 14 of 16.8 million voxels differ from 0.9's result, each in the last stored digit.
- **Documentation and citation.** The README states that ER-Decon II is the deconvolution engine and what AR-Decon adds (cryo-EM OTF estimation from half-map dFSC, and the processing workflow), and asks users to cite both the ER-Decon and the AR-Decon papers. New `CITATION.cff` listing both.
- **dFSC.** `dfsc-0.0.1.py` updates its dFSC map at each cone's voxels instead of on whole volumes: about 5× faster, byte-identical output.
- **`--seed N`** for `ardecon`, `param_search` and `dfsc-0.0.1.py`: fixed cone directions, so runs are exactly repeatable (unseeded by default, as in 0.9).
- **NumPy 2.** `Mrc.py` works with NumPy 1 and 2 (results unchanged under NumPy 1).
- **Fix.** `param_search --iter` set the wrong variable and was ignored.
- **Installation.** `environment.yml` uses conda-forge only (`nodefaults`) and adds the compiler, `make` and FFTW needed to build ER-Decon II.
- `tests/compare_with_v09.sh`: whole-pipeline comparison with AR-Decon 0.9; `tests/make_small_test_maps.py`: a small synthetic test case; `.github/workflows/ci.yml`: builds ER-Decon II and runs the pipeline on it.
