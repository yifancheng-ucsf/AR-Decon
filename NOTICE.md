# NOTICE

AR-Decon is distributed under the BSD 3-Clause License (`LICENSE`). Its deconvolution engine, ER-Decon II, is a separate package (see below).

| Component | Origin | Licence |
|---|---|---|
| `core2_decon` (not included; required) | **ER-Decon II**, from [Biochemistry-Biophysics-Cheng-Lab/ER-DECON-II](https://github.com/Biochemistry-Biophysics-Cheng-Lab/ER-DECON-II): the ER-Decon source of the Sedat and Agard laboratories (UCSF; Arigovindan *et al.*, *PNAS* 110, 17344–17349, 2013), built without Priism | BSD 3-Clause; links FFTW (GPL-2+). See that repository's `NOTICE.md` |
| `ardecon/scripts/ardecon`, `param_search`, `gen_tf_withPSF.py`, `applymask.py`, `gen_sphere_mask.py`, `core2_decon-000.com`; `tests/*` | AR-Decon (Cheng laboratory, UCSF) | BSD 3-Clause (`LICENSE`) |
| `ardecon/scripts/dfsc-0.0.1.py`, `dfsc-avg-plot.py`, `dfsc-combine.py` | distributed with AR-Decon since 0.9. `dfsc-0.0.1.py` modified in 2026-11 (`--seed`; per-cone updates at the cone voxels only, byte-identical output) | BSD 3-Clause (`LICENSE`) |
| `ardecon/scripts/Mrc.py` | Priism's `PYTHON/Mrc.py`, derived from Priithon (Sebastian Haase; changes by Lin Shao and Chris Weisiger); modified in 2026-11 for NumPy 2, with results unchanged under NumPy 1 | `ardecon/scripts/PRIITHON_LICENSE.txt` (BSD-3-style) |

**Removed (bundled in earlier versions):**
- `ardecon/bin/core2_decon` and `core2_decon_static`, the Priism-built binaries (replaced by ER-Decon II built from source).
- `ardecon/lib/*`: the Priism libraries `libive`, `libIWL`, `libWM` and `libMaskuPlugin`, plus X11/Motif/GL, FFTW and libgomp. Priism's licence does not permit redistributing the Priism libraries.
- The conda activation hooks (`etc/conda/`) that put `ardecon/lib` on `LD_LIBRARY_PATH`.
