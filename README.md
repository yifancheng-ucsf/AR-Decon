# AR-Decon: Deconvolution to restore cryo-EM maps with anisotropic resolution <!-- omit in toc -->
AR-Decon, which stands for correcting **A**nisotropic **R**esolution by **Decon**volution, is a computational pipeline designed to enhance the quality of three-dimensional maps that suffer from anisotropic resolutions, often resulting from datasets with preferred orientations. By applying advanced deconvolution techniques, AR-Decon corrects these resolution discrepancies, leading to more accurate and isotropic 3D maps.

**The deconvolution engine of AR-Decon is ER-Decon II** (entropy-regularized deconvolution; Arigovindan *et al.*, *PNAS* 2013), developed by Muthuvel Arigovindan and colleagues in the laboratories of John W. Sedat and David A. Agard at UCSF. ER-Decon II performs the central restoration step: its regularization and restoration methodology reconstructs the map. AR-Decon adds what is needed to apply it to cryo-EM maps:
- estimation of the optical transfer function (OTF) of a cryo-EM map from the directional FSC (dFSC) of its two half-maps;
- the cryo-EM processing workflow around the deconvolution: masking, running ER-Decon II with parameters suited to cryo-EM maps, optional post-processing, and a parameter search based on map-to-model dFSC.

ER-Decon II is the program `core2_decon`, built from the [ER-Decon II repository](https://github.com/Biochemistry-Biophysics-Cheng-Lab/ER-DECON-II) during installation.

> **Please cite both papers** when you use AR-Decon: the ER-Decon paper (Arigovindan *et al.*, *PNAS* 2013) for the deconvolution method, and the AR-Decon paper (Li *et al.*) for the cryo-EM OTF estimation and workflow. See [How to Cite](#how-to-cite).

## Table of Contents <!-- omit in toc -->
- [How to Cite](#how-to-cite)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Extended Usage (Post-Processing After Deconvolution)](#extended-usage-post-processing-after-deconvolution)
- [Advanced Usage (Parameter Optimization)](#advanced-usage-parameter-optimization)

## How to Cite
If you use AR-Decon, please cite **both** of the following papers:

1. **ER-Decon** (the deconvolution method and engine, ER-Decon II):
   M. Arigovindan, J. C. Fung, D. Elnatan, V. Mennella, Y.-H. M. Chan, M. Pollard, E. Branlund, J. W. Sedat and D. A. Agard, "High-resolution restoration of 3D structures from widefield images with extreme low signal-to-noise-ratio", *Proc. Natl. Acad. Sci. USA* **110**, 17344–17349 (2013). [doi:10.1073/pnas.1315675110](https://doi.org/10.1073/pnas.1315675110)

2. **AR-Decon** (cryo-EM OTF estimation and processing workflow):
   J. Li, W. Choi, Y. Chen, S. Zheng, A. McDonald, J. W. Sedat, D. A. Agard and Y. Cheng, "Deconvolution improves cryo-EM maps", *bioRxiv* (2025). [doi:10.1101/2025.02.23.639707](https://doi.org/10.1101/2025.02.23.639707)

`CITATION.cff` lists both papers. GitHub's "Cite this repository" button shows only the AR-Decon paper, so please add the ER-Decon paper as well.

## System Requirements
### 1. Hardware Requirements
AR-Decon is a CPU-based program and does not require a GPU. By default, it utilizes 4 CPU cores for parallel processing and consumes approximately 3 GB of RAM for a map with a box size of 256. For optimal performance, the following hardware specifications are recommended:
- RAM: 16 GB or more
- CPU: 8 or more cores, 4.0 GHz per core or higher

### 2. Operating System Requirements
AR-Decon requires a Linux-based operating system for proper execution. The program has been tested on the following Linux distributions:
- CentOS 7.6.1810 and 7.9.2009
- Ubuntu 20.04.6 LTS and 22.04.4 LTS

## Installation
### 1. Install Conda
Ensure that you have either Miniconda or Anaconda installed on your Linux distribution. It is recommended to install Miniconda by following these [instructions](https://docs.anaconda.com/miniconda/).
### 2. Set Up a Python Environment and Install ER-Decon II and AR-Decon
Create a Python environment named `ardecon` using `conda`, build ER-Decon II (`core2_decon`) into it, and install AR-Decon. Installation typically completes within 10 minutes.
```bash
git clone https://github.com/yifancheng-ucsf/AR-Decon.git
git clone https://github.com/Biochemistry-Biophysics-Cheng-Lab/ER-DECON-II.git
cd AR-Decon
conda env create -f environment.yml
conda activate ardecon
make -C ../ER-DECON-II -j install PREFIX=$CONDA_PREFIX FFTW_INC=$CONDA_PREFIX/include FFTW_LIB=$CONDA_PREFIX/lib
make -C ../ER-DECON-II check          # optional, a few seconds; must end with "ALL PASSED"
pip install .
```
AR-Decon uses the first `core2_decon` on `PATH`. See the [ER-Decon II README](https://github.com/Biochemistry-Biophysics-Cheng-Lab/ER-DECON-II#readme) for other ways to build it.
### 3. Reactivate the environment
After installing AR-Decon, **reactivate the environment** to ensure it is fully set up:
```bash
conda deactivate && conda activate ardecon
```
**Note**: AR-Decon is now integrated into [SBGrid](https://sbgrid.org/software/titles/ar-decon) and [COSMIC<sup>2</sup>](https://cosmic2.sdsc.edu:8443/gateway/tools.action). Users can also access AR-Decon through either of these platforms without the need for a separate installation.

## Quick Start
To run AR-Decon, use the command: **`ardecon`**.  **It requires two half-maps and a full map as input**. The full map can be either sharpened or unsharpened — you may experiment with both to see which produces better results. Using a soft-edged mask is recommended for optimal performance. A common usage example is shown below:
```bash
ardecon half-map1.mrc half-map2.mrc full-map.mrc --mask mymask.mrc
```
The runtime of the command scales with the map’s box size. As a reference, processing a map with a box size of 256 typically takes approximately 5 minutes. After the command completes, the deconvolution result can be found in the `ardecon` directory (can be customized, see [Advanced Usage](#advanced-usage-parameter-optimization)), with a filename ending in `_Decon.mrc`.

**Note**: For detailed instructions with comprehensive examples, refer to the [**Tutorial**](tutorial/Tutorial.md).

## Extended Usage (Post-Processing After Deconvolution)
The map after deconvolution can be post processed using Phenix (`phenix.auto_sharpen`) , bfactor (`bfactor.exe`), or EMReady (`EMReady.sh`). However, if the map is too noisy, sharpening after deconvolution may not be effective. Based on our experience, AR-Decon works well with EMReady, as demonstrated in our paper.

### 1. Install Post-Processing Tools
These tools are not bundled with AR-Decon and must be installed separately. You can download them from their official websites:
- [Phenix](https://phenix-online.org/documentation/install-setup-run.html)
- [bfactor](https://grigoriefflab.umassmed.edu/bfactor)
- [EMReady](http://huanglab.phys.hust.edu.cn/EMReady/)

### 2. Set Up the Environment
Before AR-Decon can call these tools, their environment must be properly set up. For example, to use Phenix, run the following command (replace `path_to_phenix` with your actual installation path):
```bash
source /path_to_phenix/phenix_env.sh
```
### 3. Run Post-Processing
If deconvolution has already been performed, you can run post-processing directly without repeating the deconvolution step. Example command using Phenix:
```bash
ardecon half-map1.mrc half-map2.mrc full-map.mrc --mask mymask.mrc --post phenix
```
Once successful, the sharpened map will be saved in the `ardecon` directory with a filename ending in `_phenix.mrc`.

Similarly, post-processing with bfactor and EMReady follows the same procedure, with the only difference being the required environment setup for each tool.

**Note**: Based on our experience, DeepEMhancer does not work well with maps after deconvolution.

## Advanced Usage (Parameter Optimization)
Additional options for `ardecon` can be displayed using the `--help` flag:
```bash
ardecon --help
```
```text
Usage: ardecon half-map1.mrc half-map2.mrc full-map.mrc [options]

Required parameters:
  half-map1.mrc            First half-map file
  half-map2.mrc            Second half-map file
  full-map.mrc             Full map file

Optional parameters:
  --mask      mask.mrc     Apply the specified mask file during dFSC calculation and deconvolution.
  --sigma     1.0          Sigma value for generating OTF (default: auto-calculated).
  --smooth    0.5          Smoothing parameter for deconvolution (default: 0.5).
  --nonlin    10000        Nonlinearity parameter for deconvolution (default: 10000).
  --iter      50           Number of deconvolution cycles (default: 50).
  --post      method       Post-processing method (requires installation): phenix | bfactor | emready.
  --outdir    directory    Directory to save output results (default: 'ardecon').
  --seed      N            Seed for the random cone directions of the dFSC step (default: not seeded).
  --help                   Display this help message and exit.

Example:
  ardecon half-map1.mrc half-map2.mrc full-map.mrc --mask mymask.mrc --post phenix
```
Among these options, `--smooth` (smoothing) and `--nonlin` (nonlinearity) are two key parameters that can be tuned for better deconvolution results. The default values (`--smooth 0.5`, `--nonlin 10000`) generally work well, but if the map after deconvolution does not look optimal, a grid search can be performed to identify better parameters.
### 1. Generate a Reference Map
Deconvolution results are evaluated using map-to-model dFSC, so a **reference map derived from a model is required**. You can generate a reference map using the [`molmap`](https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/midas/molmap.html) command in UCSF Chimera.
### 2. Perform a Grid Search
To perform a grid search for smoothing and nonlinearity parameters, use the command:`param_search`. It requires three files as input, a full-map, a reference map, and an OTF file generated by `ardecon`. 
```bash
param_search full-map.mrc reference.mrc HalfMapdFSC3d.otf --mask mymask.mrc 
```
This command searches within the default parameter grid:
- **Smoothing parameters**: `"5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1 1 2 5 1e1 2e1 5e1 1e2"`
- **Nonlinearity parameters**: `"1 1e1 1e2 1e3 1e4 1e5 1e6 1e7"`

If you're unsure where to start, consider running a smaller search set first to get a better understanding:
```bash
param_search full-map.mrc reference.mrc HalfMapdFSC3d.otf \
      --mask mymask.mrc --smooth "5e-1 1 2 5 1e1" --nonlin "1 1e1 1e2"
```
### 3. Review the results
After running the grid search, check the `Combined_dFSC1d.png` file in the `ParamSearch` directory.
- If all dFSC curves appear similar, consider expanding the search grid by running `param_search` with a larger parameter set.
- `param_search` automatically skips previously computed combinations, ensuring efficiency.
- If the search space is large enough, you may observe different dFSC curve patterns for maps deconvolved with different parameters.

Choose optimal parameters.
1. Eliminate parameter regions where dFSC curves appear unreasonable. 
2. Select parameters that reduce dFSC spread and produce a better averaged dFSC curve.
3. Locate the corresponding deconvolution results in the `ParamSearch/Deconvolution/` directory and visually inspect the maps.


### 4. Run AR-Decon with Optimal Parameters
Once you have identified the optimal parameters (e.g., smoothing: `5e-1`, nonlinearity: `1e3`), you can rerun AR-Decon using these values:
```bash
ardecon half-map1.mrc half-map2.mrc full-map.mrc --mask mymask.mrc \
    --smooth 5e-1 --nonlin 1e3 --post phenix --outdir ardecon_new
```
This command applies the optimized parameters to the original full map. These parameters may also be used for new 3D reconstructions from the same dataset.