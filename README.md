# AR-Decon: Deconvolution to restore cryo-EM maps with anisotropic resolution
AR-Decon, which stands for correcting **A**nisotropic **R**esolution by **Decon**volution, is a computational pipeline designed to enhance the quality of three-dimensional maps that suffer from anisotropic resolutions, often resulting from datasets with preferred orientations. By applying advanced deconvolution techniques, AR-Decon corrects these resolution discrepancies, leading to more accurate and isotropic 3D maps.

## Installation
### 1. Install Conda
AR-Deon is a Linux-based program. Ensure that you have either Miniconda or Anaconda installed on your Linux distribution. It is recommended to install Miniconda by following these [instructions](https://docs.anaconda.com/miniconda/).
### 2. Set Up a Python Environment and Install AR-Decon
Create a Python environment named `ardecon using `conda` and install AR-Decon.
```bash
git clone git@github.com:yifancheng-ucsf/AR-Decon.git
cd AR-Decon
conda env create -f environment.yml
conda activate ardecon
pip install .
```
### 3. Reactivate the environment
After installing AR-Decon, **reactivate the environment** to ensure it is fully set up:
```bash
conda deactivate && conda activate ardecon
```

## Quick Start
To run AR-Decon, use the command: **`ardecon`**.  **It requires two half-maps and a full map as input**. The full map can be either sharpened or unsharpened — you may experiment with both to see which produces better results. Using a soft-edged mask is recommended for optimal performance. A common usage example is shown below:
```bash
ardecon half-map1.mrc half-map2.mrc full-map.mrc --mask mymask.mrc
```
After the command completes, the deconvolution result can be found in the `ardecon` directory (can be customized, see [Advanced Usage](#advanced-usage-parameter-optimization)), with a filename ending in `_Decon.mrc`.

## Extended Usage (Post-Processing After Deconvolution)
The map after deconvolution can be post processed using Phenix (`phenix.auto_sharpen`) , bfactor (`befactor.exe`), or EMReady (`EMReady.sh`). However, if the map is too noisy, sharpening after deconvolution may not be effective. Based on our experience, AR-Decon works well with EMReady, as demonstrated in our paper.

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