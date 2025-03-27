# AR-Decon Tutorial <!-- omit in toc -->
## Table of Contents <!-- omit in toc -->
- [Introduction](#introduction)
- [Setting Up AR-Decon](#setting-up-ar-decon)
- [Data Preparation](#data-preparation)
  - [Downloading Tutorial Data](#downloading-tutorial-data)
  - [Understanding Raw Data](#understanding-raw-data)
  - [Preprocessed Data Explanation](#preprocessed-data-explanation)
- [Running AR-Decon](#running-ar-decon)
  - [Basic Usage with a Mask](#basic-usage-with-a-mask)
  - [Post-processing The Results](#post-processing-the-results)
- [Optimizing Parameters](#optimizing-parameters)
  - [Running Grid Search](#running-grid-search)
  - [Run AR-Decon with Optimal Parameters](#run-ar-decon-with-optimal-parameters)
- [Final Directory Structure](#final-directory-structure)
- [Troubleshooting](#troubleshooting)

## Introduction
AR-Decon (**A**nisotropic **R**esolution **Decon**volution) is a tool for correcting anisotropic resolutions in cryo-EM maps. This tutorial walks you through how to use AR-Decon to improve the quality of maps with anisotropic resolutions.

In this tutorial, we'll use an influenza hemagglutinin (HA) trimer as our example dataset. You'll learn how to:
* Set up AR-Decon
* Prepare data correctly
* Run AR-Decon and post-processing
* Optimize parameters for better results


## Setting Up AR-Decon
Before starting, make sure you have `AR-Decon` installed following the [GitHub instruction](https://github.com/yifancheng-ucsf/AR-Decon), and activated the python environment `ardecon`. If you see `(ardecon)` in front of your Linux BASH prompt, you are all set. Otherwise, you need to run the following command to activate `ardecon` environment.
```bash
conda activate ardecon
```

## Data Preparation
### Downloading Tutorial Data
First, let's download and extract the [tutorial dataset from Zenodo](https://zenodo.org/records/15071063):
```bash
wget https://zenodo.org/records/15071063/files/AR-Decon_Tutorial.zip
unzip AR-Decon_Tutorial.zip
cd AR-Decon_Tutorial
```

After extraction, you'll have the following directory structure:
```text
AR-Decon_Tutorial/
├── PreProcess/          # Preprocessed maps ready for AR-Decon
└── RawData/             # Original maps from EMDB
```

> **Note:** In the directory `AR-Decon_Tutorial`, all data required by deconvolution is ready. If you only want to practice running AR-Decon, you can skip to the "[Running AR-Decon](#running-ar-decon)" section.

### Understanding Raw Data
The tutorial uses cryo-EM maps from the Electron Microscopy Data Bank (EMDB) with accessing number [8731](https://www.ebi.ac.uk/emdb/EMD-8731) and [21954](https://www.ebi.ac.uk/emdb/EMD-21954). We include these maps in the `RawData/` folder:
- `emd_8731.mrc`: The target map showing anisotropic resolution that we want to improve.
- `emd_8731_half_map_1.mrc` and `emd_8731_half_map_2.mrc`: Two half-maps from 3D refinement.
- `emd_8731_msk_1.mrc`: A mask used during refinement.
- `emd_21954.mrc`: A high-quality map without anisotropic resolution (our ground truth).

### Preprocessed Data Explanation
For AR-Decon to work properly, all maps must have the same handedness and dimensions. The `PreProcess/` folder contains maps that have already undergone two processing steps:

1. **Correctly flipped for consistent handedness**  
   The two half-maps and the mask have different handedness from `emd_8731.mrc`, they are flipped using UCSF Chimera command `vop zflip #vol_num` (replace `vol_num` with the actual volume number). 
2. **Resampled to ensure matching pixel size and box dimensions**  
   `emd_21954.mrc` has a different pixel size and box dimensions than `emd_8731.mrc`. To address this, first load `emd_8731.mrc` and `emd_21954.mrc` sequentially in UCSF Chimera. Then fit `emd_21954.mrc` to `emd_8731.mrc` using the command `fitmap #1 #0`. Next, resample `emd_21954.mrc` onto the grid of `emd_8731.mrc` with the command `vop resample #1 on #0`. Finally, save the resampled map as `emd_21954_FitPS1p31Box256.mrc`.

The prepared files include:
- `emd_8731_half_map_1_RightH.mrc` and `emd_8731_half_map_2_RightH.mrc`: Half-maps with corrected handedness. These two maps are used to generate optical transfer function (OTF) for deconvolution.
- `emd_8731_msk_1_RightH.mrc`: Mask with corrected handedness.
- `emd_21954_FitPS1p31Box256.mrc`: Reference map resampled to match our target map. This map is used to calculate map to map dFSC in the parameter optimization step.


## Running AR-Decon
### Basic Usage with a Mask
Let's run AR-Decon with a mask on our example dataset:
```bash
# Make sure you're in the tutorial directory
cd AR-Decon_Tutorial/

# Run AR-Decon with a mask
ardecon PreProcess/emd_8731_half_map_1_RightH.mrc \
        PreProcess/emd_8731_half_map_2_RightH.mrc \
        RawData/emd_8731.mrc \
        --mask PreProcess/emd_8731_msk_1_RightH.mrc
```
This command:
1. Takes two half-maps as the first two arguments
2. Takes the target map to be deconvolved as the third argument
3. Uses a mask to focus the deconvolution on the region of interest

After completion, check the `ardecon` directory for the map after deconvolution `emd_8731_Masked_Decon.mrc`. 

Additionally, the half-map dFSC plot `emd_8731_Masked_HalfMapdFSC1d.png` and its vector image version `emd_8731_Masked_HalfMapdFSC1d.svg` are also available in the directory for your reference. 

### Post-processing The Results
This tutorial assumes the post-processing tools Phenix (`phenix.auto_sharpen`), bfactor (`bfactor.exe`), or EMReady (`EMReady.sh`) are installed in the directory `/opt`. If you install them in a different directory, you'll need to update the following commands accordingly. 

> **Note:** You don't need to run all these commands. Choose the post-processing tool you prefer or have available.

```bash
# Option 1: Post-processing with Phenix
source /opt/phenix-1.18.2-3874/phenix_env.sh
ardecon PreProcess/emd_8731_half_map_1_RightH.mrc \
        PreProcess/emd_8731_half_map_2_RightH.mrc \
        RawData/emd_8731.mrc \
        --mask PreProcess/emd_8731_msk_1_RightH.mrc --post phenix

# Option 2: Post-processing with bfactor
export PATH=/opt/bfactor/:$PATH
ardecon PreProcess/emd_8731_half_map_1_RightH.mrc \
        PreProcess/emd_8731_half_map_2_RightH.mrc \
        RawData/emd_8731.mrc \
        --mask PreProcess/emd_8731_msk_1_RightH.mrc --post bfactor

# Option 3: Post-processing with EMReady
export PATH=/opt/EMReady/:$PATH
ardecon PreProcess/emd_8731_half_map_1_RightH.mrc \
        PreProcess/emd_8731_half_map_2_RightH.mrc \
        RawData/emd_8731.mrc \
        --mask PreProcess/emd_8731_msk_1_RightH.mrc --post emready
```
The post-processing will skip the deconvolution step if it has already been completed. Otherwise, it will run deconvolution first and then perform post-processing. 

Once the post-processing job finishes, the resulting map will be saved in the `ardecon` directory with a filename ending in `_phenix.mrc`, `_bfactor.mrc`, or `_emready.mrc`, depending on which method you used.

## Optimizing Parameters
To get the best results from AR-Decon, you can optimize the smoothing and nonlinearity parameters. We will use `emd_21954_FitPS1p31Box256.mrc` as the reference map and the OTF `emd_8731_Masked_HalfMapdFSC3d.otf` generated in "[Basic Usage with a Mask](#basic-usage-with-a-mask)" section to perform grid search. 

### Running Grid Search
To save time, the grid search is demonstrated with a small search set, where smoothing parameter set is `"5e-1 1 2"` and nonlinearity parameter set is `"1 1e1 1e2"`.
```bash
cd ardecon
param_search ../RawData/emd_8731.mrc \
             ../PreProcess/emd_21954_FitPS1p31Box256.mrc \
             emd_8731_Masked_HalfMapdFSC3d.otf \
             --mask ../PreProcess/emd_8731_msk_1_RightH.mrc \
             --smooth "5e-1 1 2" --nonlin "1 1e1 1e2"
```

This command:
1. Tests deconvolution with 9 combinations of parameters (3 smoothing × 3 nonlinearity values)
2. Calculates map to map dFSC between the reference map and map after deconvolution
3. Creates comparison plots to help you choose the best parameters

After running, open `ParamSearch/Combined_dFSC1d.png` to view the results. Look for parameter combinations that:
* Reduce the spread of the directional FSC curves
* Give the best overall average FSC curve

The corresponding map after deconvolution using the selected parameter can be found in the `ParamSearch/Deconvolution/` directory for visual inspection. 

### Run AR-Decon with Optimal Parameters
Once you've identified the best parameters, run AR-Decon again with those values (e.g., smoothing: `5e-1`, nonlinearity: `1e2`):
```bash
cd AR-Decon_Tutorial/
ardecon PreProcess/emd_8731_half_map_1_RightH.mrc \
        PreProcess/emd_8731_half_map_2_RightH.mrc \
        RawData/emd_8731.mrc \
        --mask PreProcess/emd_8731_msk_1_RightH.mrc \
        --smooth 5e-1 --nonlin 1e2 --post phenix --outdir ardecon_5e-1_1e2
```
This command creates a new directory `ardecon_5e-1_1e2` with your optimized results.

## Final Directory Structure
After completing this tutorial, you should have the following folder structure:
```text
AR-Decon_Tutorial/
├── ardecon/             # Directory for default outputs of AR-Decon 
│   └── ParamSearch/     # Parameter optimization results
├── ardecon_5e-1_1e2/    # Results of AR-Decon with specified parameters
├── PreProcess/          # Preprocessed maps ready for AR-Decon
└── RawData/             # Original maps from EMDB
```

## Troubleshooting
Common issues and solutions:
* **Environment activation fails**: Make sure you've installed AR-Decon following the GitHub instructions
* **"Command not found" errors**: Check that the conda environment is activated
* **Directory not found**: Make sure you're in the correct working directory
* **Post-processing fails**: Verify that the post-processing tools are installed at the specified paths