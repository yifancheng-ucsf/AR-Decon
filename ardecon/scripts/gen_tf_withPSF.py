#!/usr/bin/env python

import numpy as np
import os, sys
from ardecon.scripts import Mrc

import warnings
warnings.filterwarnings("ignore")

def main(args):
    # Read psf and dfsc using Mrc.py and the returned numpy array is C_CONTIGUOUS, and
    # the index order is (sections, y, x) with the last index changing fastest
    mask_raw = Mrc.bindFile(args.mask)
    hdr = mask_raw.Mrc.hdr
    spacing = mask_raw.Mrc.hdr.getSpacing()
    # hdr.Num is the number of x samples, number of y samples, and number of sections
    xn, yn, zn = (hdr.Num[0]+1)//2, (hdr.Num[1]+1)//2, (hdr.Num[2]+1)//2
    mask = np.copy(mask_raw.Mrc.data)

    sigma = 1
    resolution = 3
    if args.fsc:
        fsc = np.loadtxt(args.fsc)
        fsc_len = len(fsc)
        for i in range(fsc_len - 1, 0, -1):  # Iterate from the end of the array
            if fsc[i] > 0.143:
                sigma = round(0.225 * 2. * fsc_len / (i+1), 3)
                resolution = round(spacing[0] * 2. * fsc_len / (i+1), 3)
                break
    if args.sigma:  # If sigma is provided, use the provided sigma
        sigma = args.sigma
    print(f"sigma: {sigma}\nresolution: {resolution}")

    if args.psf:
        psf = Mrc.bindFile(args.psf)
        print("psf_flags\n", psf.flags)
    else:
        z, y, x = np.meshgrid(np.arange(-zn, zn), np.arange(-yn, yn), np.arange(-xn, xn))
        r = np.sqrt(z**2 + y**2 + x**2)
        gaussian_3d = np.float32(1.0 / (np.sqrt(2*np.pi) * sigma)**3 * np.exp( - r**2 / (2 * sigma**2))) 
        psf = gaussian_3d
        # Mrc.save(psf, f"PSF_gaussian3D_sigma{sigma}.mrc", ifExists='overwrite')
    
    d = 3
    mask[zn-d:zn+d+1, yn-d:yn+d+1, xn-d:xn+d+1] = 1.0

    psf_tf = np.fft.fftn(np.fft.ifftshift(psf))/(hdr.Num[0]*hdr.Num[1]*hdr.Num[2])
    psf_tf_masked = np.complex64(psf_tf * np.fft.ifftshift(mask))
    # psf_tf_masked = np.float32(np.real(psf_tf * np.fft.ifftshift(mask)))
    tf_half = psf_tf_masked[:,:,0:xn+1]
    mtf = np.abs(tf_half)
    min_mtf = np.min(mtf)
    max_mtf = np.max(mtf)
    mean_mtf = np.mean(mtf)
    tf_mrc = Mrc.Mrc2(args.output, mode='w')
    tf_mrc.initHdrForArr(tf_half)
    tf_mrc.hdr._setmmm1([min_mtf,max_mtf,mean_mtf])
    tf_mrc.hdr._setwave(0)
    tf_mrc.hdr.setSpacing(spacing[0], spacing[1], spacing[2])
    tf_mrc.writeHeader()
    tf_mrc.writeStack(tf_half)
    tf_mrc.close()
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--psf", help="Input an MRC file as a PSF template, default is a 3d Gaussian function with sigma equals 1")
    parser.add_argument("--mask", help="Input a 3D dFSC MRC file as a mask to create an OTF")
    parser.add_argument("--sigma", type=float, help="Sigma of 3D Gaussian function, overwrite the sigma calculated from dFSC")
    parser.add_argument("--fsc", type=str, help="Text file of average dFSC of two half-maps")
    parser.add_argument("--output", default='TF.otf', help="Output transfer function")
    sys.exit(main(parser.parse_args()))

