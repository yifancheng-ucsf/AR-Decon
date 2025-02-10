#!/usr/bin/env python

import numpy as np
import os, sys
from ardecon.scripts import Mrc

import warnings
warnings.filterwarnings("ignore")

def main(args):
    # Read map using Mrc.py and the returned numpy array is C_CONTIGUOUS, and
    # the index order is (sections, y, x) with the last index changing fastest
    map_raw = Mrc.bindFile(args.map)
    hdr = map_raw.Mrc.hdr
    spacing = map_raw.Mrc.hdr.getSpacing()
    # hdr.Num is the number of x samples, number of y samples, and number of sections
    xn, yn, zn = (hdr.Num[0]+1)//2, (hdr.Num[1]+1)//2, (hdr.Num[2]+1)//2
    map_data = np.copy(map_raw.Mrc.data)

    ind_nonzero = np.where(map_data > args.threshold)
    x_max, x_min = np.max(ind_nonzero[2]), np.min(ind_nonzero[2])
    y_max, y_min = np.max(ind_nonzero[1]), np.min(ind_nonzero[1])
    z_max, z_min = np.max(ind_nonzero[0]), np.min(ind_nonzero[0])
    # print(x_max, x_min)
    x_center = (x_max + x_min) // 2
    y_center = (y_max + y_min) // 2
    z_center = (z_max + z_min) // 2

    x_length = x_max - x_min
    y_length = y_max - y_min
    z_length = z_max - z_min

    r = np.sqrt(x_length**2 + y_length**2 + z_length**2) / 2 + args.extend

    z, y, x = np.meshgrid(np.arange(hdr.Num[2]), np.arange(hdr.Num[1]), np.arange(hdr.Num[0]), indexing='ij')
    dist = np.sqrt((z-z_center)**2 + (y-y_center)**2 + (x-x_center)**2)
    sphere = np.zeros(tuple(hdr.Num))
    sphere[dist <= r] = 1
    edge_ind = np.logical_and(dist>r, dist<=r+args.width)
    sphere[edge_ind] = np.cos(0.5 * np.pi * (dist[edge_ind] - r)/args.width)
    # print(sphere[edge_ind])

    Mrc.save(sphere.astype(np.float32), args.output, ifExists='overwrite', hdr=hdr)
 



    # psf_tf = np.fft.fftn(np.fft.ifftshift(psf))/(hdr.Num[0]*hdr.Num[1]*hdr.Num[2])
    # psf_tf_masked = np.complex64(psf_tf * np.fft.ifftshift(mask))
    # # psf_tf_masked = np.float32(np.real(psf_tf * np.fft.ifftshift(mask)))
    # tf_half = psf_tf_masked[:,:,0:xn+1]
    # mtf = np.abs(tf_half)
    # min_mtf = np.min(mtf)
    # max_mtf = np.max(mtf)
    # mean_mtf = np.mean(mtf)
    # tf_mrc = Mrc.Mrc2(args.output, mode='w')
    # tf_mrc.initHdrForArr(tf_half)
    # tf_mrc.hdr._setmmm1([min_mtf,max_mtf,mean_mtf])
    # tf_mrc.hdr._setwave(0)
    # tf_mrc.hdr.setSpacing(spacing[0], spacing[1], spacing[2])
    # tf_mrc.writeHeader()
    # tf_mrc.writeStack(tf_half)
    # tf_mrc.close()
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser=argparse.ArgumentParser(description='Generate a shpere mask with soft edge')
    parser.add_argument('map',help='Input a map')
    parser.add_argument('-t', '--threshold', type=float, help="Threshold for displaying the map")
    parser.add_argument('-e', '--extend', type=float, default=0, help="Extend radius of the sphere")
    parser.add_argument('-w', '--width', type=float, default=0, help="Width of edge (in pixel)")
    parser.add_argument('-o', '--output', default='SphereMask.mrc', help="Name of the shpere mask")
    sys.exit(main(parser.parse_args()))

