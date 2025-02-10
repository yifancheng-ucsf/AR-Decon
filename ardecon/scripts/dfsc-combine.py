#!/usr/bin/env python

import os,sys,math,random,numpy,struct,argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from ardecon.scripts import Mrc

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the data files")
parser.add_argument("--smooth_set", type=str, required=True, help="Smoothing set values separated by spaces")
parser.add_argument("--nonlin_set", type=str, required=True, help="Nonlinearity set values separated by spaces")
parser.add_argument("--cone", metavar = ('alpha', 'beta', 'apex'), nargs=3, type=float, \
                     help="Cone parameters, where beta and alpha is the \
                     Euler angles of a cone, apex is the apex angle of the cone")
parser.add_argument('--output', help='Basename of the combined dFSC plot', default="Combined")
args = parser.parse_args()

if args.output:
    output = args.output
else:
    output = "Combined"

input_dir = args.input_dir
smooth_set = args.smooth_set.split()
nonlin_set = args.nonlin_set.split()

nrows = len(smooth_set)
ncols = len(nonlin_set)
fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols,3*nrows))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.2)
# fig.tight_layout()

# Get pixel size from a 3D dFSC file
dFSC3d_mrc = os.path.join(input_dir, smooth_set[0]+"_"+nonlin_set[0]+'_dFSC3d.mrc')
dFSC3d = Mrc.bindFile(dFSC3d_mrc)
apix = dFSC3d.Mrc.hdr.getSpacing()[0]

if args.cone:
    alpha_cone, beta_cone, apex_cone = np.deg2rad(args.cone)
    axis_cone = np.array([np.sin(beta_cone)*np.cos(alpha_cone), np.sin(beta_cone)*np.sin(alpha_cone), np.cos(beta_cone)])

for row in range(len(smooth_set)):
    for col in range(len(nonlin_set)):
        param_text = smooth_set[row]+"_"+nonlin_set[col]        

        dFSC = np.loadtxt(os.path.join(input_dir, param_text+'_dFSC1d.txt'))
        fibo = np.loadtxt(os.path.join(input_dir, param_text+'_FibonacciPoints.txt'))

        x = np.arange(float(dFSC.shape[1])) + 1
        x2 = x / (2. * dFSC.shape[1] * apix)

        # Plot the average dFSC
        dFSC_avg = np.average(dFSC,axis=0)
        axs[row, col].plot(x2, dFSC_avg, 'green', linewidth=2.0, alpha=1.0, label='average dFSC')

        if args.cone:
            proj_ref = np.abs(np.cos(apex_cone/2))
            inside = np.abs(np.dot(axis_cone, fibo.T)) > proj_ref
            outside = np.invert(inside)
        
            for i in range(0, fibo.shape[0]):
                if inside[i]:
                    ## voxelarray[tuple(points[:,i])] = False
                    axs[row, col].plot(x2, dFSC[i,:], 'blue', linewidth=1.0, alpha=0.05)
                else:
                    axs[row, col].plot(x2, dFSC[i,:], 'purple', linewidth=1.0, alpha=0.05)

            dFSC_in_avg = np.mean(dFSC[inside,:], axis=0)
            dFSC_out_avg = np.mean(dFSC[outside,:], axis=0)
            axs[row, col].plot(x2, dFSC_in_avg, 'blue', linewidth=2.0, alpha=1.0, label='average dFSC inside the cone')
            axs[row, col].plot(x2, dFSC_out_avg, 'purple', linewidth=2.0, alpha=1.0, label='average dFSC outside the cone')
        else:
            for i in range(0, fibo.shape[0]):
                axs[row, col].plot(x2, dFSC[i,:], 'purple', linewidth=1.0, alpha=0.05)

        axs[row, col].set_title(param_text)
        axs[row, col].set_xlim(left=0, right=1./(2.*apix))
        axs[row, col].set_ylim(bottom=-0.1, top=1)

plt.savefig(output+'_dFSC1d.svg', format='svg')
plt.savefig(output+'_dFSC1d.png')
