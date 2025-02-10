#!/usr/bin/env python


import sys,math,random,numpy,struct,argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("dFSC", type = str, help='Input 1D dFSC for plot.')
parser.add_argument("fibo", type = str, help="Fibonacci points used for dFSC calculation.")
parser.add_argument("--apix", type = float, required=True, help="Pixel size (in Angstrom/pixel)")
parser.add_argument("--wedge", metavar = ('alpha', 'beta', 'gamma', 'dihedral'), nargs=4, type=float, \
                     help="Wedge parameters, where alpha, beta, gamma and dihedral are the Euler angles and \
                     dihedral angle of a wedge. \
                     E.g. [90, 0, 0, 60] represent a wedge with z-axis as an axis of symmetry, \
                     y-axis as a union line and a dihedral angle of 60 degree.")
parser.add_argument("--cone", metavar = ('alpha', 'beta', 'apex'), nargs=3, type=float, \
                     help="Cone parameters, where beta and alpha is the \
                     Euler angles of a cone, apex is the apex angle of the cone")
parser.add_argument("--xyz", metavar = ('apex'), nargs=1, type=float, \
                     help="Calculate average dFSC in XYZ direction.")
parser.add_argument("--goodbad", metavar = ('threshold', 'highres'), nargs=2, type=float, \
                     help='Threshold and resolution range to define good and bad dirctions \
                     when compared with average dFSC in all directins. By default, use 0.8 as threshold and \
                     use a resolution where average dFSC drops below 0.001 as the high resolution. \
                     E.g., [0.8, 3] means if 80 percent of dFSC values at frequencies lowe than 3 angstrom \
                     are better than average dFSC in a dirction, the direction is defined as good. \
                     Meantime, the direction where 80 percent of dFSC values are worse than average dFSC \
                     is classified as a bad direction.')
parser.add_argument('--output', help='Basename of the output file')
args = parser.parse_args()

dFSC = np.loadtxt(args.dFSC)
fibo = np.loadtxt(args.fibo)
apix = args.apix

if args.output:
    output = args.output
else:
    output = args.dFSC.replace("dFSC1d.txt","Avg")

x = np.arange(float(dFSC.shape[1])) + 1
x2 = x / (2. * dFSC.shape[1] * apix)
    
dFSC_avg = np.average(dFSC,axis=0)
plt.plot(x2, dFSC_avg, 'green', linewidth=2.0, alpha=1.0, label='average dFSC')
np.savetxt(output+'dFSC1dAll.txt',dFSC_avg,fmt="%.3f")

## # The lines that begin with "##" are used to display missing region in 3D.
## arsize = 60
## points = ((fibo + 1) * arsize//2).astype(int).transpose()
## points[points >= arsize] = arsize - 1  
## voxelarray = np.full((arsize,arsize,arsize), False)
## fccmap = numpy.empty(voxelarray.shape, dtype=object)
## for i in range(0, fibo.shape[0]):
##     voxelarray[tuple(points)] = True
##     fccmap[tuple(points)] = 'purple'

if args.wedge:
    alpha, beta, gamma, dihedral = numpy.deg2rad(args.wedge)
    # The elements of a vector is in x, y, z order.
 
    rot_matrix = R.from_euler('zyz', [alpha, beta, gamma]).as_matrix()
    print(rot_matrix)
    # New union line of the wedge after rotation
    new_y = np.dot(rot_matrix, np.array([0, 1, 0]))
    # New axis of symmetry
    new_z = np.dot(rot_matrix, np.array([0, 0, 1]))    

    fibo_new_y_proj = np.dot(new_y, fibo.T)
    fibo_new_z_proj = np.dot(new_z, fibo.T)

    # Inside the wedge: |new_y_prjection| < |new_z_projection| * tan(dihedral_half)|
    inside = np.abs(fibo_new_y_proj) < np.abs(fibo_new_z_proj * np.tan(dihedral/2))   
    outside = np.invert(inside)

    for i in range(0, fibo.shape[0]):
        if inside[i]:
            ## voxelarray[tuple(points[:,i])] = False
            plt.plot(x2, dFSC[i,:], 'blue', linewidth=1.0, alpha=0.05)
        else:
            plt.plot(x2, dFSC[i,:], 'purple', linewidth=1.0, alpha=0.05)

    dFSC_in_avg = np.mean(dFSC[inside,:], axis=0)
    dFSC_out_avg = np.mean(dFSC[outside,:], axis=0)
    plt.plot(x2, dFSC_in_avg, 'blue', linewidth=2.0, alpha=1.0, label='average dFSC inside the wedge')
    plt.plot(x2, dFSC_out_avg, 'purple', linewidth=2.0, alpha=1.0, label='average dFSC outside the wedge')
    np.savetxt(output+'dFSC1dInWedge.txt',dFSC_in_avg,fmt="%.3f")
    np.savetxt(output+'dFSC1dOutWedge.txt',dFSC_out_avg,fmt="%.3f")

if args.cone:
    alpha_cone, beta_cone, apex_cone = np.deg2rad(args.cone)
    axis_cone = np.array([np.sin(beta_cone)*np.cos(alpha_cone), np.sin(beta_cone)*np.sin(alpha_cone), np.cos(beta_cone)])
 
    proj_ref = np.abs(np.cos(apex_cone/2))
    inside = np.abs(np.dot(axis_cone, fibo.T)) > proj_ref
    outside = np.invert(inside)

    for i in range(0, fibo.shape[0]):
        if inside[i]:
            ## voxelarray[tuple(points[:,i])] = False
            plt.plot(x2, dFSC[i,:], 'blue', linewidth=1.0, alpha=0.05)
        else:
            plt.plot(x2, dFSC[i,:], 'purple', linewidth=1.0, alpha=0.05)

    dFSC_in_avg = np.mean(dFSC[inside,:], axis=0)
    dFSC_out_avg = np.mean(dFSC[outside,:], axis=0)
    plt.plot(x2, dFSC_in_avg, 'blue', linewidth=2.0, alpha=1.0, label='average dFSC inside the cone')
    plt.plot(x2, dFSC_out_avg, 'purple', linewidth=2.0, alpha=1.0, label='average dFSC outside the cone')
    np.savetxt(output+'dFSC1dInCone.txt',dFSC_in_avg,fmt="%.3f")
    np.savetxt(output+'dFSC1dOutCone.txt',dFSC_out_avg,fmt="%.3f")

if args.xyz:
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    apex = np.deg2rad(args.xyz)

    proj_ref = np.abs(np.cos(apex/2))
    inside_x = np.abs(np.dot(x, fibo.T)) > proj_ref
    inside_y = np.abs(np.dot(y, fibo.T)) > proj_ref
    inside_z = np.abs(np.dot(z, fibo.T)) > proj_ref

    for i in range(0, fibo.shape[0]):
        plt.plot(x2, dFSC[i,:], 'purple', linewidth=1.0, alpha=0.05)

    dFSC_x_avg = np.mean(dFSC[inside_x,:], axis=0)
    dFSC_y_avg = np.mean(dFSC[inside_y,:], axis=0)
    dFSC_z_avg = np.mean(dFSC[inside_z,:], axis=0)
    plt.plot(x2, dFSC_x_avg, 'purple', linewidth=2.0, alpha=1.0, label='average dFSC in X axis')
    plt.plot(x2, dFSC_x_avg, 'yellow', linewidth=1.5, alpha=1.0, label='average dFSC in y axis')
    plt.plot(x2, dFSC_z_avg, 'blue', linewidth=1.0, alpha=1.0, label='average dFSC in Z axis')
    np.savetxt(output+'dFSC1dX.txt',dFSC_x_avg,fmt="%.3f")
    np.savetxt(output+'dFSC1dY.txt',dFSC_y_avg,fmt="%.3f")
    np.savetxt(output+'dFSC1dZ.txt',dFSC_z_avg,fmt="%.3f")

if args.goodbad:
    threshold, highres = args.goodbad
    dFSC_avg_cumsum = running_mean(dFSC_avg, 3)
    if highres > 0.0:
        ind_cutoff = np.int32(2. * dFSC.shape[1] * apix / highres)
    else:
        ind_cutoff = np.argmax(dFSC_avg_cumsum < 0.001)
   
    print(ind_cutoff)
 
    bad = np.count_nonzero(dFSC[:,0:ind_cutoff] <= dFSC_avg[0:ind_cutoff], axis=1) > threshold * ind_cutoff
    good = np.count_nonzero(dFSC[:,0:ind_cutoff] >= dFSC_avg[0:ind_cutoff], axis=1) > threshold * ind_cutoff
    
    for i in range(0, fibo.shape[0]):
        if bad[i]:
            ## voxelarray[tuple(points[:,i])] = False
            plt.plot(x2, dFSC[i,:], 'blue', linewidth=1.0, alpha=0.05)
        elif good[i]:
            plt.plot(x2, dFSC[i,:], 'purple', linewidth=1.0, alpha=0.05)
        else:
            plt.plot(x2, dFSC[i,:], 'purple', linewidth=1.0, alpha=0.05)
    
    dFSC_bad_avg = np.mean(dFSC[bad,:], axis=0)
    dFSC_good_avg = np.mean(dFSC[good,:], axis=0)
    plt.plot(x2, dFSC_bad_avg, 'blue', linewidth=2.0, alpha=1.0, label='average dFSC in bad directions')
    plt.plot(x2, dFSC_good_avg, 'purple', linewidth=2.0, alpha=1.0, label='average dFSC in good directions')
    np.savetxt(output+'dFSC1dBad.txt',dFSC_bad_avg,fmt="%.3f")
    np.savetxt(output+'dFSC1dGood.txt',dFSC_good_avg,fmt="%.3f")

plt.legend(loc='best')
plt.axis([0., 1./(2.*apix), -0.1, 1])
plt.xlabel('Spatial frequency (1/Angstroms)')
plt.ylabel('dFSC')
plt.savefig(output+'dFSC1d.svg', format='svg')
plt.savefig(output+'dFSC1d.png')
   

## ax = plt.figure().add_subplot(projection='3d')
## ax.voxels(voxelarray, facecolors=fccmap)
## 
## plt.show()





