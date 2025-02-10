#!/usr/bin/env python

import numpy as np
import os.path
from ardecon.scripts import Mrc
import argparse

import warnings
warnings.filterwarnings("ignore")


def save(data, header, name):
    Mrc.save(data, name, ifExists='overwrite', hdr=header)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("mrc", help="Input mrc file")
parser.add_argument("--mask", help="Input mask MRC file")
parser.add_argument("--add", type=float, help="Add a constant to the map")
parser.add_argument("--trim", type=float, help="Trim the map at given value")

args = parser.parse_args()

original = Mrc.bindFile(args.mrc)
orig_data = original
# print(orig_data.shape)
save_name = os.path.basename(args.mrc)

if args.mask is not None:
    mask = Mrc.bindFile(args.mask)
    orig_data_masked = orig_data * mask
    save_name = save_name.replace(".mrc","_Masked.mrc")
    if args.add is not None:
        orig_data_masked_added = orig_data_masked + args.add
        add_string = str(args.add).replace(".", "p")
        save_name = save_name.replace(".mrc", "_Add"+add_string+".mrc")
        if args.trim is not None:
            orig_data_masked_added_trimed = np.copy(orig_data_masked_added)
            orig_data_masked_added_trimed[orig_data_masked_added_trimed < args.trim] = args.trim
            trim_string = str(args.trim).replace(".", "p")
            save_name = save_name.replace(".mrc", "_Trim"+trim_string+".mrc")
            save(orig_data_masked_added_trimed, original.Mrc.hdr, save_name)
        else:
            save(orig_data_masked_added, original.Mrc.hdr, save_name)
    else:
        if args.trim is not None:
            orig_data_masked_trimed = np.copy(orig_data_masked)
            orig_data_masked_trimed[orig_data_masked_trimed < args.trim] = args.trim
            trim_string = str(args.trim).replace(".", "p")
            save_name = save_name.replace(".mrc", "_Trim"+trim_string+".mrc")
            save(orig_data_masked_trimed, original.Mrc.hdr, save_name)
        else:
            save(orig_data_masked, original.Mrc.hdr, save_name)
else:
    if args.add is not None:
        orig_data_added = orig_data + args.add
        add_string = str(args.add).replace(".", "p")
        save_name = save_name.replace(".mrc", "_Add"+add_string+".mrc")
        if args.trim is not None:
            orig_data_added_trimed = np.copy(orig_data_added)
            orig_data_added_trimed[orig_data_added_trimed < args.trim] = args.trim
            trim_string = str(args.trim).replace(".", "p")
            save_name = save_name.replace(".mrc", "_Trim"+trim_string+".mrc")
            save(orig_data_added_trimed, original.Mrc.hdr, save_name)
        else:
            save(orig_data_added, original.Mrc.hdr, save_name)
    else:
        if args.trim is not None:
            orig_data_trimed = np.copy(orig_data)
            orig_data_trimed[orig_data_trimed < args.trim] = args.trim
            trim_string = str(args.trim).replace(".", "p")
            save_name = save_name.replace(".mrc", "_Trim"+trim_string+".mrc")
            save(orig_data_trimed, original.Mrc.hdr, save_name)
        else:
            print("Nothing was done.\n")
            parser.print_help()




