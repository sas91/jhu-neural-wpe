#!/usr/bin/env python

#  2018 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from load_data import save_list
import argparse

parser = argparse.ArgumentParser(description='Create REVERB list')
parser.add_argument('--data_dir', default='wav/REVERB_WSJCAM0_tr/data/mc_train',
                    help='Base directory of the REVERB challenge parallel simulation data.')
parser.add_argument('--dest_dir', default='data',
                    help='Base directory to store the list.')
args = parser.parse_args()

save_list(args.data_dir, args.dest_dir)
