#!/usr/bin/env python

#  2018 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os


def save_list(data_dir, dest_dir):
    flist = list()
    flist_dt = list()
    for snr in ['5dB', '10dB', '20dB', '100dB']:
        flist_subdir = os.listdir(os.path.join(data_dir, snr,
                                  'primary_microphone', 'si_tr'))
        flist_subdir_dt = flist_subdir[-4:]
        flist_subdir_tr = flist_subdir[:-4]
        for subdir in flist_subdir_tr:
            flist_temp = os.listdir(os.path.join(data_dir, snr,
                                    'primary_microphone', 'si_tr', subdir))
            flist_ext = [i for i in flist_temp if i.endswith('ch1.NLR.wav')]
            flist_clean = [os.path.join(data_dir, snr, 'primary_microphone',
                           'si_tr', subdir, i.replace('_ch1.NLR.wav', ''))
                           for i in flist_ext]
            flist += flist_clean
        for subdir in flist_subdir_dt:
            flist_temp = os.listdir(os.path.join(data_dir, snr,
                                    'primary_microphone', 'si_tr', subdir))
            flist_ext = [i for i in flist_temp if i.endswith('ch1.NLR.wav')]
            flist_clean = [os.path.join(data_dir, snr, 'primary_microphone',
                           'si_tr', subdir, i.replace('_ch1.NLR.wav', ''))
                           for i in flist_ext]
            flist_dt += flist_clean
    filename = os.path.join(dest_dir, 'list_tr')
    with open(filename, 'w') as fid:
        for f in flist:
            fid.write(f + '\n')
    filename = os.path.join(dest_dir, 'list_dt')
    with open(filename, 'w') as fid:
        for f in flist_dt:
            fid.write(f + '\n')
