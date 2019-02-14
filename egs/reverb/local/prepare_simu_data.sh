#!/bin/bash
#
# Copyright 2018 Johns Hopkins University (Author: Shinji Watanabe)
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0
# This script is adapted from data preparation scripts in the Kaldi reverb recipe
# https://github.com/kaldi-asr/kaldi/tree/master/egs/reverb/s5/local

# Begin configuration section.
wavdir=${PWD}/wav
# End configuration section
. ../../tools/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <reverb-dir> <wsjcam0-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora5/REVERB_2014/REVERB /export/corpora3/LDC/LDC95S24/wsjcam0"
  exit 1
fi

set -e -o pipefail

reverb=$1
wsjcam0=$2

# tool directory
tooldir=${PWD}/data/local/reverb_tools

# working directory
dir=${PWD}/data/
mkdir -p ${dir}

# make a one dot file for train, dev, and eval data
# the directory structure of WSJCAM0 is not consistent and we need such process for each task
cat ${wsjcam0}/data/primary_microphone/etc/si_dt*.dot | sort > ${dir}/dt.dot
cat ${wsjcam0}/data/*/si_et*/*/*.dot | sort > ${dir}/et.dot

noiseword="<NOISE>";
for nch in 1; do
    taskdir=data/local/reverb_tools/ReleasePackage/reverb_tools_for_asr_ver2.0/taskFiles/${nch}ch
    # make a wav list
    for task in dt et; do
	for x in `ls ${taskdir} | grep SimData | grep _${task}_ | grep -e far -e near`; do
	    perl -se 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, " ", $dir, $_, "\n"; } }' -- -dir=${reverb}/REVERB_WSJCAM0_${task}/data ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_simu_${nch}ch_input_wav.scp
	cat ${dir}/${task}_simu_${nch}ch_input_wav.scp | cut -d " " -f2 > ${dir}/${task}_simu_input_files
    done

    for task in dt et; do
	for x in `ls ${taskdir} | grep SimData | grep _${task}_ | grep -e far -e near`; do
	    perl -se 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, " ", $dir, $_, "\n"; } }' -- -dir=${wavdir}/DNN-WPE/8ch/REVERB_WSJCAM0_${task}/data ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_simu_${nch}ch_output_wav.scp
	cat ${dir}/${task}_simu_${nch}ch_output_wav.scp | cut -d " " -f2 > ${dir}/${task}_simu_output_files_dnnwpe
	for x in `ls ${taskdir} | grep SimData | grep _${task}_ | grep -e far -e near`; do
	    perl -se 'while (<>) { chomp; if (m/\/(\w{8})[^\/]+$/) { print $1, " ", $dir, $_, "\n"; } }' -- -dir=${wavdir}/DNN/REVERB_WSJCAM0_${task}/data ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_simu_${nch}ch_output_wav.scp
	cat ${dir}/${task}_simu_${nch}ch_output_wav.scp | cut -d " " -f2 > ${dir}/${task}_simu_output_files_dnn
    done
done

rm data/*.scp
rm data/*.dot
