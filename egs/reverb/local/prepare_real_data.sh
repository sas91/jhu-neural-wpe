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
if [ $# -ne 1 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <reverb-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora5/REVERB_2014/REVERB"
  exit 1
fi

set -e -o pipefail

reverb=$1

# working directory
dir=${PWD}/data/
mkdir -p ${dir}

noiseword="<NOISE>";
for nch in 1; do
    taskdir=data/local/reverb_tools/ReleasePackage/reverb_tools_for_asr_ver2.0/taskFiles/${nch}ch
    # make a wav list
    for task in dt et; do
	if [ ${task} == 'dt' ]; then
	    audiodir=${reverb}/MC_WSJ_AV_Dev
	    audiodir_dnnwpe=${wavdir}/DNN-WPE/8ch/MC_WSJ_AV_Dev
	    audiodir_dnn=${wavdir}/DNN/MC_WSJ_AV_Dev
	elif [ ${task} == 'et' ]; then
	    audiodir=${reverb}/MC_WSJ_AV_Eval
	    audiodir_dnnwpe=${wavdir}/DNN-WPE/8ch/MC_WSJ_AV_Eval
	    audiodir_dnn=${wavdir}/DNN/MC_WSJ_AV_Eval
	fi
	for x in `ls ${taskdir} | grep RealData | grep _${task}_`; do
	    perl -se 'while(<>){m:^\S+/[\w\-]*_(T\w{6,7})\.wav$: || die "Bad line $_"; $id = lc $1; print "$id $dir$_";}' -- -dir=${audiodir} ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_real_${nch}ch_input_wav.scp
	cat ${dir}/${task}_real_${nch}ch_input_wav.scp | cut -d " " -f2 > ${dir}/${task}_real_input_files
	for x in `ls ${taskdir} | grep RealData | grep _${task}_`; do
	    perl -se 'while(<>){m:^\S+/[\w\-]*_(T\w{6,7})\.wav$: || die "Bad line $_"; $id = lc $1; print "$id $dir$_";}' -- -dir=${audiodir_dnnwpe} ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_real_${nch}ch_output_wav.scp
	cat ${dir}/${task}_real_${nch}ch_output_wav.scp | cut -d " " -f2 > ${dir}/${task}_real_output_files_dnnwpe
	for x in `ls ${taskdir} | grep RealData | grep _${task}_`; do
	    perl -se 'while(<>){m:^\S+/[\w\-]*_(T\w{6,7})\.wav$: || die "Bad line $_"; $id = lc $1; print "$id $dir$_";}' -- -dir=${audiodir_dnn} ${taskdir}/$x |\
		sed -e "s/^\(...\)/\1_${x}_\1/"
	done > ${dir}/${task}_real_${nch}ch_output_wav.scp
	cat ${dir}/${task}_real_${nch}ch_output_wav.scp | cut -d " " -f2 > ${dir}/${task}_real_output_files_dnn
    done
done

rm data/*.scp
