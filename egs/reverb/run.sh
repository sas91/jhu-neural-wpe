#!/bin/bash

#  2018 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Requirements) matlab and tcsh
if [ ! `which tcsh` ]; then
  echo "Install tcsh, which is used in some REVERB scripts"
  exit 1
fi
if [ ! `which matlab` ]; then
  echo "Install matlab, which is used to generate multi-condition data"
  exit 1
fi

. ./path.sh

stage=0
nch_se=8
# flag for turing on computation of dereverberation measures
compute_se=true
# please make sure that you or your institution have the license to report PESQ before turning on the below flag
enable_pesq=false

. ../../tools/parse_options.sh
# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# please make sure to set the paths of the REVERB and WSJ0 data
if [[ $(hostname -f) == *.clsp.jhu.edu ]] ; then
  # Data path in CLSP grid
  reverb=/export/corpora5/REVERB_2014/REVERB
  export wsjcam0=/export/corpora3/LDC/LDC95S24/wsjcam0
  # set LDC WSJ0 directory to obtain LMs
  # REVERB data directory only provides bi-gram (bcb05cnp), but this recipe also uses 3-gram (tcb05cnp.z)
  export wsj0=/export/corpora5/LDC/LDC93S6A/11-13.1 #LDC93S6A or LDC93S6B
  # It is assumed that there will be a 'wsj0' subdirectory
  # within the top-level corpus directory
elif [[ $(hostname -f) == *bc-login* ]] ; then
  # Data path in MARCC grid
  reverb=/data/swatana4/REVERB_2014/REVERB
  export wsjcam0=/data/swatana4/LDC/LDC95S24/wsjcam0
  export wsj0=/data/swatana4/LDC/LDC93S6A/11-13.1
else
  echo "Set the data directory locations." && exit 1;
fi

# number of jobs for feature extraction and model training
nj=92

wavdir=${PWD}/wav
pesqdir=${PWD}/local
if [ ${stage} -le 1 ]; then
  echo "stage 1: Data preparation"
  local/generate_data.sh --wavdir ${wavdir} ${wsjcam0}
  local/prepare_simu_data.sh --wavdir ${wavdir} ${reverb} ${wsjcam0}
  local/prepare_real_data.sh --wavdir ${wavdir} ${reverb}
fi

if [ $stage -le 2 ]; then
  echo "stage 2: Feature extraction"
  if [[ $(hostname -f) == *bc-login* ]] ; then
    local/prepare_data_parallel.sh --cmd slurm.pl
  else
    local/prepare_data_parallel.sh
  fi
fi

if [ $stage -le 3 ]; then
  echo "stage 3: Train DNN"
  if [[ $(hostname -f) == *bc-login* ]] ; then
    $cuda_cmd_slurm log/run_train.log local/run_train.sh
  else
    $cuda_cmd log/run_train.log local/run_train.sh
  fi
fi

if [ $stage -le 4 ]; then
  echo "stage 4: Enhancement"
  if [[ $(hostname -f) == *bc-login* ]] ; then
    $cuda_cmd_slurm log/enhance.log local/enhance.sh
  else
    $cuda_cmd log/enhance.log local/enhance.sh
  fi
fi

if [ $stage -le 5 ] && $compute_se; then
  echo "stage 5: Compute dereverberation scores"
  if [ ! -d local/REVERB_scores_source ] || [ ! -d local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools/SRMRToolbox ] || [ ! -f local/PESQ ]; then
    # download and install speech enhancement evaluation tools
    local/download_se_eval_tool.sh
  fi
  local/compute_se_scores.sh --nch $nch_se --enable_pesq $enable_pesq $reverb $wavdir $pesqdir
  cat data/compute_se_${nch_se}ch/scores/score_SimData
  cat data/compute_se_${nch_se}ch/scores/score_RealData
fi
