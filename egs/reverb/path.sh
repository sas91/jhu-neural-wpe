MAIN_ROOT=$PWD/../..
NWPE_ROOT=$MAIN_ROOT/src
NWPE_BIN=$MAIN_ROOT/src/bin
TOOLS_ROOT=$MAIN_ROOT/tools

source $MAIN_ROOT/tools/venv/bin/activate
export PYTHONPATH=$NWPE_ROOT/:$PYTHONPATH
export PATH=$TOOLS_ROOT/:$NWPE_BIN/:$PATH

export cuda_cmd="queue.pl --mem 8G --gpu 1 --config conf/gpu.conf"
export cuda_cmd_slurm="slurm.pl --mem 10G --gpu 1 --num_threads 6 --config conf/slurm_gpu.conf"
export slurm_cmd="slurm.pl --mem 2G --config conf/slurm.conf"
