#!/bin/bash
#  2018 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

# Config:
nj=20
cmd=run.pl

. ../../tools/parse_options.sh || exit 1;

rm -rf data/tr/
rm -rf data/dt/
rm -rf data/list*
rm -rf data/list*
rm -rf data/file_list*
rm -rf log

local/create_list.py

split -n l/$nj -d -a 3 data/list_dt data/list_dt-
split -n l/$nj -d -a 3 data/list_tr data/list_tr-
mkdir -p log

for n in `seq $nj`; do
printf -v BATCH_ID "%03d" $((n - 1 ))
cat <<-EOF > log/prepare_data.$n.sh
. ./path.sh
local/prepare_data.py --thread_id ${BATCH_ID} --list_tr data/list_tr-${BATCH_ID} --list_dt data/list_dt-${BATCH_ID}
EOF
done

if [[ $cmd == 'slurm.pl' ]] ; then
  cmd=$slurm_cmd
fi
chmod a+x log/prepare_data.*.sh
$cmd JOB=1:$nj log/prepare_data.JOB.log \
  log/prepare_data.JOB.sh

cat data/file_list_tr_* > data/file_list_tr
cat data/file_list_dt_* > data/file_list_dt
