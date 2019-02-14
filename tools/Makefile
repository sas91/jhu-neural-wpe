CONDA_URL = https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
.PHONY: all clean

all: virtualenv download-kaldi-tools sph2pipe hickle

download-kaldi-tools:
	wget https://github.com/kaldi-asr/kaldi/raw/master/egs/wsj/s5/utils/parallel/queue.pl -O queue.pl
	wget https://github.com/kaldi-asr/kaldi/raw/master/egs/wsj/s5/utils/parallel/run.pl -O run.pl
	wget https://github.com/kaldi-asr/kaldi/raw/master/egs/wsj/s5/utils/parallel/slurm.pl -O slurm.pl
	wget https://github.com/kaldi-asr/kaldi/raw/master/egs/wsj/s5/utils/parse_options.sh -O parse_options.sh
	chmod 755 queue.pl
	chmod 755 run.pl
	chmod 755 slurm.pl
	chmod 755 parse_options.sh 

miniconda.sh:
	test -f miniconda.sh || wget $(CONDA_URL) -O miniconda.sh

conda: miniconda.sh
	test -d $(PWD)/venv || bash miniconda.sh -b -p $(PWD)/venv
	. venv/bin/activate && conda config --set always_yes yes --set changeps1 no
	. venv/bin/activate && conda update conda
	. venv/bin/activate && conda info -a

virtualenv: conda requirements.txt
	. venv/bin/activate && conda install -y numpy matplotlib
	. venv/bin/activate && cat requirements.txt | pip install -r /dev/stdin
	. venv/bin/activate && pip install cupy==5.0.0

hickle: virtualenv
	test -d $(PWD)/hickle || git clone https://github.com/karandwivedi42/hickle.git
	. venv/bin/activate && cd hickle && python setup.py install

sph2pipe: sph2pipe_compiled

sph2pipe_compiled: sph2pipe_v2.5/sph2pipe

sph2pipe_v2.5/sph2pipe: | sph2pipe_v2.5
	cd sph2pipe_v2.5/; \
	$(CC) -o sph2pipe  *.c -lm

sph2pipe_v2.5:
	test -d $(PWD)/sph2pipe_v2.5 || wget -T 10 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz || \
	wget --no-check-certificate -T 10  https://sourceforge.net/projects/kaldi/files/sph2pipe_v2.5.tar.gz
	test -d $(PWD)/sph2pipe_v2.5 || tar xzf sph2pipe_v2.5.tar.gz 

clean:
	rm -fr venv hickle sph2pipe_v2.5
	rm -f *.sh
	find . -iname "*.pyc" -delete
