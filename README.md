# jhu-neural-wpe

jhu-neural-wpe is a Neural network based dereverberation toolkit which includes an open source implementation of
Keisuke Kinoshita, et al. "Neural network-based spectrum estimation for online WPE dereverbertion.", Proc. Interspeech. 2017.
jhu-neural-wpe uses [chainer](https://chainer.org/) as the deep learning engine.


## Installation

### Step 1) setting of the environment

To use cuda (and cudnn), make sure to set paths in your `.bashrc` or `.bash_profile` appropriately.
```
CUDAROOT=/path/to/cuda

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
```

### Step 2) installation of tools

Install Python libraries with miniconda and other required tools
```sh
$ cd tools
$ make
```

### Step 3) installation check

You can check whether the install is succeeded via the following commands
```sh
$ cd tools
$ source venv/bin/activate && python check_install.py
```
If you have no warning, you are ready to run the recipe!

If there are some problems in python libraries, you can re-setup only python environment via following commands
```sh
$ cd tools
$ make clean
```

## Execution with REVERB data

```sh
$ cd egs/reverb
$ ./run.sh
```

### Setup in your cluster

Change `$cuda_cmd` in `path.sh` according to your cluster setup.
For more information see http://kaldi-asr.org/doc/queue.html.

## Acknowledgement

This work was supported by Yahoo Japan Corporation.
