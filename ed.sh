#!/bin/bash

#SBATCH -p dgx
#SBATCH -t 3-00:00:00
#SBATCH --mem=200000
#SBATCH -N 1 -n 1 --cpus-per-task=32

conda activate tcm-test

pushd "/zfs/hybrilit.jinr.ru/user/a/astrakh/nqs_frustrated_phase/data/square/24/$1"
/zfs/hybrilit.jinr.ru/user/a/astrakh/HPhi.build/src/HPhi -e namelist.def 2>&1 | tee stdout.txt
popd
