#!/bin/bash
#SBATCH -D /zfs/hybrilit.jinr.ru/user/a/astrakh/SU3_stag/builds/hydra/logs
#SBATCH -o /zfs/hybrilit.jinr.ru/user/a/astrakh/nqs_frustrated_phase/output_square_phase_dense64
#SBATCH -e /zfs/hybrilit.jinr.ru/user/a/astrakh/nqs_frustrated_phase/error_36_large.err
#SBATCH -t 3-00:00:00
#SBATCH -p dgx
#SBATCH --job-name=square_qua_phase_36_dense
##SBATCH --cpus-per-task=12
#SBATCH --mem=300000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 --cpus-per-task=32
##SBATCH -n 1
##SBATCH --ntasks-per-node=4
##SBATCH --nodelist=tp141,tp142
#--hostfile /s/ls4/users/astrakhantsev/SU3_stag/tests_runs/fjordhostfile
start=$(date +%s)

python3 /zfs/hybrilit.jinr.ru/user/a/astrakh/nqs_frustrated_phase/generalisation_very_large.py /zfs/hybrilit.jinr.ru/user/a/astrakh/nqs_frustrated_phase/config_square_phase_K
#python3 /zfs/hybrilit.jinr.ru/user/a/astrakh/nqs_frustrated_phase/rng.py

finish=$(date +%s)
echo $[$finish-$start]
