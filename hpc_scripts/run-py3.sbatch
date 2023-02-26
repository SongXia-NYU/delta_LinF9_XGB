#!/bin/bash
#
#SBATCH --job-name=run_xgb_lin_f9
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --mem=12GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

MGLPATH=/scratch/sx801/scripts/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs:./script
singularity exec  \
            --overlay /scratch/sx801/singularity-envs/DeltaLinF9XGB-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; export PYTHONPATH=$MGLPATH:$PYTHONPATH; \
            /ext3/miniconda3/bin/python test_on_casf/calc_dock.py "
