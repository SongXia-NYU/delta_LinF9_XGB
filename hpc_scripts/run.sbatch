#!/bin/bash
#
#SBATCH --job-name=third_run_xgb_lin_f9
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=24GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

MGLPATH=/scratch/sx801/scripts/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs:./script
singularity exec  \
    --overlay /home/sx801/conda_envs/DeltaLinF9XGB-15GB-500K.sqf \
    --overlay /scratch/sx801/data/PL_dataset_martini.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    bash -c "source /ext3/env.sh; conda activate py2.7; export PYTHONPATH=$MGLPATH:$PYTHONPATH; \
        /ext3/miniconda3/bin/python script_sx/calc_pl_cy.py "
