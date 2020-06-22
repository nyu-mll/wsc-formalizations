#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=wh629@nyu.edu
#SBATCH --mem=24000
#SBATCH --job-name=wsc
#SBATCH --output=/scratch/wh629/nlu/projects/wsc/test/xl/%j_preproc.out
module purge
module load anaconda3/5.3.1
module load cuda/10.1.105
module load gcc/6.3.0
source activate wsc
export NLU_DATA_DIR=/scratch/wh629/nlu/projects/wsc/data
export NLU_RESULTS_DIR=/scratch/wh629/nlu/projects/wsc/results

echo python $COMMAND
python $COMMAND
