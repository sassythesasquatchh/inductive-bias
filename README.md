# Welcome to Protein Stability!

## Setup
### Data
This repo assumes that data is stored in a folder ./data located in the main directory.

### Python environment
source /cluster/project/math/dagraber/patrick/ib_env/bin/activate

```bash
source /cluster/project/math/dagraber/patrick/venv/bin/activate
```

However, to run signalP:
```bash
source /cluster/project/math/dagraber/patrick/miniconda3/bin/activate
conda activate signalp
```

To run esmfold
```bash
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1
# unset PYTHONHOME
source /cluster/project/math/dagraber/patrick/miniconda3/bin/activate
conda activate esmfold
```

### Slurm
To submit a job:
```bash
sbatch job.sh
```

### Useful bash commands for gpu
```bash
srun --gpus 1 --gres gpumem:20g --time 03:00:00 --mem-per-cpu=32G --pty bash
module load stack/2024-05 gcc/13.2.0 python_cuda/3.11.6 cuda/12.2.1
source /cluster/project/math/dagraber/patrick/ib_env/bin/activate
```

### Useful bash commands for cpu
```bash
srun --time 10:00:00 --mem-per-cpu=32G --pty bash
module load stack/2024-05 gcc/13.2.0
source /cluster/project/math/dagraber/patrick/ib_env/bin/activate
```

### Commands to run python modules
```bash
python3 -m train.ablation.train
python3 -m train.train
```

### Preprocessing
```bash
python3 -m preprocessing.protstab.combine
python3 -m preprocessing.add_meltome_info
python3 -m preprocessing.protstab.crop_structures
python3 -m preprocessing.protstab.update_sequences
```

### AlphaFold url
https://alphafold.ebi.ac.uk/files/AF-Q6KZI3-F1-model_v4.pdb