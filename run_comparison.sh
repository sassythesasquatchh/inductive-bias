#!/bin/bash

#SBATCH --job-name=compare_models
#SBATCH --output=compare_models.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpumem:12g 
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=padowd@student.ethz.ch

module load stack/2024-05 gcc/13.2.0 python_cuda/3.11.6 cuda/12.2.1
source /cluster/project/math/dagraber/patrick/ib_env/bin/activate

training_files=(
    "normal_training_1000"
    # "normal_training_1000_0.01"
    # "normal_training_100"
    # "sparse_training_100"
    # "sparse_training_100_0.01"
)

models=(
    "unstructured"
    "informed"
    "hybrid"
    "baseline"
    "fully-informed"
    "informed-cnn"
    "informed-hybrid"
)

# models=(
#     "baseline"
# )

for training_file in "${training_files[@]}"; do
    for model in "${models[@]}"; do
        echo "Training with $training_file, model=$model"
        python3 train_rwm.py --model $model --context 33 --forecast 33 --hidden_dim 64 --epochs 300 --batch_size 32 --embedding_dim 3 --train_path "data/$training_file.pkl"
    done
done
