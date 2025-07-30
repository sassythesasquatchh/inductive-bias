#!/bin/bash


# for model in "hybrid" "informed"; do
#     for k in 1 5 10 30 50; do
#         # Add your commands here
#         python3 train_all_to_all.py --model $model --max_k $k
#     done
# done

# for i in {3..12}; do
# for k in 1 5 10 30 50; do
# for k in 10; do
#     # for model in "hybrid" "informed" "unstructured"; do
#     for model in "unstructured" "informed"; do
#         python3 train_all_to_all.py --model $model --max_k $k --embedding_dim 3 --hidden_dim 128
#     done
# done

training_files=(
    "normal_training_1000"
    "normal_training_1000_0.01"
    "normal_training_100"
    "sparse_training_100"
    "sparse_training_100_0.01"
)

# models=(
#     "unstructured"
#     "informed"
#     "hybrid"
# )

models=(
    "baseline"
)

for training_file in "${training_files[@]}"; do
    for model in "${models[@]}"; do
        echo "Training with $training_file, model=$model"
        python3 train_rwm.py --model $model --context 32 --forecast 32 --hidden_dim 64 --epochs 300 --batch_size 32 --train_path "data/$training_file.pkl"
    done
done


