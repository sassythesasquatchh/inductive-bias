#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="config.txt"
EXPERIMENT_TAG="default"
FORECAST=8
ALPHA=0.95
EPOCHS=250
TRAINING_FILE="data/normal_training_1000.pkl"
VALIDATION_FILE="data/validation_100.pkl"
VISUALISATION_FILE="data/visualisation.pkl"
CONTINUITY_FILE="data/continuity_test.pkl"


# Usage message
usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -c, --config FILE          Path to configuration file (default: config.txt)"
    echo "  --training-file PATH       Path to training data file"
    echo "  --validation-file PATH     Path to validation data file"
    echo "  --visualisation-file PATH  Path to visualisation data file"
    echo "  --continuity-file PATH     Path to continuity test data file"
    echo "  -t, --tag TAG              Experiment tag (default: default)"
    echo "  -f, --forecast N           Forecast horizon (default: 8)"
    echo "  -a, --alpha FLOAT          Alpha value (default: 0.95)"
    echo "  -e, --epochs N             Number of training epochs (default: 250)"
    echo "  -h, --help                 Show this help message and exit"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -t|--tag)
            EXPERIMENT_TAG="$2"
            shift 2
            ;;
        -f|--forecast)
            FORECAST="$2"
            shift 2
            ;;
        -a|--alpha)
            ALPHA="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --training-file)
            TRAINING_FILE="$2"
            shift 2
            ;;
        --validation-file)
            VALIDATION_FILE="$2"
            shift 2
            ;;
        --visualisation-file)
            VISUALISATION_FILE="$2"
            shift 2
            ;;
        --continuity-file)
            CONTINUITY_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "=== Experiment parameters ==="
echo "Config file: $CONFIG_FILE"
echo "Experiment tag: $EXPERIMENT_TAG"
echo "Forecast horizon: $FORECAST"
echo "Alpha: $ALPHA"
echo "Epochs: $EPOCHS"
echo "Training file: $TRAINING_FILE"
echo "Validation file: $VALIDATION_FILE"
echo "Visualisation file: $VISUALISATION_FILE"
echo "Continuity file: $CONTINUITY_FILE"
echo "=============================="

# Context models
while read -r encoder dynamics decoder context latent_dim; do
    [[ -z "$encoder" || "$encoder" =~ ^# ]] && continue

    echo "Running experiment with encoder: $encoder, dynamics: $dynamics, decoder: $decoder, context: $context, latent_dim: $latent_dim"
    python3 -m context_models_3.train \
        --encoder "$encoder" \
        --decoder "$decoder" \
        --dynamics "$dynamics" \
        --context "$context" \
        --embedding_dim "$latent_dim" \
        --epochs "$EPOCHS" \
        --forecast "$FORECAST" \
        --alpha "$ALPHA" \
        --tags "$EXPERIMENT_TAG" \
        --train_path "$TRAINING_FILE" \
        --val_path "$VALIDATION_FILE" \
        --visualisation_data_path "$VISUALISATION_FILE" \
        --continuity_data_path "$CONTINUITY_FILE" \
        --supervise_end_to_end \
        --supervise_rollout \
        --penalise_latent_mismatch
done < "$CONFIG_FILE"

# HNN models
# for encoder in informed unstructured; do
#     for decoder in informed unstructured; do
#         echo "Running HNN experiment with encoder: $encoder, decoder: $decoder"
#         python3 -m hnn.train \
#             --encoder "$encoder" \
#             --decoder "$decoder" \
#             --segment_length 33 \
#             --epochs 2000 \
#             --forecast "$FORECAST" \
#             --tags "$EXPERIMENT_TAG"
#     done
# done

# # FLD
# python3 -m fld.train \
#     --context 33 \
#     --tags "$EXPERIMENT_TAG" \
#     --embedding_dim 3 \
#     --forecast "$FORECAST" \
#     --epochs 300
