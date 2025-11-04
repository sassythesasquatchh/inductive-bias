encoder="unstructured"
dynamics="unstructured"
decoder="unstructured"
context=33
latent_dim=3
EPOCHS=250
FORECAST=8
ALPHA=0.9
EXPERIMENT_TAG="loss_experiment_2"
TRAINING_FILE="data/normal_training_1000.pkl"
VALIDATION_FILE="data/validation_100.pkl"
VISUALISATION_FILE="data/visualisation.pkl"
CONTINUITY_FILE="data/continuity_test.pkl"

python3 -m context_models.train \
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
    --run_name "supervise_rollout" \
    --supervise_rollout

python3 -m context_models.train \
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
    --run_name "supervise_end_to_end" \
    --supervise_end_to_end

python3 -m context_models.train \
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
    --run_name "supervise_both" \
    --supervise_end_to_end \
    --supervise_rollout

python3 -m context_models.train \
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
    --run_name "supervise_both_penalise_mismatch" \
    --supervise_end_to_end \
    --supervise_rollout \
    --penalise_latent_mismatch 


python3 -m context_models.train \
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
    --run_name "supervise_both_all_penalties" \
    --supervise_end_to_end \
    --supervise_rollout \
    --penalise_latent_mismatch \
    --penalise_latent_dynamics \
    --penalise_latent_magnitude