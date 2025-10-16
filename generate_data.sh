source venv/bin/activate

python3 -m util.data_generation --num_trajectories 1000 --option normal_training --only-small
python3 -m util.data_generation --num_trajectories 100 --option validation --only-small
python3 -m util.data_generation --option visualisation --only-small
python3 -m util.data_generation --option continuity_test --only-small

# For testing
python3 -m util.data_generation --num_trajectories 5 --option normal_training --only-small
python3 -m util.data_generation --num_trajectories 5 --option validation --only-small