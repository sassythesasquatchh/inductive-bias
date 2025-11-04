source venv/bin/activate

python3 -m util.data_generation --num_trajectories 1000 --option normal_training
python3 -m util.data_generation --num_trajectories 20 --option normal_training
python3 -m util.data_generation --num_trajectories 1000 --option sparse_training
python3 -m util.data_generation --num_trajectories 100 --option validation
python3 -m util.data_generation --num_trajectories 100 --option test
python3 -m util.data_generation --option visualisation
python3 -m util.data_generation --option continuity_test

# For testing
python3 -m util.data_generation --num_trajectories 5 --option normal_training
python3 -m util.data_generation --num_trajectories 5 --option validation