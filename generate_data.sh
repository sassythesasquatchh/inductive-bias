source venv/bin/activate

python3 -m util.data_generation --num_trajectories 1000 --option normal_training --only-small
python3 -m util.data_generation --num_trajectories 20 --option normal_training --only-small
python3 -m util.data_generation --num_trajectories 1000 --option sparse_training --only-small
python3 -m util.data_generation --num_trajectories 100 --option validation --only-small
python3 -m util.data_generation --num_trajectories 100 --option test --only-small
python3 -m util.data_generation --option visualisation --only-small
python3 -m util.data_generation --option continuity_test --only-small

# For testing
python3 -m util.data_generation --num_trajectories 5 --option normal_training --only-small
python3 -m util.data_generation --num_trajectories 5 --option validation --only-small

########################################################################################3

python3 -m util.data_generation --num_trajectories 1000 --option normal_training
python3 -m util.data_generation --num_trajectories 40 --option normal_training
python3 -m util.data_generation --num_trajectories 1000 --option sparse_training
python3 -m util.data_generation --num_trajectories 100 --option validation
python3 -m util.data_generation --num_trajectories 100 --option test
python3 -m util.data_generation --option visualisation
python3 -m util.data_generation --option continuity_test

# For testing
python3 -m util.data_generation --num_trajectories 5 --option normal_training
python3 -m util.data_generation --num_trajectories 5 --option validation

################################################################################

python3 -m util.data_generation --num_trajectories 1000 --option normal_training --only-small --damping 0.1
python3 -m util.data_generation --num_trajectories 20 --option normal_training --only-small --damping 0.1
python3 -m util.data_generation --num_trajectories 1000 --option sparse_training --only-small --damping 0.1
python3 -m util.data_generation --num_trajectories 100 --option validation --only-small --damping 0.1
python3 -m util.data_generation --num_trajectories 100 --option test --only-small --damping 0.1
python3 -m util.data_generation --option visualisation --only-small --damping 0.1
python3 -m util.data_generation --option continuity_test --only-small --damping 0.1

# For testing
python3 -m util.data_generation --num_trajectories 5 --option normal_training --only-small --damping 0.1
python3 -m util.data_generation --num_trajectories 5 --option validation --only-small --damping 0.1

########################################################################################3

python3 -m util.data_generation --num_trajectories 1000 --option normal_training --damping 0.1
python3 -m util.data_generation --num_trajectories 40 --option normal_training --damping 0.1
python3 -m util.data_generation --num_trajectories 1000 --option sparse_training --damping 0.1
python3 -m util.data_generation --num_trajectories 100 --option validation --damping 0.1
python3 -m util.data_generation --num_trajectories 100 --option test --damping 0.1
python3 -m util.data_generation --option visualisation --damping 0.1
python3 -m util.data_generation --option continuity_test --damping 0.1

# For testing
python3 -m util.data_generation --num_trajectories 5 --option normal_training --damping 0.1
python3 -m util.data_generation --num_trajectories 5 --option validation --damping 0.1