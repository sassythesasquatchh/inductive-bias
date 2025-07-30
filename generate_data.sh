python3 diff_data_gen.py --num_trajectories 1000 --option normal_training
python3 diff_data_gen.py --num_trajectories 1000 --option normal_training --noise=0.01
python3 diff_data_gen.py --num_trajectories 100  --option validation
python3 diff_data_gen.py --num_trajectories 2 --option testing
python3 diff_data_gen.py --num_trajectories 100 --option sparse_training
python3 diff_data_gen.py --num_trajectories 100 --option sparse_training --noise=0.01
python3 diff_data_gen.py --num_trajectories 100 --option normal_training 