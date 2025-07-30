import pickle
import ipdb

with open("data/pendulum_trajectories.pkl", "rb") as f:
    data = pickle.load(f)
ipdb.set_trace()
