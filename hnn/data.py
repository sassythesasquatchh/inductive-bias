import pickle
import jax
import jax.numpy as jnp
from util.config import Config

class PendulumDataset:
    config: Config
    
    def __init__(self, file_path, sequence_length=-1):
        self.sequence_length = sequence_length
        self._load_data(file_path)
        self._preprocess_data()
        
    def _load_data(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.trajectories = data["trajectories"]
        self.config = Config(**data["simulation_config"])
        print(f"Loaded {len(self.trajectories)} trajectories from {file_path}")
        
    def _preprocess_data(self):
        self.obs_sequences = []
        self.canonical_sequences = []
        
        if self.sequence_length <= 0:
            self.sequence_length = self.trajectories[0]["observed"].shape[1]  # Use full length if not specified

        for traj in self.trajectories:
            obs = traj["observed"]
            canonical = traj["phase"]
            T = obs.shape[0]
            
            # Break into sequences of length L
            for start_idx in range(0, T - self.sequence_length + 1, self.sequence_length):
                end_idx = start_idx + self.sequence_length
                self.obs_sequences.append(obs[start_idx:end_idx])
                self.canonical_sequences.append(canonical[start_idx:end_idx])

        self.obs_sequences = jnp.array(self.obs_sequences)
        self.canonical_sequences = jnp.array(self.canonical_sequences)
        self.num_samples = len(self.obs_sequences)
        print(f"Created {self.num_samples} sequences of length {self.sequence_length}")

    def get_input_dim(self):
        return self.obs_sequences.shape[-1]
    
    def get_latent_dim(self):
        return self.canonical_sequences.shape[-1]
        
    def get_batch(self, key, batch_size):
        idx = jax.random.choice(key, self.num_samples, shape=(batch_size,), replace=False)
        return self.obs_sequences[idx], self.canonical_sequences[idx]