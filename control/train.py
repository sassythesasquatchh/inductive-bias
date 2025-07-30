import gymnasium as gym
import torch
import numpy as np
import time
from collections import deque
import wandb  # Added for logging
import os
from dotenv import load_dotenv
import ipdb

# Load environment variables
load_dotenv()

# Custom environment and PPO agent
from control.pendulum_env import (
    PendulumEnv,
    default_map_state_to_observation,
    DEFAULT_OBSERVATION_SHAPE,
)
from control.ppo import PPOAgent
from control.encoder import get_encoder

# --- Configuration ---
ENV_ID = "CustomPendulum-v0"
MAX_EP_LEN = 200
SEED = 0
UPDATE_TIMESTEP = 4000
K_EPOCHS = 80
EPS_CLIP = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
ACTION_STD_INIT = 0.6
MAX_TRAINING_TIMESTEPS = 1_000_000
PRINT_FREQ = UPDATE_TIMESTEP * 2
SAVE_FREQ = UPDATE_TIMESTEP * 10

# Observation mapping
map_state_to_obs = default_map_state_to_observation
obs_shape = DEFAULT_OBSERVATION_SHAPE

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

latent_space = "unstructured"

encoder = get_encoder(latent_space).to(device)
encoder.eval()


def encode_state(state):
    """Encodes the state using the encoder."""
    with torch.no_grad():
        encoded_state = encoder(torch.tensor(state.T, dtype=torch.float32).to(device))
        return encoded_state.cpu().numpy()


def main():
    # --- wandb Init ---
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name=f"control_{latent_space}",
        config={
            "env_id": ENV_ID,
            "max_ep_len": MAX_EP_LEN,
            "seed": SEED,
            "update_timestep": UPDATE_TIMESTEP,
            "k_epochs": K_EPOCHS,
            "eps_clip": EPS_CLIP,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "lr_actor": LR_ACTOR,
            "lr_critic": LR_CRITIC,
            "action_std_init": ACTION_STD_INIT,
            "max_training_timesteps": MAX_TRAINING_TIMESTEPS,
            "encoder_type": latent_space,
        },
    )

    # Environment setup
    gym.register(
        id=ENV_ID,
        entry_point="control.pendulum_env:PendulumEnv",
        max_episode_steps=MAX_EP_LEN,
        kwargs={
            "map_state_to_observation_fn": map_state_to_obs,
            "observation_space_shape": obs_shape,
            "render_mode": "rgb_array",
        },
    )
    env = gym.make(ENV_ID)

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Agent setup
    # state_dim = env.observation_space.shape[0]
    state_dim = 3
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    ppo_agent = PPOAgent(
        state_dim,
        action_dim,
        LR_ACTOR,
        LR_CRITIC,
        GAMMA,
        GAE_LAMBDA,
        K_EPOCHS,
        EPS_CLIP,
        ACTION_STD_INIT,
        device,
    )

    # --- Training ---
    print(f"Training on {ENV_ID} for {MAX_TRAINING_TIMESTEPS} timesteps...")
    start_time = time.time()

    memory = {
        "states": [],
        "actions": [],
        "logprobs": [],
        "rewards": [],
        "values": [],
        "dones": [],
    }

    time_step = 0
    i_episode = 0
    ep_rewards = deque(maxlen=100)

    while time_step < MAX_TRAINING_TIMESTEPS:
        state, _ = env.reset(seed=SEED + i_episode)
        state = encode_state(state)
        current_ep_reward = 0
        done = False

        for t in range(1, MAX_EP_LEN + 1):
            action_scaled, log_prob, value = ppo_agent.select_action(state)
            action = action_scaled * max_action

            memory["states"].append(state)
            memory["actions"].append(action_scaled)
            memory["logprobs"].append(log_prob)
            memory["values"].append(value)

            state, reward, terminated, truncated, _ = env.step(action)
            state = encode_state(state)
            done = terminated or truncated

            memory["rewards"].append(reward)
            memory["dones"].append(done)

            current_ep_reward += reward
            time_step += 1

            if time_step % UPDATE_TIMESTEP == 0:
                _, _, last_val = ppo_agent.select_action(state)
                memory["values"].append(last_val)
                memory["dones"].append(done)

                ppo_agent.update(memory)
                memory = {k: [] for k in memory}

                wandb.log({"policy_update_timestep": time_step})

            if time_step % PRINT_FREQ == 0:
                avg_reward = np.mean(ep_rewards) if ep_rewards else 0
                wandb.log(
                    {
                        "timestep": time_step,
                        "avg_reward_100_episodes": avg_reward,
                        "episodes": i_episode,
                    }
                )

            if time_step % SAVE_FREQ == 0:
                save_path = f"./PPO_{ENV_ID}_ts{time_step}.pth"
                ppo_agent.save(save_path)
                wandb.save(save_path)
                wandb.log({"model_saved_at_timestep": time_step})

            if done:
                break

        ep_rewards.append(current_ep_reward)
        wandb.log(
            {
                "episode_reward": current_ep_reward,
                "episode_length": t,
                "episode": i_episode,
                "timestep": time_step,
            }
        )

        i_episode += 1

    # --- Cleanup ---
    env.close()
    end_time = time.time()

    # Save final model
    final_save_path = f"./PPO_{ENV_ID}_final.pth"
    ppo_agent.save(final_save_path)
    wandb.save(final_save_path)
    wandb.log({"training_time_sec": end_time - start_time})
    print(f"Training completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        ipdb.post_mortem()
