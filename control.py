import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt  # Optional for plotting results
import time  # Optional for rendering delay

SAMPLING_POINTS = np.array([1.0, 0.8, 0.6, 0.4])


# --- 1. Deterministic Conversion Function (Placeholder) ---
# YOU MUST REPLACE THIS with your actual deterministic function
# It takes the standard pole state [angle, angular_velocity]
# and returns your custom 12D representation.
def sample_observables(pole_state):
    """
    Args:
        pole_state (np.ndarray): Array of shape (2,) containing [pole_angle, pole_angular_velocity].

    Returns:
        np.ndarray: Array of shape (12,) representing the custom pole state.
    """
    angle, ang_vel = pole_state
    # Calculate observables
    x = SAMPLING_POINTS.reshape(-1, 1) * np.sin(angle)
    y = -SAMPLING_POINTS.reshape(-1, 1) * np.cos(angle)
    linear_velocity = SAMPLING_POINTS.reshape(-1, 1) * ang_vel

    observables = np.vstack((x, y, linear_velocity))

    return observables.astype(np.float32)


# --- 2. Custom CartPole Environment ---
class CustomCartPoleEnv(gym.Env):
    """
    Custom CartPole Environment based on Gymnasium's CartPole-v1
    but with a modified observation space.

    Observation Space: A dictionary with:
        - 'cart_state': Box(2,) [cart_position, cart_velocity]
        - 'pole_state': Box(12,) [custom 12D pole representation]

    Action Space: Discrete(2) [0: push left, 1: push right]

    Internal State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # --- Physics Parameters (same as CartPole-v1) ---
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # --- Failure Thresholds ---
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # --- Define Observation Space ---
        # Cart state: position, velocity
        cart_state_space = spaces.Box(
            low=np.array([-self.x_threshold * 2, -np.finfo(np.float32).max]),
            high=np.array([self.x_threshold * 2, np.finfo(np.float32).max]),
            shape=(2,),
            dtype=np.float32,
        )
        # Pole state: your custom 12D representation
        # Define bounds based on your expected representation range, or use -inf, inf
        pole_state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        # Combine into a dictionary space
        self.observation_space = spaces.Dict(
            {"cart_state": cart_state_space, "pole_state": pole_state_space}
        )

        # --- Define Action Space ---
        self.action_space = spaces.Discrete(2)  # 0: push left, 1: push right

        # --- Rendering Setup ---
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        # --- Internal State ---
        # This holds the standard [x, x_dot, theta, theta_dot] used for physics
        self.state = None
        self.steps_beyond_terminated = None

    def _get_obs(self):
        """
        Constructs the observation dictionary from the internal 4D state.
        This is where the conversion to the custom pole state happens.
        """
        if self.state is None:
            raise ValueError(
                "Cannot get observation when internal state is None. Call reset first."
            )

        x, x_dot, theta, theta_dot = self.state

        # Extract standard cart state
        cart_state = np.array([x, x_dot], dtype=np.float32)

        # Extract standard pole state for conversion
        standard_pole_state = np.array([theta, theta_dot], dtype=np.float32)

        # Convert to custom 12D representation using the provided function
        custom_pole_state = convert_pole_state_to_12d(standard_pole_state)

        # Return the observation dictionary
        return {"cart_state": cart_state, "pole_state": custom_pole_state}

    def step(self, action):
        """Performs one step in the environment"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0 or 1.")
        if self.state is None:
            raise ValueError("Cannot call step before calling reset.")

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        # --- Physics Simulation (Standard CartPole Equations) ---
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # --- Integration (Euler method) ---
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler (more stable but not default in original gym)
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # Update internal state
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # --- Check Termination Conditions ---
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        # --- Calculate Reward ---
        reward = 0.0
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0  # Reward for the step it fell on
        else:
            # Already terminated, should have been reset
            if self.steps_beyond_terminated == 0:
                gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0  # No reward after termination

        # --- Get Observation in Custom Format ---
        observation = self._get_obs()

        # --- Render (if requested) ---
        if self.render_mode == "human":
            self.render()

        # --- Return standard Gymnasium tuple ---
        # We don't implement truncation based on time limit here, but could be added
        truncated = False
        info = {}  # No extra info to return
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state"""
        super().reset(seed=seed)  # Important for seeding random number generators

        # Reset internal state to random values near zero
        low = -0.05
        high = 0.05
        self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
            np.float32
        )
        self.steps_beyond_terminated = None

        # Get initial observation in the custom format
        observation = self._get_obs()
        info = {}  # No extra info to return

        # Render (if requested)
        if self.render_mode == "human":
            self.render()

        return observation, info

    def render(self):
        """Renders the environment (optional)"""
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify a render_mode ('human' or 'rgb_array') "
                "during initialization."
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame` for rendering"
            )

        # Initialize Pygame if not already done
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Custom CartPole")
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Check if state is valid for rendering
        if self.state is None:
            return None

        # --- Drawing parameters ---
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)  # Visual length of pole
        cartwidth = 50.0
        cartheight = 30.0

        # --- Extract state for rendering ---
        x = self.state
        cartx = (
            x[0] * scale + self.screen_width / 2.0
        )  # Cart center horizontal position
        carty = self.screen_height * 0.6  # Vertical position of cart base

        # --- Create drawing surface ---
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))  # White background

        # --- Draw Track ---
        track_y = carty + cartheight / 2
        pygame.draw.line(
            self.surf,
            (0, 0, 0),  # Black color
            (0, track_y),
            (self.screen_width, track_y),
            1,  # Line width
        )

        # --- Draw Cart ---
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords_shifted = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords_shifted, (0, 0, 0))  # Black outline
        gfxdraw.filled_polygon(
            self.surf, cart_coords_shifted, (150, 150, 150)
        )  # Grey fill

        # --- Draw Pole ---
        pole_l, pole_r, pole_t, pole_b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole_coords = [
            (pole_l, pole_b),
            (pole_l, pole_t),
            (pole_r, pole_t),
            (pole_r, pole_b),
        ]
        # Rotate pole around its base (axle point)
        pole_angle = -x[2]  # Negative because pygame y increases downwards
        axle_x = cartx
        axle_y = carty - axleoffset  # Axle slightly above cart center
        rotated_pole_coords = []
        for coord in pole_coords:
            x_coord, y_coord = coord
            new_x = x_coord * math.cos(pole_angle) - y_coord * math.sin(pole_angle)
            new_y = x_coord * math.sin(pole_angle) + y_coord * math.cos(pole_angle)
            rotated_pole_coords.append((new_x + axle_x, new_y + axle_y))

        gfxdraw.aapolygon(self.surf, rotated_pole_coords, (0, 0, 0))  # Black outline
        gfxdraw.filled_polygon(
            self.surf, rotated_pole_coords, (204, 153, 102)
        )  # Brown fill

        # --- Draw Axle ---
        gfxdraw.aacircle(
            self.surf, int(axle_x), int(axle_y), int(polewidth / 2), (100, 100, 100)
        )  # Grey outline
        gfxdraw.filled_circle(
            self.surf, int(axle_x), int(axle_y), int(polewidth / 2), (128, 128, 128)
        )  # Darker grey fill

        # --- Flip surface vertically (Pygame origin is top-left) ---
        # self.surf = pygame.transform.flip(self.surf, False, True) # Flip if needed based on coordinate system

        # --- Blit surface to screen ---
        self.screen.blit(self.surf, (0, 0))

        # --- Update display and manage timing (for human mode) ---
        if self.render_mode == "human":
            pygame.event.pump()  # Process events
            self.clock.tick(self.metadata["render_fps"])  # Control frame rate
            pygame.display.flip()  # Update the full display

        # --- Return RGB array (for rgb_array mode) ---
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2),  # HWC format
            )

    def close(self):
        """Closes the rendering window"""
        if self.screen is not None:
            try:
                import pygame

                pygame.display.quit()
                pygame.quit()
                self.isopen = False
                self.screen = None
                self.clock = None
            except ImportError:
                pass  # Pygame not installed, nothing to close


# --- 3. RL Agent (Policy Gradient - REINFORCE with Encoder) ---
class PolicyGradientAgent(nn.Module):
    """
    REINFORCE Agent with separate encoder for pole state.

    Architecture:
    - Pole State (12D) -> Encoder -> Latent Pole State (latent_dim)
    - Latent Pole State + Cart State (2D) -> Policy Network -> Action Logits (n_actions)
    """

    def __init__(
        self, cart_dim, pole_obs_dim, latent_dim, n_actions, lr=1e-3, gamma=0.99
    ):
        super(PolicyGradientAgent, self).__init__()

        self.gamma = gamma
        self.cart_dim = cart_dim
        self.pole_obs_dim = pole_obs_dim
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        # --- Encoder Network ---
        # Maps the 12D custom pole observation to a lower-dimensional latent space
        self.encoder = nn.Sequential(
            nn.Linear(pole_obs_dim, 64),  # Input: 12D pole state
            nn.ReLU(),
            nn.Linear(64, latent_dim),  # Output: latent_dim
            nn.ReLU(),  # Activation for latent space (ReLU, Tanh, or None)
        )

        # --- Policy Network ---
        # Takes the concatenated latent pole state and cart state as input
        # Outputs logits for each action
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim + cart_dim, 128),  # Input: latent + cart dims
            nn.ReLU(),
            nn.Linear(128, n_actions),  # Output: action logits
        )

        # --- Optimizer ---
        # Optimizes parameters of both the encoder and the policy network
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # --- Memory for REINFORCE ---
        # Stores log probabilities of actions and rewards for one episode
        self.log_probs = []
        self.rewards = []

    def forward(self, cart_state, pole_state):
        """
        Defines the forward pass of the agent.

        Args:
            cart_state (torch.Tensor): Tensor of cart states (batch_size, cart_dim).
            pole_state (torch.Tensor): Tensor of custom pole states (batch_size, pole_obs_dim).

        Returns:
            torch.Tensor: Action logits (batch_size, n_actions).
        """
        # 1. Encode the pole state
        latent_pole_state = self.encoder(pole_state)

        # 2. Concatenate latent pole state and cart state
        # Ensure dimensions match for concatenation (dim=-1 handles batch dimension)
        combined_state = torch.cat((latent_pole_state, cart_state), dim=-1)

        # 3. Pass combined state through the policy network
        action_logits = self.policy_net(combined_state)

        return action_logits

    def get_action(self, observation):
        """
        Selects an action based on the current observation using the policy.

        Args:
            observation (dict): Dictionary containing 'cart_state' and 'pole_state'.

        Returns:
            int: The selected action (0 or 1).
        """
        # Extract states and convert to tensors, add batch dimension (unsqueeze(0))
        cart_state = torch.from_numpy(observation["cart_state"]).float().unsqueeze(0)
        pole_state = torch.from_numpy(observation["pole_state"]).float().unsqueeze(0)

        # --- Get Action Probabilities ---
        # No gradient needed for action selection phase
        with torch.no_grad():
            action_logits = self.forward(cart_state, pole_state)

        # Convert logits to probabilities using Softmax
        action_probs = F.softmax(action_logits, dim=-1)

        # --- Sample Action ---
        # Create a categorical distribution and sample an action
        dist = Categorical(action_probs)
        action = dist.sample()

        # --- Store Log Probability ---
        # Store the log probability of the chosen action for the REINFORCE update
        # We use the log_prob from the distribution object
        self.log_probs.append(dist.log_prob(action))

        return action.item()  # Return the action as a Python integer

    def store_reward(self, reward):
        """Stores the reward received after taking an action"""
        self.rewards.append(reward)

    def update_policy(self):
        """
        Updates the policy network using the REINFORCE algorithm based on
        the rewards and log probabilities collected during the episode.
        """
        # --- Calculate Discounted Returns (G_t) ---
        discounted_rewards = []
        cumulative_return = 0
        # Iterate backwards through the rewards
        for r in self.rewards[::-1]:
            cumulative_return = r + self.gamma * cumulative_return
            discounted_rewards.insert(0, cumulative_return)  # Prepend to maintain order

        # Convert rewards and log_probs to tensors
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # --- Normalize Discounted Rewards (Optional but Recommended) ---
        # Helps stabilize training by scaling rewards to have zero mean and unit variance
        eps = np.finfo(np.float32).eps.item()  # Small epsilon to avoid division by zero
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + eps
        )

        # --- Calculate Policy Loss ---
        policy_loss = []
        # For each step, loss is -log_prob * discounted_return
        for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * Gt)

        # Sum the losses for the episode
        # Ensure policy_loss is not empty before concatenating
        if not policy_loss:
            print(
                "Warning: Trying to update policy with no steps taken in the episode."
            )
            # Clear memory even if no update happens
            self.log_probs = []
            self.rewards = []
            return 0.0  # Return 0 loss if no update

        policy_loss = torch.cat(policy_loss).sum()

        # --- Perform Backpropagation and Optimization ---
        self.optimizer.zero_grad()  # Reset gradients
        policy_loss.backward()  # Calculate gradients
        # Optional: Gradient Clipping (prevents exploding gradients)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()  # Update network weights

        # --- Clear Memory for Next Episode ---
        self.log_probs = []
        self.rewards = []

        return policy_loss.item()  # Return the loss value for logging


# --- 4. Training Loop ---
if __name__ == "__main__":
    # --- Environment Setup ---
    # Use render_mode="human" to watch the agent train (slower)
    # Use render_mode=None for faster training without visualization
    render_training = False  # Set to True to watch
    env_render_mode = "human" if render_training else None
    env = CustomCartPoleEnv(render_mode=env_render_mode)

    # --- Agent Setup ---
    # Get dimensions from the environment's observation space
    cart_dim = env.observation_space["cart_state"].shape[0]
    pole_obs_dim = env.observation_space["pole_state"].shape[0]
    n_actions = env.action_space.n

    # Hyperparameters for the agent and training (TUNE THESE)
    latent_dim = 16  # Dimension of the latent space after encoding pole state
    learning_rate = 0.005  # Learning rate for the optimizer
    gamma = 0.98  # Discount factor for future rewards
    num_episodes = 2000  # Total number of episodes to train for
    max_steps_per_episode = 500  # Max steps before ending episode (like CartPole-v1)
    log_interval = 50  # Print progress every N episodes
    solved_reward_threshold = (
        475  # Target average reward over 100 episodes for "solved"
    )

    # Instantiate the agent
    agent = PolicyGradientAgent(
        cart_dim, pole_obs_dim, latent_dim, n_actions, lr=learning_rate, gamma=gamma
    )

    # --- Training Initialization ---
    episode_rewards = []  # List to store total reward per episode
    recent_scores = deque(maxlen=100)  # Efficiently store last 100 scores for averaging

    print(f"--- Starting Training ---")
    print(f"Environment: Custom CartPole")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Agent: REINFORCE with Encoder")
    print(f"  - Cart Dim: {cart_dim}")
    print(f"  - Pole Obs Dim: {pole_obs_dim}")
    print(f"  - Latent Dim: {latent_dim}")
    print(f"  - Actions: {n_actions}")
    print(f"Hyperparameters:")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Gamma: {gamma}")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Max Steps/Episode: {max_steps_per_episode}")
    print(f"-------------------------")

    # --- Main Training Loop ---
    for episode in range(num_episodes):
        # Reset environment for a new episode
        observation, info = env.reset()
        current_episode_reward = 0
        episode_loss = 0

        # Run the episode
        for step in range(max_steps_per_episode):
            # Agent selects action based on current observation
            action = agent.get_action(observation)

            # Environment takes action and returns results
            next_observation, reward, terminated, truncated, info = env.step(action)

            # Agent stores the reward for this step
            agent.store_reward(reward)

            # Update total reward for the episode
            current_episode_reward += reward

            # Move to the next state
            observation = next_observation

            # Optional: Add a small delay if rendering to make it watchable
            if render_training:
                time.sleep(0.01)

            # End episode if terminated (pole fell, cart out of bounds) or truncated (max steps reached)
            if terminated or truncated:
                break

        # --- End of Episode ---
        # Update the agent's policy network
        episode_loss = agent.update_policy()

        # Store and log results
        episode_rewards.append(current_episode_reward)
        recent_scores.append(current_episode_reward)
        average_score = np.mean(recent_scores)

        if (episode + 1) % log_interval == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} | Avg Reward (Last 100): {average_score:.2f} | "
                f"Current Reward: {current_episode_reward:.2f} | Loss: {episode_loss:.4f}"
            )

        # Check if the environment is considered solved
        if average_score >= solved_reward_threshold:
            print(
                f"\nEnvironment SOLVED at episode {episode + 1}! Average reward over last 100 episodes: {average_score:.2f}"
            )
            # Optionally save the trained model
            # model_save_path = 'custom_cartpole_agent_solved.pth'
            # torch.save(agent.state_dict(), model_save_path)
            # print(f"Model saved to {model_save_path}")
            break  # Stop training

    # --- End of Training ---
    env.close()  # Close the rendering window
    print("--- Training Finished ---")

    # --- Optional: Plotting Results ---
    if episode_rewards:
        plt.figure(figsize=(12, 6))
        plt.plot(episode_rewards, label="Reward per Episode", alpha=0.7)
        # Calculate and plot moving average
        moving_avg = [
            np.mean(episode_rewards[max(0, i - 99) : i + 1])
            for i in range(len(episode_rewards))
        ]
        plt.plot(
            moving_avg, label="100-Episode Moving Average", color="red", linewidth=2
        )
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress: Custom CartPole with Encoder Agent")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)  # Rewards are non-negative
        # Add a line for the solve threshold
        if len(episode_rewards) > 1:
            plt.hlines(
                solved_reward_threshold,
                0,
                len(episode_rewards) - 1,
                colors="green",
                linestyles="--",
                label=f"Solved Threshold ({solved_reward_threshold})",
            )
            plt.legend()  # Update legend to include threshold line
        plt.show()
    else:
        print("No episodes completed, skipping plot.")
