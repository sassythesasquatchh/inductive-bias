import gymnasium as gym
import numpy as np
from gymnasium import spaces
from os import path
from typing import Optional, Callable, Tuple, Any

"""
Custom Pendulum Environment using a Symplectic Integrator (Leapfrog/Verlet)
and a customizable observation space.
"""


# Default mapping: Standard Pendulum-v1 observation space
def default_map_state_to_observation(state: np.ndarray) -> np.ndarray:
    """Maps canonical state [theta, thetadot] to observation."""
    theta, thetadot = state
    return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)


# Define the shape of the default observation space
DEFAULT_OBSERVATION_SHAPE = (3,)


class SymplecticPendulumEnv(gym.Env):
    """
    Custom Pendulum Environment using Leapfrog integration.

    Canonical State: [theta, thetadot]
        theta: angle of the pendulum (normalized to [-pi, pi])
        thetadot: angular velocity of the pendulum

    Action: [torque]
        Torque applied to the pendulum.

    Observation: Customizable via `map_state_to_observation_fn`.
        Defaults to [cos(theta), sin(theta), thetadot].

    Args:
        g (float): Gravity constant.
        m (float): Mass of the pendulum bob.
        l (float): Length of the pendulum rod.
        dt (float): Time step for integration.
        max_speed (float): Maximum angular velocity.
        max_torque (float): Maximum applicable torque.
        map_state_to_observation_fn (Callable): Function mapping the
            internal state [theta, thetadot] to the observation vector
            returned by step() and reset().
        observation_space_shape (Tuple): Shape of the observation space
            produced by `map_state_to_observation_fn`.
        render_mode (str, optional): Rendering mode ('human', 'rgb_array').
        screen_dim (int): Dimension for rendering.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        g: float = 9.81,
        m: float = 1.0,
        l: float = 1.0,
        dt: float = 0.05,
        max_speed: float = 8.0,
        max_torque: float = 2.0,
        map_state_to_observation_fn: Callable[
            [np.ndarray], np.ndarray
        ] = default_map_state_to_observation,
        observation_space_shape: Tuple = DEFAULT_OBSERVATION_SHAPE,
        render_mode: Optional[str] = None,
        screen_dim: int = 500,
    ):
        super().__init__()

        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.screen_dim = screen_dim
        self.render_mode = render_mode

        self.state: Optional[np.ndarray] = (
            None  # Internal canonical state [theta, thetadot]
        )
        self._map_state_to_observation = map_state_to_observation_fn

        # Define action space
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        # Define observation space based on the provided mapping function
        # We assume the observation values are generally within [-inf, inf]
        # If your mapping has specific bounds, adjust low/high accordingly.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_space_shape, dtype=np.float32
        )

        # Rendering setup (optional, adapted from gymnasium's Pendulum)
        self.screen = None
        self.clock = None
        self.isopen = True

    def _dynamics(self, theta: float, torque: float) -> float:
        """Calculates angular acceleration."""
        # Formula derived from Pendulum-v1 source for consistency if needed,
        # or use standard physics: (torque - m*g*l*sin(theta)) / (m*l*l)
        # Using the one similar to gymnasium's discrete version for torque scaling:
        return (
            -3 * self.g / (2 * self.l) * np.sin(theta + np.pi)
            + 3.0 / (self.m * self.l**2) * torque
        )

    def _symplectic_step(
        self, state: np.ndarray, torque: float, dt: float
    ) -> np.ndarray:
        """
        Performs one step using the Leapfrog (Verlet) integrator.
        p = theta_dot * m * l^2 (angular momentum, though not directly used here)
        q = theta (angle)
        H = p^2 / (2*m*l^2) - m*g*l*cos(theta) + torque*theta (Hamiltonian + work)
        dq/dt = dH/dp = p / (m*l^2) = theta_dot
        dp/dt = -dH/dq = -m*g*l*sin(theta) + torque
        => d(theta_dot)/dt = (-m*g*l*sin(theta) + torque) / (m*l^2)
                            = -g/l * sin(theta) + torque / (m*l^2)

        Leapfrog steps:
        1. v_{n+1/2} = v_n + (dt/2) * a(x_n, u_n)
        2. x_{n+1}   = x_n + dt * v_{n+1/2}
        3. v_{n+1}   = v_{n+1/2} + (dt/2) * a(x_{n+1}, u_{n+1})
           (Note: We use u_n for acceleration at n+1 as torque is constant during dt)

        Let's use the Position Verlet form (simpler):
        1. x_{n+1} = x_n + v_n*dt + 0.5*a(x_n, u_n)*dt^2
        2. v_{n+1} = v_n + 0.5*(a(x_n, u_n) + a(x_{n+1}, u_n))*dt
        """
        theta, thetadot = state
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        # --- Leapfrog Implementation ---
        # Step 1: Update velocity by half step
        accel_n = self._dynamics(theta, torque)
        thetadot_half = thetadot + 0.5 * dt * accel_n

        # Step 2: Update position by full step using half-step velocity
        theta_new = theta + dt * thetadot_half
        theta_new = self._angle_normalize(theta_new)  # Normalize angle

        # Step 3: Update velocity by remaining half step using new position
        accel_n_plus_1 = self._dynamics(theta_new, torque)  # Use same torque
        thetadot_new = thetadot_half + 0.5 * dt * accel_n_plus_1

        # --- End Leapfrog ---

        # Clip speed
        thetadot_new = np.clip(thetadot_new, -self.max_speed, self.max_speed)

        return np.array([theta_new, thetadot_new], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action, update state using symplectic solver, return observation.
        """
        if self.state is None:
            raise ValueError("Environment not reset yet.")

        torque = action[0]
        dt = self.dt

        # Use symplectic integrator
        self.state = self._symplectic_step(self.state, torque, dt)
        theta, thetadot = self.state

        # Calculate reward (same as standard Pendulum-v1)
        reward = -(
            self._angle_normalize(theta) ** 2 + 0.1 * thetadot**2 + 0.001 * (torque**2)
        )
        reward = float(reward)  # Ensure reward is a float

        # Termination/Truncation - Pendulum is continuing, no natural termination
        terminated = False  # No specific goal state that terminates
        truncated = False  # Handled by TimeLimit wrapper usually

        # Map internal state to the observation the agent sees
        observation = self._map_state_to_observation(self.state)

        if self.render_mode == "human":
            self.render()

        # info dict can be empty or include internal state if needed for debugging
        info = {"internal_state": self.state.copy()}

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if options is None:
            high = np.array([np.pi, 1])
            low = -high
            self.state = self.np_random.uniform(low=low, high=high).astype(np.float32)
        else:
            # Example of using options to set a specific initial state
            if "initial_state" in options:
                self.state = np.array(options["initial_state"], dtype=np.float32)
                self.state[0] = self._angle_normalize(self.state[0])
            else:
                # Default random if options provided but no initial state
                high = np.array([np.pi, 1])
                low = -high
                self.state = self.np_random.uniform(low=low, high=high).astype(
                    np.float32
                )

        # Map internal state to observation
        observation = self._map_state_to_observation(self.state)
        info = {"internal_state": self.state.copy()}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _angle_normalize(self, x: float) -> float:
        """Normalize angle to the range [-pi, pi]."""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="human")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        surf = pygame.Surface((self.screen_dim, self.screen_dim))
        surf.fill((255, 255, 255))  # White background

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.1 * scale

        # Pendulum drawing
        if self.state is not None:
            theta, _ = self.state  # Use internal state for rendering
            # Adjust angle for pygame coordinates (0 is up)
            pygame_angle = theta + np.pi / 2

            # Rod endpoint
            end_x = offset + rod_length * np.cos(pygame_angle)
            end_y = offset - rod_length * np.sin(pygame_angle)  # Pygame y is inverted

            # Draw rod
            rod_coords = [
                (
                    offset + rod_width / 2 * np.sin(pygame_angle),
                    offset + rod_width / 2 * np.cos(pygame_angle),
                ),
                (
                    offset - rod_width / 2 * np.sin(pygame_angle),
                    offset - rod_width / 2 * np.cos(pygame_angle),
                ),
                (
                    end_x - rod_width / 2 * np.sin(pygame_angle),
                    end_y - rod_width / 2 * np.cos(pygame_angle),
                ),
                (
                    end_x + rod_width / 2 * np.sin(pygame_angle),
                    end_y + rod_width / 2 * np.cos(pygame_angle),
                ),
            ]
            gfxdraw.aapolygon(surf, rod_coords, (0, 0, 0))  # Black rod outline
            gfxdraw.filled_polygon(surf, rod_coords, (128, 128, 128))  # Gray rod fill

            # Draw axle
            gfxdraw.aacircle(surf, offset, offset, int(rod_width / 2), (0, 0, 0))
            gfxdraw.filled_circle(
                surf, offset, offset, int(rod_width / 2), (128, 128, 128)
            )

            # Draw bob (at the end of the rod)
            bob_radius = int(rod_width)
            gfxdraw.aacircle(
                surf, int(end_x), int(end_y), bob_radius, (0, 0, 0)
            )  # Black bob outline
            gfxdraw.filled_circle(
                surf, int(end_x), int(end_y), bob_radius, (200, 50, 50)
            )  # Red bob fill

        # Flip to match coordinate system (optional, depends on preference)
        surf = pygame.transform.flip(surf, False, True)

        if self.render_mode == "human":
            self.screen.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None  # Return None for human mode
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.screen = None
