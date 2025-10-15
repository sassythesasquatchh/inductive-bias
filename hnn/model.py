import equinox as eqx
import jax
import jax.numpy as jnp
from jax import grad, lax, vmap
from jaxtyping import Float

from common.classes import RolloutOutput
from util.config import Config


class PendulumHNN(eqx.Module):
    encoder: eqx.Module
    hnn: eqx.Module
    decoder: eqx.Module
    latent_dim: int
    config: Config

    def __init__(
        self,
        key,
        input_dim,
        latent_dim=2,
        config: Config = Config(),
        use_canonical_decoder=False,
        use_canonical_encoder=False,
    ):
        key1, key2, key3 = jax.random.split(key, 3)
        if use_canonical_encoder:
            self.encoder = CanonicalEncoder(config)
        else:
            self.encoder = eqx.nn.MLP(input_dim, latent_dim, 64, 2, key=key1)
        self.hnn = eqx.nn.MLP(latent_dim, 1, 32, 2, activation=jax.nn.tanh, key=key2)
        if use_canonical_decoder:
            self.decoder = CanonicalDecoder(config)
        else:
            self.decoder = eqx.nn.MLP(latent_dim, input_dim, 64, 2, key=key3)

        self.latent_dim = latent_dim
        self.config = config

    def encode(
        self, state: Float[jax.Array, "input_dim"]
    ) -> Float[jax.Array, "latent_dim"]:
        return self.encoder(state)

    def batch_encode(
        self, states: Float[jax.Array, "... input_dim"]
    ) -> Float[jax.Array, "... latent_dim"]:
        encoder = self.encoder
        for i in range(len(states.shape) - 1):
            encoder = vmap(encoder)
        return encoder(states)

    def decode(
        self, latent: Float[jax.Array, "latent_dim"]
    ) -> Float[jax.Array, "input_dim"]:
        return self.decoder(latent)

    def batch_decode(
        self, latent: Float[jax.Array, "... latent_dim"]
    ) -> Float[jax.Array, "... input_dim"]:
        decoder = self.decoder
        for i in range(len(latent.shape) - 1):
            decoder = vmap(decoder)
        return decoder(latent)

    def hamiltonian(
        self, latent: Float[jax.Array, "latent_dim"]
    ) -> Float[jax.Array, ""]:
        return self.hnn(latent).squeeze(-1)

    def time_derivative(
        self, latent: Float[jax.Array, "... latent_dim"]
    ) -> Float[jax.Array, "... latent_dim"]:
        # Compute the Hamiltonian derivatives
        grad_hamiltonian = grad(self.hamiltonian)

        for i in range(len(latent.shape) - 1):
            grad_hamiltonian = vmap(grad_hamiltonian)
        dH = grad_hamiltonian(latent)

        n = latent.shape[-1] // 2
        dq_dt = dH[..., n:]
        dp_dt = -dH[..., :n]
        return jnp.concatenate([dq_dt, dp_dt], axis=-1)

    def hnn_derivatives(
        self, latent_state: Float[jax.Array, "batch sequence_length latent_dim"]
    ) -> Float[jax.Array, "batch sequence_length latent_dim"]:
        grad_hamiltonian = grad(self.hamiltonian)
        dH = vmap(vmap(grad_hamiltonian))(latent_state)

        n = latent_state.shape[-1] // 2
        dq_dt = dH[..., n:]
        dp_dt = -dH[..., :n]
        return jnp.concatenate([dq_dt, dp_dt], axis=-1)

    @eqx.filter_jit
    def rollout(self, x: Float[jax.Array, "batch sequence_length observable_dim"]):
        latent_gt = self.batch_encode(x)

        # carry, x
        def step(latent, _):
            dlatent = self.config.DT * self.time_derivative(latent)
            next_latent = latent + dlatent
            next_obs = self.batch_decode(next_latent)
            # carry, output
            return next_latent, (next_latent, next_obs)

        initial_latent = latent_gt[:, 0, :]
        initial_obs = self.batch_decode(initial_latent)

        # Use lax.scan to unroll the simulation efficiently
        _, (latent_trajectory, rollout_obs) = lax.scan(
            step, initial_latent, None, length=x.shape[1] - 1
        )

        latent_trajectory = jnp.transpose(latent_trajectory, (1, 0, 2))
        rollout_obs = jnp.transpose(rollout_obs, (1, 0, 2))

        # Include the initial latent state at the beginning
        latent_rollout = jnp.concatenate(
            [initial_latent[:, jnp.newaxis, :], latent_trajectory], axis=1
        )
        rollout_obs = jnp.concatenate(
            [initial_obs[:, jnp.newaxis, :], rollout_obs], axis=1
        )

        def step(obs, _):
            latent = self.batch_encode(obs)
            dlatent = self.config.DT * self.time_derivative(latent)
            next_latent = latent + dlatent
            next_obs = self.batch_decode(next_latent)
            return next_obs, (next_latent, next_obs)

        initial_latent = latent_gt[:, 0, :]
        initial_obs = self.batch_decode(initial_latent)

        # Use lax.scan to unroll the simulation efficiently
        _, (latent_trajectory, end_to_end_obs) = lax.scan(
            step, initial_obs, None, length=x.shape[1] - 1
        )

        latent_trajectory = jnp.transpose(latent_trajectory, (1, 0, 2))
        end_to_end_obs = jnp.transpose(end_to_end_obs, (1, 0, 2))

        # Include the initial latent state at the beginning

        latent_end_to_end = jnp.concatenate(
            [initial_latent[:, jnp.newaxis, :], latent_trajectory], axis=1
        )
        obs_end_to_end = jnp.concatenate(
            [initial_obs[:, jnp.newaxis, :], end_to_end_obs], axis=1
        )

        return RolloutOutput(
            latent_gt=latent_gt,
            latent_rollout=latent_rollout,
            latent_end_to_end=latent_end_to_end,
            obs_gt=x,
            obs_rollout=rollout_obs,
            obs_end_to_end=obs_end_to_end,
        )


class CanonicalDecoder(eqx.Module):
    sampling_positions: jnp.ndarray

    def __init__(self, config: Config = Config()):
        self.sampling_positions = jnp.array(config.SAMPLING_POSITIONS) * config.L

    def __call__(
        self, latent: Float[jax.Array, "batch sequence_length 2"]
    ) -> Float[jax.Array, "batch sequence_length 3"]:
        n = latent.shape[-1] // 2
        theta = latent[..., :n]
        theta_dot = latent[..., n:]
        x = self.sampling_positions * jnp.sin(theta)
        y = -self.sampling_positions * jnp.cos(theta)
        linear_velocity = self.sampling_positions * theta_dot
        return jnp.concatenate([x, y, linear_velocity], axis=-1)


class CanonicalEncoder(eqx.Module):
    sampling_positions_excluding_zero: jnp.ndarray

    def __init__(self, config: Config = Config()):
        self.sampling_positions = jnp.array(config.SAMPLING_POSITIONS) * config.L

    def __call__(
        self, observables: Float[jax.Array, "batch sequence_length input_dim"]
    ) -> Float[jax.Array, "batch sequence_length 2"]:
        n = observables.shape[-1] // 3
        x = observables[..., :n]
        y = observables[..., n : 2 * n]
        linear_velocity = observables[..., 2 * n :]

        theta = jnp.mean(jnp.arctan2(x, -y))
        # theta_dot = jnp.mean(linear_velocity / self.sampling_positions_excluding_zero)
        theta_dot = jnp.mean(linear_velocity / self.sampling_positions)
        return jnp.stack([theta, theta_dot], axis=-1)
