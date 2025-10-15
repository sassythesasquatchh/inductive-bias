import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import vmap
from jaxtyping import Float

from util.dataset import JaxTrajectoryDataset

from .data import PendulumDataset
from .model import PendulumHNN


def compute_loss(
    model: PendulumHNN,
    batch: tuple[
        Float[jax.Array, "batch sequence_length observable_dim"],
        Float[jax.Array, "batch sequence_length latent_dim"],
    ],
):
    obs, canonicals = batch
    b, L, _ = obs.shape

    latent_states = vmap(vmap(model.encode))(obs)

    recon_states = vmap(vmap(model.decode))(latent_states)
    recon_loss = jnp.mean(jnp.square(obs - recon_states))

    # Compute Hamiltonian derivatives
    latent_derivs = model.hnn_derivatives(latent_states)

    # Finite difference over sequence dimension
    latent_finite_diff = jnp.gradient(latent_states, axis=1) / model.config.DT

    dynamics_loss = jnp.mean(jnp.square(latent_derivs - latent_finite_diff))

    # Canonicalisation loss
    n_dim = latent_states.shape[-1] // 2
    canonicalisation_loss = jnp.mean(
        jnp.square(latent_states[..., n_dim:] - latent_finite_diff[..., :n_dim])
    )

    total_loss = recon_loss + dynamics_loss + canonicalisation_loss
    return total_loss, (recon_loss, dynamics_loss, canonicalisation_loss)


def train(
    training_data_path: str,
    validation_data_path: str,
    num_epochs: int = 10000,
    batch_size: int = 64,
    sequence_length: int = 20,
    lr: float = 1e-3,
):
    print("Loading dataset...")
    training_dataset = PendulumDataset(training_data_path, sequence_length)
    print(f"Dataset created with {training_dataset.num_samples} sequences.")

    print("Loading dataset...")
    validation_dataset = PendulumDataset(validation_data_path, sequence_length)
    print(f"Dataset created with {validation_dataset.num_samples} sequences.")

    assert training_dataset.get_input_dim() == validation_dataset.get_input_dim(), (
        "Input data dimension mismatch between training and validation datasets."
    )
    assert training_dataset.get_latent_dim() == validation_dataset.get_latent_dim(), (
        "Latent data dimension mismatch between training and validation datasets."
    )

    key = jax.random.PRNGKey(0)
    model = PendulumHNN(
        key,
        training_dataset.get_input_dim(),
        training_dataset.get_latent_dim(),
        config=training_dataset.config,
        use_canonical_decoder=True,
        use_canonical_encoder=False,
    )

    optimizer = optax.chain(optax.clip(1.0), optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        (loss, components), grads = eqx.filter_value_and_grad(
            compute_loss, has_aux=True
        )(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, components

    @eqx.filter_jit
    def validate_step(model, batch):
        (loss, components) = compute_loss(model, batch)
        return loss, components

    # Training loop
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        batch = training_dataset.get_batch(subkey, batch_size)
        validate_batch = validation_dataset.get_batch(subkey, batch_size)
        model, opt_state, loss, (recon_loss, dyn_loss, h_reg) = train_step(
            model, opt_state, batch
        )
        val_loss, (val_recon_loss, val_dyn_loss, val_h_reg) = validate_step(
            model, validate_batch
        )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} "
                f"| Train Recon: {recon_loss:.4f} | Val Recon: {val_recon_loss:.4f} "
                f"| Train Dyn: {dyn_loss:.4f} | Val Dyn: {val_dyn_loss:.4f} "
                f"| Train Canon: {h_reg:.4f} | Val Canon: {val_h_reg:.4f}"
            )

    print("\nTraining complete.")

    return model


def main(args):
    from util.rollout import evaluate_rollout
    from util.test_continuity import test_continuity
    from util.visualisation import animate_trajectories

    model = train(
        training_data_path=args.train_path,
        validation_data_path=args.val_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.segment_length,
        lr=args.learning_rate,
    )

    visualisation_dataset = JaxTrajectoryDataset(
        data_path=args.visualisation_data_path, type="observed"
    )

    continuity_dataset = JaxTrajectoryDataset(
        data_path=args.continuity_data_path, type="observed"
    )

    visualisation_rollout = model.rollout(visualisation_dataset.data).to_torch()
    continuity_rollout = model.rollout(continuity_dataset.data).to_torch()
    evaluate_rollout(visualisation_rollout, visualisation_dataset)
    animate_trajectories(
        visualisation_rollout,
        visualisation_dataset.config,
        visualisation_dataset.traj_names,
        args.run_name,
    )

    test_continuity(continuity_rollout, continuity_dataset.initial_velocities)


if __name__ == "__main__":
    from util.pre_util import parse_args

    args = parse_args()
    args.run_name = f"{args.model}_{args.segment_length}"
    main(args)
