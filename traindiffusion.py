import jax, pickle
import jax.numpy as jnp
import optax
import json
from model import VAE
import flax.linen as nn
from jax import random
import utils as utils
import numpy as np
import tensorflow as tf
from flax import jax_utils
from diffusion import Unet, GaussianDiffusion, Trainer


def main():
    with open("config.json", "r") as stream:
        config = json.load(stream)

    vae_params = config["local_vae"]
    workdir = "runs/diff_" + input("run name: diff_")

    lat_size = utils.get_lat_size(config)
    lat_dim = utils.get_lat_dim(config)
    print(f"Latent Diffusion size: {lat_size}, {lat_size}, {lat_size}, {lat_dim}. Dimensionality: {(lat_size**3)*lat_dim}")

    print("Loading VAE local model...")
    resolution = vae_params["resolution"]
    vae_state = utils.get_vae_state(resolution, vae_params, random.PRNGKey(0), random.PRNGKey(0), "runs/" + config["local_run"], 0.01)
    init_data = jnp.ones((1, resolution, resolution, resolution, 1), jnp.float32)
    vae_local = VAE(**vae_params).bind({'params': vae_state.params}, init_data, random.PRNGKey(0))

    unet_config =config["unet"]
    model = Unet(
        dim=unet_config["dim"],
        dim_mults=unet_config["dim_mults"],
        out_dim=lat_dim
    )

    diffusion_model = GaussianDiffusion(
        model=model,
        image_size=lat_size,
        channels=lat_dim,
        timesteps=config["timesteps"],  # number of steps
        sampling_timesteps=config["sampling_timesteps"],  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type=config["loss_type"],  # L1 or L2
    )

    trainer = Trainer(
        diffusion_model=diffusion_model,
        train_batch_size=32,
        train_lr=2e-5,  # Ho et al have 2e-5
        train_num_steps=1000000,  # total training steps
        warmup_steps=1000,  # Ho et al have 5000
        workdir=workdir,
        vae_local=vae_local,
        h5path="/media/thomas/Data/latents/" + config["local_run"],
        verbose=True
    )

    trainer.train()


if __name__ == "__main__":
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    main()