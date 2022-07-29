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

    lat_size = config["scene_res"] // vae_params["resolution"]
    lat_dim = vae_params["z_channels"] * ((vae_params["resolution"] // (2 ** (len(vae_params["ch_mult"])-1))) ** 3)
    print(f"Latent Diffusion size: {lat_size}, {lat_size}, {lat_size}, {lat_dim}. Dimensionality: {(lat_size**3)*lat_dim}")

    if True:
        print("Loading VAE local model...")
        rng = random.PRNGKey(0)
        resolution = vae_params["resolution"]
        rng, key, z_rng = random.split(rng, 3)
        vae_state = utils.get_vae_state(resolution, vae_params, key, rng, "runs/" + config["local_run"], 0.01)
        init_data = jnp.ones((1, resolution, resolution, resolution, 1), jnp.float32)
        vae_local = VAE(**vae_params).bind({'params': vae_state.params}, init_data, z_rng)
    else:
        vae_local = None

    model = Unet(
        dim=config["dim"],
        dim_mults=(1, 2),
        out_dim=lat_dim
    )

    diffusion = GaussianDiffusion(
        model=model,
        image_size=lat_size,
        channels=lat_dim,
        timesteps=config["timesteps"],  # number of steps
        sampling_timesteps=config["sampling_timesteps"],  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type='l1',  # L1 or L2
    )

    trainer = Trainer(
        diffusion_model=diffusion,
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