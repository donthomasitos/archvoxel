import jax, pickle
import json
import flax.linen as nn
from jax import random
import utils as utils
from model import VAE, reparameterize
import numpy as np
from multiprocessing import Process, Pool
import matplotlib, time, optax
import jax.numpy as jnp
from diffusion import Unet, GaussianDiffusion
from flax.training import train_state
from flax import jax_utils

#jax.config.update('jax_platform_name', 'cpu')

def main():
    rng = random.PRNGKey(0)
    rng, key, z_rng = random.split(rng, 3)
    start_time = time.time()

    with open("config.json", "r") as stream:
        config = json.load(stream)

    vae_params = config["local_vae"]

    lat_size = utils.get_lat_size(config)
    lat_dim = utils.get_lat_dim(config)
    print(f"Latent Diffusion size: {lat_size}, {lat_size}, {lat_size}, {lat_dim}. Dimensionality: {(lat_size ** 3) * lat_dim}")

    print("Loading VAE local model...")
    resolution = vae_params["resolution"]
    vae_state = utils.get_vae_state(resolution, vae_params, random.PRNGKey(0), random.PRNGKey(0), "runs/" + config["local_run"], 0.01)
    init_data = jnp.ones((1, resolution, resolution, resolution, 1), jnp.float32)
    vae_local = VAE(**vae_params).bind({'params': vae_state.params}, init_data, random.PRNGKey(0))

    unet_config = config["unet"]
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

    image_size = diffusion_model.image_size
    init_data = jnp.ones((1, image_size, image_size, image_size, 2 * diffusion_model.channels))  # 2* bc of mean&logvar
    variables = diffusion_model.init(rng, rng, init_data)['params']

    diff_state = train_state.TrainState.create(
        apply_fn=diffusion_model.apply,
        params=variables,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.linear_schedule(1e-9, 0.1, 1000)))  # TODO shitty code duplication
    )
    diff_state = utils.restore_checkpoint(diff_state, "runs/" + config["diff_run"])
    diff_state = jax_utils.replicate(diff_state)
    model_bound = diffusion_model.bind({'params': jax_utils.unreplicate(diff_state).params})

    @jax.jit
    def decode(mean, logvar, z_rng, state):
        def eval_model(vae):
            z = reparameterize(z_rng, mean, logvar, is_training=False)
            recon_x = vae.decoder(z)
            return recon_x

        return nn.apply(eval_model, VAE(**vae_params))({'params': state.params})

    pool = Pool(processes=16)
    proc_res = []

    print("Picking random latents...")
    start_lat = jax.random.normal(jax.random.PRNGKey(0), diffusion_model.get_img_shape(1))
    final_lat = jax.random.normal(jax.random.PRNGKey(1), diffusion_model.get_img_shape(1))

    INTERPOL_STEPS = 32
    interp_latents = []
    print("Exporting frames...")
    for it in range(INTERPOL_STEPS):
        mixv = it / (INTERPOL_STEPS-1)
        if False:
            latent_mean = latent_s[0] * mixv + latent_f[0] * (1.0 - mixv)
            latent_var  = latent_s[1] * mixv + latent_f[1] * (1.0 - mixv)
        else:
            pre_shape = start_lat.shape
            interp_latents.append(jnp.reshape(utils.slerp(mixv, start_lat.flatten(), final_lat.flatten()), pre_shape))
    interp_latents = jnp.concatenate(interp_latents, axis=0)
    diff_x0 = model_bound.ddim_sample(interp_latents, None)
    #utils.latent_to_voxels(diff_x0, vae_local)

    #proc_res.extend([pool.apply_async(func=utils.save_image, args=(np.expand_dims(recon_images[it], axis=0), "interpolation/slerp_%d.png" % it, 1)) for it in range(INTERPOL_STEPS)])
    for p_c in proc_res:
        p_c.get(timeout=100)

    print("Duration:", time.time() - start_time)


if __name__ == "__main__":
    main()