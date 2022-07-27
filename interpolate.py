import jax, pickle
import jax.numpy as jnp
import optax
import json
import flax.linen as nn
from jax import random
import utils as utils
from model import VAE, reparameterize
import numpy as np
from multiprocessing import Process, Pool

#jax.config.update('jax_platform_name', 'cpu')


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


def main():
    rng = random.PRNGKey(0)
    rng, key, z_rng = random.split(rng, 3)

    with open("config.json", "r") as stream:
        config = json.load(stream)

    l_vae_params = config["local_vae"]
    resolution = l_vae_params["resolution"]
    workdir = "runs/" + config["local_run"]

    @jax.jit
    def eval(images, z_rng, state):
        def eval_model(vae):
            recon_images, mean, logvar = vae(images, z_rng)
            return recon_images

        return nn.apply(eval_model, VAE(**l_vae_params))({'params': state.params})

    @jax.jit
    def get_latent(images, state):
        def eval_model(vae):
            mean, logvar = vae.encoder(images)
            return mean, logvar

        return nn.apply(eval_model, VAE(**l_vae_params))({'params': state.params})

    @jax.jit
    def decode(mean, logvar, z_rng, state):
        def eval_model(vae):
            z = reparameterize(z_rng, mean, logvar, is_training=False)
            recon_x = vae.decoder(z)
            return recon_x

        return nn.apply(eval_model, VAE(**l_vae_params))({'params': state.params})

    print("Loading model...")
    state = utils.get_vae_state(resolution, l_vae_params, key, rng, workdir, 0.01)

    pool = Pool(processes=16)
    proc_res = []

    print("Loading model datas...")
    model_datas = utils.load_model_data(resolution, load_pickled=True, min_size=resolution)
    rng = np.random.default_rng(2)
    start_img = np.expand_dims(utils.get_random_part(rng, model_datas[0], resolution), axis=[0, -1])
    final_img = np.expand_dims(utils.get_random_part(rng, model_datas[1], resolution), axis=[0, -1])

    proc_res.append(pool.apply_async(func=utils.save_image, args=(start_img, "interpolation/a_start_img.png", 1)))
    proc_res.append(pool.apply_async(func=utils.save_image, args=(final_img, "interpolation/z_final_img.png", 1)))

    print("Latent start/stop...")
    latent_s = get_latent(start_img, state)
    latent_f = get_latent(final_img, state)

    INTERPOL_STEPS = 16

    print("Exporting frames...")
    means, vars = [], []
    for it in range(INTERPOL_STEPS):
        mixv = it / (INTERPOL_STEPS-1)
        if False:
            latent_mean = latent_s[0] * mixv + latent_f[0] * (1.0 - mixv)
            latent_var  = latent_s[1] * mixv + latent_f[1] * (1.0 - mixv)
        else:
            pre_shape = latent_s[0].shape
            means.append(np.reshape(slerp(mixv, latent_s[0].flatten(), latent_f[0].flatten()), pre_shape))
            vars.append(np.reshape(slerp(mixv, latent_s[1].flatten(), latent_f[1].flatten()), pre_shape))
    means = np.concatenate(means, axis=0)
    vars = np.concatenate(vars, axis=0)

    recon_images = decode(means, vars, z_rng, state)

    proc_res.extend([pool.apply_async(func=vae_utils.save_image, args=(np.expand_dims(recon_images[it], axis=0), "interpolation/slerp_%d.png" % it, 1)) for it in range(INTERPOL_STEPS)])
    for p_c in proc_res:
        p_c.get(timeout=100)

if __name__ == "__main__":
    main()