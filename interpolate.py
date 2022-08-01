import jax, pickle
import json
import flax.linen as nn
from jax import random
import utils as utils
from model import VAE, reparameterize
import numpy as np
from multiprocessing import Process, Pool
import matplotlib, time
#jax.config.update('jax_platform_name', 'cpu')

def main():
    rng = random.PRNGKey(0)
    rng, key, z_rng = random.split(rng, 3)
    start_time = time.time()

    with open("config.json", "r") as stream:
        config = json.load(stream)

    l_vae_params = config["local_vae"]
    resolution = l_vae_params["resolution"]
    workdir = "runs/" + config["local_run"]

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
    model_datas = utils.load_model_data(load_pickled=True, min_size=resolution)
    rng = np.random.default_rng(3)
    start_img = np.expand_dims(utils.get_random_part(rng, model_datas[0], resolution, use_empty_probab=0.0), axis=[0, -1])
    final_img = np.expand_dims(utils.get_random_part(rng, model_datas[1], resolution, use_empty_probab=0.0), axis=[0, -1])

    proc_res.append(pool.apply_async(func=utils.save_image, args=(start_img, "interpolation/a_start_img.png", 1)))
    proc_res.append(pool.apply_async(func=utils.save_image, args=(final_img, "interpolation/z_final_img.png", 1)))

    print("Latent start/stop...")
    latent_s = get_latent(start_img, state)
    latent_f = get_latent(final_img, state)

    INTERPOL_STEPS = 32

    print("Exporting frames...")
    means, vars = [], []
    for it in range(INTERPOL_STEPS):
        mixv = it / (INTERPOL_STEPS-1)
        if False:
            latent_mean = latent_s[0] * mixv + latent_f[0] * (1.0 - mixv)
            latent_var  = latent_s[1] * mixv + latent_f[1] * (1.0 - mixv)
        else:
            pre_shape = latent_s[0].shape
            means.append(np.reshape(utils.slerp(mixv, latent_s[0].flatten(), latent_f[0].flatten()), pre_shape))
            vars.append(np.reshape(utils.slerp(mixv, latent_s[1].flatten(), latent_f[1].flatten()), pre_shape))
    means = np.concatenate(means, axis=0)
    vars = np.concatenate(vars, axis=0)

    recon_images = decode(means, vars, z_rng, state)

    proc_res.extend([pool.apply_async(func=utils.save_image, args=(np.expand_dims(recon_images[it], axis=0), "interpolation/slerp_%d.png" % it, 1)) for it in range(INTERPOL_STEPS)])
    for p_c in proc_res:
        p_c.get(timeout=100)

    print("Duration:", time.time() - start_time)


if __name__ == "__main__":
    main()