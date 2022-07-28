import jax
import jax.numpy as jnp
import tensorflow as tf
import os, time, random
from tensorboardX import SummaryWriter
import flax.linen as nn
from jax import random
import utils as utils
import matplotlib
import numpy as np
from flax import jax_utils
from model import VAE
import glob, os, shutil
from multiprocessing import Process, Pool
import json, functools
from functools import partial
from tqdm.auto import tqdm
from flax.training import checkpoints
from jax.config import config
#config.update("jax_debug_nans", True)

# ref: https://github.com/google/flax/blob/main/examples/vae/train.py

NUM_EVAL_EXAMPLES = 4
# if >0.5, balance towards filled voxels. Got bloated/expanded result.
# Did not encounter "stuck in local minimum of emtpyness". From https://arxiv.org/pdf/1608.04236.pdf
FILLED_WEIGHT = 0.5

# Multi GPU code: https://github.com/google/flax/blob/main/examples/imagenet/train.py
# VAE Tricks : https://nbviewer.org/github/neurokinetikz/deep-pensieve/blob/master/Deep%20Pensieve.ipynb
# Y should be up
# Resolve memory leak: sudo fuser -v /dev/nvidia*  ;   sudo kill -9 XXXXX
# TODO encode block and neighbors and feed decoder with encode(embed(bock) + embed(neighbors)) to recreate input. should result in less seams.


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy(probs, labels):
    #return jnp.sum(jnp.square(probs - labels))  # MSE

    probs = jnp.clip(probs, 1e-7, 1.0 - 1e-7)  # TODO if fused with log_sigmoid, the clipping is not necessary
    return - jnp.sum(FILLED_WEIGHT * labels * jnp.log(probs) + (1.0 - FILLED_WEIGHT) * (1 - labels) * jnp.log(1 - probs))


def compute_metrics(recon_x, x, mean, logvar):
    bce_loss = binary_cross_entropy(recon_x, x).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    metrics = {
        'bce': bce_loss,
        'kld': kld_loss,
        'loss': bce_loss + kld_loss
    }
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics


def main():
    run_name = "vae_" + input("run name: vae_")

    with open("config.json", "r") as stream:
        config = json.load(stream)

    l_vae_params = config["local_vae"]
    resolution = l_vae_params["resolution"]

    @jax.jit
    def train_step(state, batch, z_rng):
        def loss_fn(params):
            recon_x, mean, logvar = VAE(**l_vae_params).apply({'params': params}, batch, z_rng)
            bce_loss = binary_cross_entropy(recon_x, batch).mean()
            kld_loss = kl_divergence(mean, logvar).mean()
            # https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
            loss = bce_loss + config["kl_weight"] * kld_loss #* jnp.clip(state.step/5000, 0.0001, 1.0)
            return loss, (bce_loss, kld_loss)

        grads, (bce_loss, kld_loss) = jax.grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')
        return state.apply_gradients(grads=grads), (bce_loss, kld_loss)

    @jax.jit
    def eval(images, z, z_rng, params):
        def eval_model(vae):
            is_training = False
            recon_images, mean, logvar = vae(images, z_rng, is_training)
            comparison = jnp.concatenate([images[:NUM_EVAL_EXAMPLES], recon_images[:NUM_EVAL_EXAMPLES]])
            generate_images = vae.decoder(z)
            metrics = compute_metrics(recon_images, images, mean, logvar)
            return metrics, comparison, generate_images

        return nn.apply(eval_model, VAE(**l_vae_params))({'params': params})

    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')
    model_datas = utils.load_model_data(load_pickled=True, min_size=resolution)

    if config["batch_size"] % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    def random_crop_mirror(random_int):
        data_id = random_int % len(model_datas)  # TODO sample proportionally to model size
        random_int = random_int // len(model_datas)
        rng = np.random.default_rng(random_int)
        x = utils.get_random_part(rng, model_datas[data_id], resolution)
        return np.expand_dims(x, -1)

    def augmentation(random_int):
        [image, ] = tf.numpy_function(random_crop_mirror, [random_int], [tf.float32], name="augmentation")
        image.set_shape([resolution, resolution, resolution, 1])
        return image

    dataset = tf.data.Dataset.random(seed=int(time.time()))
    dataset = dataset.map(augmentation)
    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.prefetch(tf.data.AUTOTUNE).repeat()
    dataset = map(utils.prepare_tf_data, dataset)
    dataset = jax_utils.prefetch_to_device(dataset, 2)

    test_out_ds = next(dataset)

    in_shape = test_out_ds.shape[2:]  # skip device and batch
    #test_out_ds = np.reshape(test_out_ds, (-1, *in_shape))
    print("Occupancy Ratio:", np.mean(test_out_ds), "Inshape:", in_shape)
    os.makedirs(f"results/{run_name}", exist_ok=True)
    workdir = "runs/" + run_name
    writer = SummaryWriter(workdir)
    shutil.copyfile("config.json", workdir + "/config.json")

    rng = random.PRNGKey(0)
    rng, key = random.split(rng)

    state = utils.get_vae_state(resolution, l_vae_params, key, rng, workdir, config["learning_rate"], verbose=True)
    ema = state.params
    ema = utils.restore_checkpoint(ema, workdir + "_ema")
    # state.replace(tx=optax.adam(1e-3))

    state = jax_utils.replicate(state)
    rng, z_key, eval_rng = random.split(rng, 3)
    eval_rng = jnp.stack([eval_rng, eval_rng])
    z_size = l_vae_params["resolution"] // (2 ** (len(l_vae_params["ch_mult"]) - 1))
    z = random.normal(z_key, (jax.local_device_count(), NUM_EVAL_EXAMPLES,) + (z_size, z_size, z_size, l_vae_params["z_channels"]))
    print("Latent size: ", z_size * z_size*z_size*l_vae_params["z_channels"])

    train_step_p = jax.pmap(train_step, axis_name="batch")
    eval_p = jax.pmap(eval, axis_name="batch")

    pool = Pool(processes=4)
    step = int(state.step[0])

    with tqdm(initial=step, total=200e3) as pbar:
        while True:
            batch = next(dataset)
            rng, key = random.split(rng)
            key = jnp.stack([key, key])
            state, train_stats = train_step_p(state, batch, key)
            train_bce = np.mean(train_stats[0])
            train_kld = np.mean(train_stats[1])

            if step != 0 and step % 100 == 0:
                # follow https://arxiv.org/pdf/2207.04316.pdf with 0.99 every 100 steps
                e_decay = 0.99
                ema = jax.tree_map(lambda e, p: e * e_decay + (1 - e_decay) * p, ema, jax_utils.unreplicate(state).params)

            if step != 0 and step % 1000 == 0:
                utils.save_checkpoint(state, workdir)
                checkpoints.save_checkpoint(workdir + "_ema", ema, step, keep=2)

                # no difference
                #_, comparison, sample = eval_p(test_out_ds, z, eval_rng, jax_utils.replicate(ema))
                #pool.apply_async(func=utils.save_image, args=(np.array(comparison[0]), f'results/{run_name}/ema_reconstruction_{state.step[0]}.png', NUM_EVAL_EXAMPLES))
                #pool.apply_async(func=utils.save_image, args=(np.array(sample[0]), f'results/{run_name}/ema_sample_{state.step[0]}.png', NUM_EVAL_EXAMPLES))

                metrics, comparison, sample = eval_p(test_out_ds, z, eval_rng, state.params)
                pool.apply_async(func=utils.save_image, args=(np.array(comparison[0]), f'results/{run_name}/reconstruction_{state.step[0]}.png', NUM_EVAL_EXAMPLES))
                pool.apply_async(func=utils.save_image, args=(np.array(sample[0]), f'results/{run_name}/sample_{state.step[0]}.png', NUM_EVAL_EXAMPLES))

                print('step: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}, Train: BCE: {:.4f}, KLD: {:.4f}'.format(step, np.mean(metrics['loss']), np.mean(metrics['bce']),
                                                                                                                          np.mean(metrics['kld']), train_bce, train_kld))
                writer.add_scalar("vae/loss", np.mean(metrics['loss']), state.step[0])
                writer.add_scalar("vae/bce", np.mean(metrics['bce']), state.step[0])
                writer.add_scalar("vae/kld", np.mean(metrics['kld']), state.step[0])

            writer.add_scalar("vae/train_bce", train_bce, state.step[0])
            writer.add_scalar("vae/train_kld", train_kld, state.step[0])

            pbar.set_description(f'BCE:{train_bce:.3f}, KLD:{train_kld:.3f}')

            step += 1
            pbar.update(1)


if __name__ == "__main__":
    main()
