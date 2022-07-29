import jax, pickle
import jax.numpy as jnp
import optax
import json, h5py, random, os
from tqdm.auto import tqdm
from model import VAE
import flax.linen as nn
import utils as utils
import numpy as np
from skimage.util.shape import view_as_blocks
import tensorflow as tf
from flax import jax_utils
from datetime import datetime

BATCH_SIZE = 128
DESIRED_SAMPLES = 5e6

# This generation is theoretically suboptimal, as it can generate duplicates. BUT:
# 1- calculating all possible augmentations quickly creates petabytes of examples
# 2- a subset heuristic would also be bad.
# So, due to 1. the collision risk is low and now we can steer the # of desired samples better.

def main():
    with open("config.json", "r") as stream:
        config = json.load(stream)

    vae_params = config["local_vae"]
    resolution = vae_params["resolution"]

    # This is the latent dimension of the space used for latent diffusion. Not the VAE latent space.
    lat_size = config["scene_res"] // vae_params["resolution"]
    lat_dim = vae_params["z_channels"] * ((vae_params["resolution"] // (2 ** (len(vae_params["ch_mult"])-1))) ** 3)
    print(f"Latent Diffusion size: {lat_size}, {lat_size}, {lat_size}, {lat_dim}. Dimensionality: {(lat_size ** 3) * lat_dim}")

    model_datas = utils.load_model_data(load_pickled=True, min_size=config["scene_res"])

    rng = jax.random.PRNGKey(0)
    rng, key, z_rng = jax.random.split(rng, 3)
    local_device_count = jax.local_device_count()

    def random_crop_mirror(random_int):
        data_id = random_int % len(model_datas)  # TODO sample proportionally to model size
        random_int = random_int // len(model_datas)
        rng = np.random.default_rng(random_int)
        x = utils.get_random_part(rng, model_datas[data_id], config["scene_res"], use_empty_probab=0.0)
        x_b = view_as_blocks(x, block_shape=(resolution, resolution, resolution))
        return np.expand_dims(x_b, -1)

    def augmentation(random_int):
        [image, ] = tf.numpy_function(random_crop_mirror, [random_int], [tf.float32], name="augmentation")
        image.set_shape([lat_size, lat_size, lat_size, resolution, resolution, resolution, 1])
        return image

    def prepare_tf_data(xs):
        """Convert a input batch from tf Tensors to numpy arrays."""
        def _prepare(x):
            # Use _numpy() for zero-copy conversion between TF and NumPy.
            x = x._numpy()  # pylint: disable=protected-access
            # reshape (host_batch_size, height, width, 3) to
            # (local_devices, device_batch_size, height, width, 3)
            return x.reshape((local_device_count,) + x.shape[1:])

        return jax.tree_map(_prepare, xs)

    random.seed(datetime.now())
    dataset = tf.data.Dataset.random(random.randint(0, 100000))
    dataset = dataset.map(augmentation)
    dataset = dataset.batch(local_device_count)
    dataset = dataset.prefetch(tf.data.AUTOTUNE).repeat()
    dataset = map(prepare_tf_data, dataset)
    dataset = jax_utils.prefetch_to_device(dataset, 2)

    print("Loading VAE local model...")
    vae_state = utils.get_vae_state(resolution, vae_params, key, rng, "runs/" + config["local_run"], 0.01)
    vae_model = VAE(**vae_params)

    @jax.jit
    def get_local_latent(images):
        def eval_model(vae, x):
            return vae.encoder(x)

        batch = jnp.reshape(images, (-1, *images.shape[-4:]))
        latent_splits = batch.shape[0] // BATCH_SIZE
        # print(f"Dividing {batch.shape[0]} into {latent_splits}. batch.shape: {batch.shape}")
        assert batch.shape[0] % BATCH_SIZE == 0, "number of voxels must be dividable by batch size OR implement non-full remaining batch..."
        batch = jnp.split(batch, latent_splits, axis=0)
        apply_fn = nn.apply(eval_model, vae_model)
        latents = []
        for s in range(latent_splits):
            split = batch[s]
            #print("in_data", split.shape)
            mean, logvar = apply_fn({'params': vae_state.params}, split)
            #print("mean", mean.shape)
            local_out = jnp.stack([mean, logvar], axis=1)  # xyzc are later merged into one latent variable. separate the data right after the batch dimension.
            res = jnp.reshape(local_out, (BATCH_SIZE, 2 * lat_dim))
            #print("res", res.shape)
            latents.append(res)
        return jnp.concatenate(latents, axis=0)

    assert BATCH_SIZE % local_device_count == 0, "Batch size must be divisable by device count for parallelization"
    get_local_latent_p = jax.pmap(get_local_latent)
    save_path = "/media/thomas/Data/latents/" + config["local_run"]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with h5py.File(save_path + '/latents_meanvar.hdf5', 'a') as out_file:
        if "latents" not in out_file:
            out_file.create_dataset("latents", shape=(0, lat_size, lat_size, lat_size, 2 * lat_dim), maxshape=(None, lat_size, lat_size, lat_size, 2 * lat_dim), compression='gzip')
        num_generated = out_file["latents"].shape[0]

    latens_write_buffer = []
    with tqdm(initial=num_generated, total=DESIRED_SAMPLES) as pbar:
        while num_generated < DESIRED_SAMPLES:
            batch = next(dataset)
            latents = get_local_latent_p(batch)
            latents = jnp.reshape(latents, (local_device_count, lat_size, lat_size, lat_size, 2 * lat_dim))

            latens_write_buffer.append(latents)
            if len(latens_write_buffer) == min(DESIRED_SAMPLES, 100):
                write_buf = jnp.concatenate(latens_write_buffer, axis=0)
                print("Saving... (Dont close)")
                with h5py.File(save_path + '/latents_meanvar.hdf5', 'a') as out_file:
                    old_file_size = out_file["latents"].shape[0]
                    out_file["latents"].resize(old_file_size + write_buf.shape[0], axis=0)
                    out_file["latents"][old_file_size:] = write_buf
                latens_write_buffer = []

            num_generated += 1
            pbar.update(1)


if __name__ == "__main__":
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')
    main()