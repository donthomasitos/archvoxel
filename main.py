import os
from functools import partial
import jax
import jax.numpy as jnp
from typing import Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from flax.training.common_utils import shard, shard_prng_key
from jax.nn.initializers import normal as normal_init
from flax.training import train_state
from flax import linen as nn
from tensorboardX import SummaryWriter
import tensorflow_datasets as tfds
import tensorflow as tf
import optax
import einops

IMAGE_CHANNELS = 3
IN_RES = 16
dataset_name = "cifar10"

class PixelShuffle(nn.Module):
    scale_factor: int

    def setup(self):
        self.layer = partial(
            einops.rearrange,
            pattern="b h w (h2 w2 c) -> b (h h2) (w w2) c",
            h2=self.scale_factor,
            w2=self.scale_factor
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layer(x)


# source: https://github.com/lweitkamp/GANs-JAX/blob/main/1_GANs.ipynb

run_name = input("Run name:")
writer = SummaryWriter("runs/" + run_name)
os.makedirs(os.path.join("results", run_name), exist_ok=True)

PRNGKey = jnp.ndarray

num_devices = jax.device_count()
print("num_devices", num_devices)

args = {}

# Set the parallel batch size.
args['seed'] = 1337
args['batch_size'] = num_devices * 256
args['epochs'] = 100
args['batch_size_p'] = args['batch_size'] // num_devices

args['true_label'] = jnp.ones((args['batch_size_p'], 1), dtype=jnp.int32)
args['false_label'] = jnp.zeros((args['batch_size_p'], 1), dtype=jnp.int32)


def set_range(batch):
    batch = tf.image.convert_image_dtype(batch['image'], tf.float32)
    batch = (batch - 0.5) / 0.5  # tanh range is -1, 1
    return batch


def downscale(batch):
    batch = tf.image.resize(batch, (IN_RES, IN_RES), tf.image.ResizeMethod.AREA)
    return batch


mnist_train = tfds.load(dataset_name)['train']
mnist_test = tfds.load(dataset_name)['test']
batches_in_epoch = len(mnist_train) // args['batch_size']

data_gen_ds = iter(tfds.as_numpy(
    mnist_train
        .map(set_range)
        .map(downscale)
        .cache()
        .shuffle(len(mnist_train), seed=args['seed'])
        .repeat()
        .batch(args['batch_size'])
))

data_gen_full = iter(tfds.as_numpy(
    mnist_train
        .map(set_range)
        .cache()
        .shuffle(len(mnist_train), seed=args['seed'])
        .repeat()
        .batch(args['batch_size'])
))

data_gen_test_ds = iter(tfds.as_numpy(
    mnist_test
        .map(set_range)
        .map(downscale)
        .cache()
        .shuffle(len(mnist_test), seed=args['seed'])
        .repeat()
        .batch(args['batch_size'])
))


class TrainState(train_state.TrainState):
    batch_stats: Any


def upscale_img(i):
    return jax.image.resize(i, (i.shape[0], 2 * IN_RES, 2 * IN_RES, IMAGE_CHANNELS), jax.image.ResizeMethod.NEAREST, antialias=False)


def downscale_img(i):
    return jax.image.resize(i, (i.shape[0], IN_RES, IN_RES, IMAGE_CHANNELS), jax.image.ResizeMethod.LINEAR, antialias=False)


class Generator(nn.Module):
    features: int = 32
    dtype: type = jnp.float32

    @nn.compact
    def __call__(self, input: jnp.ndarray, train: bool = True):
        batch_norm = partial(nn.BatchNorm, use_running_average=not train, axis=-1, scale_init=normal_init(0.02), dtype=self.dtype)

        x = input.reshape((args['batch_size_p'], IN_RES, IN_RES, IMAGE_CHANNELS))
        x = nn.Conv(features=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', kernel_init=normal_init(0.02), dtype=self.dtype)(x)
        x = nn.PReLU()(x)
        temp = x
        for B in range(4):  # residual blocks. In paper: 16
            r_t = x
            x = nn.Conv(features=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', kernel_init=normal_init(0.02), dtype=self.dtype)(x)
            x = batch_norm()(x)
            x = nn.PReLU()(x)
            x = nn.Conv(features=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', kernel_init=normal_init(0.02), dtype=self.dtype)(x)
            x = batch_norm()(x)
            x = x + r_t
        x = nn.Conv(features=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', kernel_init=normal_init(0.02), dtype=self.dtype)(x)
        x = batch_norm()(x)

        x = x + temp
        for _ in range(1):  # Per upscaling octave. In paper: 2
            x = nn.Conv(features=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', kernel_init=normal_init(0.02), dtype=self.dtype)(x)
            x = PixelShuffle(scale_factor=2)(x)
            x = nn.PReLU()(x)
        x = nn.Conv(features=IMAGE_CHANNELS, kernel_size=[1, 1], strides=[1, 1], padding='SAME', kernel_init=normal_init(0.02), dtype=self.dtype)(x)
        x = jnp.tanh(x)

        mean_desired = upscale_img(input)
        mean_actual = upscale_img(downscale_img(x))
        if False:
            # TODO: This does not work -> artifacts. but should avoid clipping.
            x = x - mean_actual
            clip = (x - jnp.clip(x + mean_desired, -1.0, 1.0)) / (x + 1e-5)
            # Max Clip in 2x2 quad
            clip_max = jnp.max(nn.max_pool(clip, (2,2), (2,2)), axis=-1, keepdims=True)
            x = x * (1.0 - upscale_img(clip_max))
            # scale x
            x = jnp.clip(x + mean_desired, -1.0, 1.0) # no clipping should occur now
        if True:
            x = jnp.clip(x - mean_actual + mean_desired, -1.0, 1.0)
        return x


class Discriminator(nn.Module):
    features: int = 32
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True):
        batch_norm = partial(nn.BatchNorm, use_running_average=not train, axis=-1, scale_init=normal_init(0.02), dtype=self.dtype)
        # 24 * 24
        x = nn.Conv(features=self.features, kernel_size=[3, 3], strides=[1, 1], padding='SAME', kernel_init=normal_init(0.02), dtype=self.dtype)(x)
        x = nn.leaky_relu(x, 0.2)

        def block(s, f, x):
            x = nn.Conv(features=f, kernel_size=[3, 3], strides=[s, s], padding='SAME', kernel_init=normal_init(0.02), dtype=self.dtype, use_bias=False)(x)
            x = batch_norm()(x)
            x = nn.leaky_relu(x, 0.2)
            return x

        # 12 * 12
        x = block(2, self.features, x)
        x = block(1, 2 * self.features, x)

        # 6 * 6
        x = block(2, 2 * self.features, x)
        x = block(1, 4 * self.features, x)

        # finisher
        x = nn.Dense(features=8 * self.features)(x)  # paper has 16*features, ergo twice the last block feature size
        x = nn.leaky_relu(x, 0.2)
        x = nn.Dense(features=1)(x)

        x = x.reshape((args['batch_size_p'], -1))
        return x


@partial(jax.pmap, axis_name='num_devices')
def g_step(g_state: TrainState,
           d_state: TrainState,
           key: PRNGKey,
           ds_data: jnp.ndarray):
    def loss_fn(params):
        generated_data, mutables = g_state.apply_fn(
            {'params': params, 'batch_stats': g_state.batch_stats},
            ds_data, mutable=['batch_stats'])

        logits, _ = d_state.apply_fn(
            {'params': d_state.params,
             'batch_stats': d_state.batch_stats},
            generated_data, mutable=['batch_stats'])

        loss = -jnp.mean(jnp.log(nn.sigmoid(logits)))
        return loss, mutables

    # Generate data with the Generator, critique it with the Discriminator.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, mutables), grads = grad_fn(g_state.params)

    # Average across the devices.
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    # Update the Generator through gradient descent.
    new_g_state = g_state.apply_gradients(grads=grads, batch_stats=mutables['batch_stats'])
    return new_g_state, loss


@partial(jax.pmap, axis_name='num_devices')
def d_step(g_state: TrainState,
           d_state: TrainState,
           real_data: jnp.ndarray,
           ds_data: jnp.ndarray,
           key: PRNGKey):
    generated_data, _ = g_state.apply_fn(
        {'params': g_state.params,
         'batch_stats': g_state.batch_stats},
        ds_data, mutable=['batch_stats'])

    def loss_fn(params):
        logits_real, mutables = d_state.apply_fn(
            {'params': params, 'batch_stats': d_state.batch_stats},
            real_data, mutable=['batch_stats'])

        logits_generated, mutables = d_state.apply_fn(
            {'params': params, 'batch_stats': mutables['batch_stats']},
            generated_data, mutable=['batch_stats'])

        real_loss = optax.sigmoid_binary_cross_entropy(logits_real, args['true_label']).mean()
        generated_loss = optax.sigmoid_binary_cross_entropy(logits_generated, args['false_label']).mean()
        loss = (real_loss + generated_loss) / 2
        return loss, mutables

    # Critique real and generated data with the Discriminator.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, mutables), grads = grad_fn(d_state.params)

    # Average cross the devices.
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    # Update the discriminator through gradient descent.
    new_d_state = d_state.apply_gradients(grads=grads, batch_stats=mutables['batch_stats'])

    return new_d_state, loss


@partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def create_state(rng, model_cls, input_shape):
    r"""Create the training state given a model class. """
    model = model_cls()
    tx = optax.adam(0.0001, b1=0.5, b2=0.999)
    variables = model.init(rng, jnp.ones(input_shape))
    state = TrainState.create(apply_fn=model.apply, tx=tx,
                              params=variables['params'], batch_stats=variables['batch_stats'])
    return state


@jax.pmap
def sample_g(g_state, ds_data):
    """Sample from the generator in evaluation mode."""
    generated_data = g_state.apply_fn(
        {'params': g_state.params,
         'batch_stats': g_state.batch_stats},
        ds_data, train=False, mutable=False)
    return generated_data


key = jax.random.PRNGKey(seed=args['seed'])
key_g, key_d, key = jax.random.split(key, 3)
key_g = shard_prng_key(key_g)
key_d = shard_prng_key(key_d)

d_state = create_state(key_d, Discriminator, (args['batch_size_p'], *next(data_gen_full).shape[1:]))
g_state = create_state(key_g, Generator, (args['batch_size_p'], IN_RES, IN_RES, IMAGE_CHANNELS))

g_input = next(data_gen_test_ds)
g_input = shard(g_input)
t = 0

out_inputs = g_input.reshape((-1, IN_RES, IN_RES, IMAGE_CHANNELS))
fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
for ax, image in zip(sum(axes.tolist(), []), out_inputs):
    ax.imshow(0.5 + 0.5 * np.squeeze(image))
    ax.set_axis_off()
fig.savefig(f"results/{run_name}/GAN_input.png")
plt.close(fig)

for epoch in range(1, args['epochs'] + 1):
    pbar = tqdm(range(batches_in_epoch))
    for batch in pbar:
        # Generate RNG keys for generator and discriminator.
        key, key_g, key_d = jax.random.split(key, 3)
        key_g = shard_prng_key(key_g)
        key_d = shard_prng_key(key_d)

        # Take a step with the generator.
        batch_data_ds = shard(next(data_gen_ds))
        g_state, g_loss = g_step(g_state, d_state, key_g, batch_data_ds)

        # Shard the data to possible devices.
        batch_data_full = shard(next(data_gen_full))

        # Take a step with the discriminator.
        batch_data_ds = shard(next(data_gen_ds))  # TODO maybe use batch_data_ds from above? -> No speed gain
        d_state, d_loss = d_step(g_state, d_state, batch_data_full, batch_data_ds, key_d)

        metrics = jax.device_get([g_loss[0], d_loss[1]])

        message = f"Epoch: {epoch: <2} | "
        message += f"Generator loss: {metrics[0]:.3f} | "
        message += f"Discriminator loss: {metrics[1]:.3f}"
        pbar.set_description(message)
        writer.add_scalar("g_loss", metrics[0], t)
        writer.add_scalar("d_loss", metrics[1], t)
        t += 1

    # Sample from the generator using the fixed input. We need to
    # reshape if we are working on multiple devices.
    sample = sample_g(g_state, g_input)
    sample = sample.reshape((-1, IN_RES * 2, IN_RES * 2, IMAGE_CHANNELS))

    # Next, plot the static samples, save the fig to disk.
    if epoch % 10 == 0:
        fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
        for ax, image in zip(sum(axes.tolist(), []), sample):
            ax.imshow(0.5 + 0.5 * image)
            ax.set_axis_off()
        fig.savefig(f"results/{run_name}/GAN_epoch_{epoch}.png")
        plt.close(fig)

''''# Load latest epoch results.
img = mpimg.imread(os.path.join('results', run_name, f'GAN_epoch_{epoch}.png'))

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()'''
