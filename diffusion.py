import jax, pickle
from tensorboardX import SummaryWriter
import jax.numpy as jnp
import optax
import flax.linen as nn
import utils as utils
import numpy as np
import math, time, os
from inspect import isfunction
from collections import namedtuple
from functools import partial
from einops import rearrange, reduce
from tqdm.auto import tqdm
from flax.training import train_state
from flax import jax_utils
import tensorflow_io as tfio
import tensorflow as tf
import binvox_rw
from flax.training import checkpoints

# Reference (Torch), 17.07: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/2b742dd2ccd899f24111b0fba4d3726e65b8c88e/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

SQRT_2 = 2.0 ** 0.5

# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# small helper modules

class Upsample(nn.Module):
    dim_out: int

    @nn.compact
    def __call__(self, x):
        x = jax.image.resize(x, shape=(x.shape[0], 2 * x.shape[-4], 2 * x.shape[-3], 2 * x.shape[-2], x.shape[-1]), method=jax.image.ResizeMethod.NEAREST)
        x = nn.Conv(self.dim_out, (3, 3, 3))(x)
        return x


class Downsample(nn.Module):
    dim_out: int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.dim_out, (4, 4, 4), strides=2)(x)


class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        w = self.param('w', nn.initializers.normal(stddev=1.0), [self.dim // 2])
        f = 2 * jnp.pi * jnp.einsum("b,w->bw", x, w)
        return jnp.concatenate([jnp.cos(f), jnp.sin(f), jnp.reshape(x, (-1,1))], axis=-1)


class Resnet(nn.Module):
    dim: int
    dim_out: int
    groups: int = 32

    @nn.compact
    def __call__(self, x, time_emb=None):
        assert x.shape[-1] == self.dim
        scale_shift = None
        if exists(time_emb):
            time_emb = nn.Dense(2 * self.dim_out)(nn.swish(time_emb))
            time_emb = rearrange(time_emb, 'b c -> b 1 1 1 c')
            scale_shift = jnp.split(time_emb, 2, axis=-1)

        h = nn.GroupNorm(num_groups=self.groups)(x)
        h = nn.swish(h)
        h = nn.Conv(self.dim_out, (3, 3, 3))(h)

        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        h = nn.GroupNorm(num_groups=self.groups)(h)
        h = nn.swish(h)
        h = nn.Conv(self.dim_out, (3, 3, 3))(h)

        if self.dim != self.dim_out:
            return (h + nn.Conv(self.dim_out, kernel_size=(1, 1, 1))(x)) / SQRT_2
        else:
            return (h + x) / SQRT_2


class LinearAttention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    def setup(self):
        self.scale = self.dim_head ** -0.5
        self.hidden_dim = self.dim_head * self.heads

    @nn.compact
    def __call__(self, input):
        b, x, y, z, c = input.shape
        qkv = jnp.split(nn.Conv(self.hidden_dim * 3, kernel_size=(1, 1, 1), use_bias=False)(input), 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y z (h c) -> b h c (x y z)', h=self.heads), qkv)

        q = nn.softmax(q, axis=-2)  # over c
        k = nn.softmax(k, axis=-1)  # over (x y z)

        q = q * self.scale
        context = jnp.einsum('b h d n, b h c n -> b h d c', k, v)

        out = jnp.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b x y z (h c)', h=self.heads, x=y, y=y, z=z)
        out = nn.Conv(self.dim, kernel_size=(1, 1, 1))(out)
        return nn.LayerNorm()(out)


class Attention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32
    scale = 16

    def setup(self):
        self.hidden_dim = self.dim_head * self.heads

    @nn.compact
    def __call__(self, input):
        b, x, y, z, c = input.shape
        qkv = jnp.split(nn.Conv(self.hidden_dim * 3, kernel_size=(1, 1, 1), use_bias=False)(input), 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y z (h c) -> b h c (x y z)', h=self.heads), qkv)

        q = jax.nn.standardize(q)
        k = jax.nn.standardize(k)

        sim = jnp.einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = nn.softmax(sim, axis=-1)

        out = jnp.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b x y z (h d)', x=x, y=y, z=z)
        return nn.Conv(self.dim, kernel_size=(1, 1, 1))(out)


class SelfAtt(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.GroupNorm(num_groups=32)(x)
        b, _, _, _, c = x.shape
        preshape = x.shape
        h = jnp.reshape(h, (b, -1, c))
        h = nn.SelfAttention(num_heads=4)(h)
        h = jnp.reshape(h, preshape)
        return (x + h) / SQRT_2


class Unet(nn.Module):
    dim: int
    out_dim: int
    dim_mults: tuple

    resnet_block_groups = 32

    def setup(self):
        self.init_dim = self.dim

        self.dims = [self.init_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))

        # layers
        self.num_resolutions = len(self.in_out)
        self.time_dim = self.dim * 4

    @nn.compact
    def __call__(self, x, time):
        x = nn.Conv(self.init_dim, kernel_size=(7, 7, 7), padding="SAME")(x)  # TODO ref has kernel 3
        r = jnp.copy(x)

        # time embedding
        t = SinusoidalPosEmb(16)(time)
        t = nn.Dense(self.time_dim)(t)
        t = nn.swish(t)
        t = nn.Dense(self.time_dim)(t)

        resnet = partial(Resnet, groups=self.resnet_block_groups)

        h = []

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (self.num_resolutions - 1)
            #x = resnet(dim_in, dim_in)(x, t)
            #h.append(x)
            x = resnet(dim_in, dim_in)(x, t)
            x = SelfAtt()(x)
            h.append(x)

            if is_last:
                x = nn.Conv(dim_out, (3, 3, 3))(x)
            else:
                x = Downsample(dim_out)(x)

        mid_dim = self.dims[-1]
        x = resnet(mid_dim, mid_dim)(x, t)
        x = SelfAtt()(x)
        x = resnet(mid_dim, mid_dim)(x, t)

        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out)):
            is_last = ind == (len(self.in_out) - 1)
            #x = jnp.concatenate((x, h.pop()), axis=-1)
            #x = resnet(dim_out + dim_in, dim_out)(x, t)
            x = jnp.concatenate((x, h.pop()), axis=-1)
            x = resnet(dim_out + dim_in, dim_out)(x, t)
            x = SelfAtt()(x)

            if is_last:
                x = nn.Conv(dim_in, (3, 3, 3))(x)
            else:
                x = Upsample(dim_in)(x)

        if True:  # Phil Wang
            x = jnp.concatenate((x, r), axis=-1)
            x = resnet(dim=self.dim * 2, dim_out=self.dim)(x, t)
            return nn.Conv(self.out_dim, kernel_size=(1, 1, 1))(x)
        else:  # from DDMI source. plateau after 30 steps.
            x = nn.GroupNorm(32)(x)
            x = nn.swish(x)
            return nn.Conv(self.out_dim, kernel_size=(3, 3, 3))(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a[t]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# schedules from: https://github.com/CompVis/latent-diffusion/blob/171cf29fb54afe048b03ec73da8abb9d102d0614/ldm/modules/diffusionmodules/util.py
def linear_beta_schedule(timesteps, linear_start=1e-4, linear_end=2e-2):
    return np.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=np.float64) ** 2.0


def cosine_beta_schedule(timesteps, cosine_s=8e-3):
    timesteps = np.arange(timesteps + 1, dtype=np.float64) / timesteps + cosine_s
    alphas = timesteps / (1 + cosine_s) * np.pi / 2
    alphas = np.cos(alphas) ** 2.0
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    model: nn.Module = None
    image_size: int = None
    channels: int = None
    timesteps: int = 1000
    sampling_timesteps: int = None
    loss_type: str = 'l1'
    objective:str = 'pred_noise'
    beta_schedule = 'cosine'
    ddim_sampling_eta = 1.

    def setup(self):
        super().__init__()
        assert self.objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if self.beta_schedule == 'linear':
            betas = linear_beta_schedule(self.timesteps)
        elif self.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.timesteps)
        else:
            raise ValueError(f'unknown beta schedule {self.beta_schedule}')

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        assert betas.dtype == np.float64

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        def to_jnp(v):
            return jnp.array(v, dtype=jnp.float32)

        # sampling related parameters
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        self.alphas_cumprod_prev = to_jnp(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_jnp(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_jnp(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_jnp(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_jnp(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_jnp(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_jnp(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_jnp(np.log(np.maximum(1e-20, posterior_variance)))
        self.posterior_mean_coef1 = to_jnp((betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.posterior_mean_coef2 = to_jnp(((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t):
        model_output = self.model(x, t)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)
            return ModelPrediction(pred_noise, x_start)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output
            return ModelPrediction(pred_noise, x_start)

        assert False

    def p_mean_variance(self, x, t):
        preds = self.model_predictions(x, t)
        x_start = preds.pred_x_start
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, rng, t: int):
        n0 = rng
        batched_times = jnp.full((x.shape[0],), t, dtype=jnp.int32)
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=batched_times)
        noise = jax.random.normal(n0, x.shape) if t > 0 else 0.  # no noise if t == 0
        return model_mean + jnp.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, rng, shape):
        n0, n1 = jax.random.split(rng)

        img = jax.random.normal(n0, shape)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step'):
            img = self.p_sample(img, n1, t)
        return img

    # https://github.com/ermongroup/ddim/blob/main/functions/denoising.py
    def ddim_sample(self, rng, shape):
        n0, rng = jax.random.split(rng)
        batch = shape[0]
        times = jnp.linspace(0., self.num_timesteps, num=self.sampling_timesteps + 2)[:-1]
        times = list(reversed(times.astype(jnp.int32).tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = jax.random.normal(n0, shape)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]
            time_cond = jnp.full((batch,), time, dtype=jnp.int32)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond)

            sigma = self.ddim_sampling_eta * jnp.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            c = jnp.sqrt((1 - alpha_next) - sigma ** 2)
            n1, rng = jax.random.split(rng)
            noise = jax.random.normal(n1, img.shape) if time_next > 0 else 0.
            img = x_start * jnp.sqrt(alpha_next) + c * pred_noise + sigma * noise
        return img

    def sample(self, rng, batch_size=16):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(rng, (batch_size, self.image_size, self.image_size, self.image_size, self.channels))

    def interpolate(self, x1, x2, rng, t=None, lam=0.5):
        b = x1.shape[0]
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape
        t_batched = jnp.stack([jnp.array(t)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(rng, x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            rng, n0 = jax.random.split(rng)
            img = self.p_sample(img, n0, jnp.full((b,), i, dtype=jnp.int32))

        return img

    def q_sample(self, rng, x_start, t, noise=None):
        noise = default(noise, lambda: jax.random.normal(rng, x_start.shape))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, rng, mean_logvar, t, noise=None):
        x_start  = mean_logvar[..., :self.channels]
        x_logvar = mean_logvar[..., self.channels:]
        n0, n1 = jax.random.split(rng)
        noise = default(noise, lambda: jax.random.normal(n0, x_start.shape))
        x = self.q_sample(n1, x_start=x_start, t=t, noise=noise)
        model_out = self.model(x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if self.loss_type == 'l1':
            loss = jnp.abs(model_out - target)
        else:
            assert self.loss_type == 'l2'
            loss = jnp.square(model_out - target)

        if False:  # scale loss to emphasize low variance latent variables
            var = jnp.exp(x_logvar)
            loss = loss / jnp.maximum(var / jnp.mean(var, axis=(1,2,3,4), keepdims=True), 0.001)

        return jnp.mean(loss)

    def __call__(self, rng, img, *args, **kwargs):
        b, x, y, z, _ = img.shape
        n0, n1 = jax.random.split(rng)
        assert x == self.image_size and y == self.image_size and z == self.image_size, f'YXZ of image must be {self.image_size}'
        # antithetic sampling: https://github.com/ermongroup/ddim/blob/34d640e5180cc5ab378f84af6ed596cb0c810e6c/runners/diffusion.py#L147
        t = jax.random.randint(n0, (b // 2 + 1,), 0, self.num_timesteps)
        t = jnp.concatenate([t, self.num_timesteps - t])[:b]
        return self.p_losses(n1, img, t, *args, **kwargs)


def plot_latent(latent, vae_local, filename):
    latent_preshape = latent.shape
    # print("latent_preshape", latent_preshape)
    latent_size = int(round(((latent.shape[-1] // vae_local.z_channels) ** (1.0 / 3.0))))
    latent = jnp.reshape(latent, (latent.shape[0] ** 3, latent_size, latent_size, latent_size, vae_local.z_channels))
    BATCH_SIZE = 64
    assert latent.shape[0] % BATCH_SIZE == 0
    latent_splits = latent.shape[0] // BATCH_SIZE
    batch = jnp.split(latent, latent_splits, axis=0)
    print("Decoding in to voxels...")
    voxels = [vae_local.decoder(split) for split in batch]
    voxels = jnp.concatenate(voxels, axis=0)

    print("Saving to file...")
    voxels = jnp.reshape(voxels, (*latent_preshape[0:3], *voxels.shape[1:]))
    voxels = utils.undo_view_as_blocks(voxels)
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "wb") as fp:
        out_data = np.array(voxels[:, :, :, 0]) > 0.5
        binvox_rw.Voxels(out_data, (out_data.shape[0], out_data.shape[1], out_data.shape[2]), (0, 0, 0), 1.0, 'xyz').write(fp)


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_batch_size,
            train_num_steps,
            train_lr,
            adam_betas=(0.9, 0.999),
            save_and_sample_every=5000,
            eval_batchsize=1,
            warmup_steps = 5000,
            workdir=None,
            h5path=None,
            verbose=True,
            vae_local=None
    ):
        super().__init__()

        # TODO original keeps track of a exponential moving average verson of the model. Try it.
        self.model = diffusion_model
        self.eval_batchsize = eval_batchsize
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.workdir = workdir
        self.vae_local = vae_local
        dataset = tfio.IODataset.from_hdf5(h5path + "/latents_meanvar.hdf5", "/latents").repeat().shuffle(10000)
        dataset = dataset.batch(train_batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = map(utils.prepare_tf_data, dataset)
        dataset = jax_utils.prefetch_to_device(dataset, 2)
        self.dataset = dataset

        self.ref_batch = next(self.dataset)
        #latents_std = np.std(self.ref_batch)
        #rst = np.reshape(eval_latents, (-1, 256))  # TODO maybe per -1 dimension?
        #print("Testdata mean:", np.mean(rst, axis=0), "std:", np.std(rst, axis=0))
        mean, logvar = np.split(self.ref_batch, 2, -1)
        var = np.exp(logvar)
        print("Eval means - mean:", np.mean(mean), "std:", np.std(mean))
        print("Eval var   - mean:", np.mean(var), "std:", np.std(var), "min:", np.min(var), "max:", np.max(var))
        assert self.ref_batch.shape[-1] == 2 * self.model.channels

        rng = jax.random.PRNGKey(0)
        init_data = jnp.ones((1, self.image_size, self.image_size, self.image_size, 2 * diffusion_model.channels))  # 2* bc of mean&logvar
        self.ema = variables = diffusion_model.init(rng, rng, init_data)['params']

        opt_chain = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.linear_schedule(1e-9, train_lr, warmup_steps), b1=adam_betas[0], b2=adam_betas[1]))

        self.state = train_state.TrainState.create(
            apply_fn=diffusion_model.apply,
            params=variables,
            tx=opt_chain
        )
        self.state = utils.restore_checkpoint(self.state, workdir)
        self.state = jax_utils.replicate(self.state)
        self.ema = utils.restore_checkpoint(self.ema, workdir + "_ema")
        #self.ema = jax_utils.replicate(self.ema)
        if verbose:
            print(diffusion_model.model.tabulate(jax.random.PRNGKey(0), jnp.ones((1, self.image_size, self.image_size, self.image_size, diffusion_model.channels)), jnp.ones((1,))))

    def train(self):
        @jax.jit
        def train_step(state, batch, rng):
            def loss_fn(params):
                return self.model.apply({'params': params}, rng, batch)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            loss = jax.lax.pmean(loss, axis_name='batch')
            return state.apply_gradients(grads=grads), loss

        train_step_p = jax.pmap(train_step, axis_name="batch")
        writer = SummaryWriter(self.workdir)

        eval_latents = self.ref_batch[0, 0, :, :, :, :self.model.channels]  # (Device, Batch, x, y, z, c)
        print("Plotting latent_to_voxels.binvox...")
        plot_latent(eval_latents, self.vae_local, self.workdir + f"/latent_to_voxels.binvox")

        rng = jax.random.PRNGKey(0)
        rng_eval = jax.random.PRNGKey(1337)
        step = int(self.state.step[0])
        SINGLE_DATA = False
        if SINGLE_DATA:
            latents = self.ref_batch
            print("SINGLE BATCH!!!!! REMOVE THIS BEFORE FLIGHT")

        l = 0
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                #if l == 5:
                #    jax.profiler.start_trace(self.workdir)
                start_time = time.time()
                if not SINGLE_DATA:
                    latents = next(self.dataset)
                rng, key = jax.random.split(rng)
                key = jnp.stack(jax.random.split(key))
                prep_time = time.time()
                self.state, loss = train_step_p(self.state, latents, key)
                loss_mean = np.mean(loss)

                if step != 0 and step % 100 == 0:
                    # follow https://arxiv.org/pdf/2207.04316.pdf with 0.99 every 100 steps
                    e_decay = 0.99
                    self.ema = jax.tree_map(lambda e, p: e * e_decay + (1 - e_decay) * p, self.ema, jax_utils.unreplicate(self.state).params)

                #if l == 5:
                #    loss.block_until_ready()
                #    jax.profiler.stop_trace()

                train_time = time.time()
                pbar.set_description(f'loss: {loss_mean:.3f}, train perc:{100.0 * (train_time - prep_time) / (train_time - start_time) : .1f}%')
                writer.add_scalar("diff/loss", loss_mean, step)
                #writer.add_scalar("grad_mean", np.mean(grads), step)

                if step != 0 and step % self.save_and_sample_every == 0:
                    utils.save_checkpoint(self.state, self.workdir)
                    checkpoints.save_checkpoint(self.workdir + "_ema", self.ema, step, keep=1)

                    if self.vae_local is not None:
                        batches = num_to_groups(self.eval_batchsize, self.batch_size)  # TODO this eval stuff seams to leak memory
                        model_bound = self.model.bind({'params': jax_utils.unreplicate(self.state).params})
                        all_latents_list = list(map(lambda n: model_bound.sample(rng_eval, batch_size=n), batches))
                        del model_bound
                        all_latents = jnp.concatenate(all_latents_list, axis=0)
                        vox_path = self.workdir + f"/{step}.binvox"
                        plot_latent(all_latents[0], self.vae_local, vox_path)
                        print(f"Saved to {vox_path}")
                        writer.add_scalar("diff/diff_to_ref", np.mean(np.square(eval_latents - all_latents[0])), step)
                step += 1
                l += 1
                pbar.update(1)
