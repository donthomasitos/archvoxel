import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

# Reference: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
# TOD O remove bias parameters in layers before norms


def norm():
    return nn.GroupNorm(num_groups=32)


def activ(x):
    return nn.swish(x)


class Upsample(nn.Module):
    in_channels: int

    def setup(self):
        super().__init__()
        self.conv = nn.Conv(self.in_channels, kernel_size=(3, 3, 3), strides=(1, 1, 1))

    def __call__(self, x):
        xs = x.shape
        x = jax.image.resize(x, (xs[0], xs[1] * 2, xs[2] * 2, xs[3] * 2, xs[4]), method=jax.image.ResizeMethod.NEAREST)
        x = self.conv(x)
        return x


class AttnBlock(nn.Module):
    in_channels: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, x):
        h_ = x
        h_ = norm()(h_)
        preshape = x.shape
        assert len(preshape) == 1+3+1  # batch, xyz, data
        h_ = nn.SelfAttention(num_heads=self.num_heads)(jnp.reshape(h_, (x.shape[0], -1, x.shape[-1])))
        h_ = jnp.reshape(h_, preshape)
        h_ = nn.Conv(self.in_channels, (1, 1, 1), strides=(1, 1, 1))(h_)
        return x + h_


class ResnetBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        h = x
        h = norm()(h)
        h = activ(h)
        h = nn.Conv(self.out_channels, (3, 3, 3), strides=(1, 1, 1))(h)

        h = norm()(h)
        h = activ(h)
        h = nn.Conv(self.out_channels, (3, 3, 3), strides=(1, 1, 1))(h)

        if x.shape[-1] != h.shape[-1]:
            x = nn.Conv(self.out_channels, (1, 1, 1), strides=(1, 1, 1))(x)
        return x + h


class Encoder(nn.Module):
    z_channels: int
    ch: int
    num_res_blocks: int
    ch_mult: list
    resolution: int
    attn_resolutions: list

    @nn.compact
    def __call__(self, x):
        x = x - 0.5  # shift voxel data towards center

        in_ch_mult = (1,) + tuple(self.ch_mult)
        curr_res = self.resolution
        block_in = self.ch * self.ch_mult[-1]

        hs = [nn.Conv(self.ch, (3, 3, 3), strides=(1, 1, 1))(x)]
        for i_level in range(len(self.ch_mult)):
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                h = ResnetBlock(out_channels=block_out)(hs[-1])
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    h = AttnBlock(block_in)(h)
                hs.append(h)
            if i_level != len(self.ch_mult) - 1:
                hs.append(nn.Conv(block_in, (3, 3, 3), strides=(2, 2, 2))(hs[-1]))
                curr_res = curr_res // 2

        # middle
        h = hs[-1]
        h = ResnetBlock(out_channels=block_in)(h)
        h = AttnBlock(block_in)(h)
        h = ResnetBlock(out_channels=block_in)(h)

        # end
        h = norm()(h)
        h = activ(h)
        mean = nn.Conv(self.z_channels, (3, 3, 3), strides=(1, 1, 1))(h)
        logvar = nn.Conv(self.z_channels, (3, 3, 3), strides=(1, 1, 1))(h)
        return mean, logvar


class Decoder(nn.Module):
    z_channels: int
    ch: int
    num_res_blocks: int
    ch_mult: list
    resolution: int
    attn_resolutions: list
    out_ch = 1

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        num_resolutions = len(self.ch_mult)
        block_in = self.ch * self.ch_mult[num_resolutions - 1]
        curr_res = self.resolution // 2 ** (num_resolutions - 1)

        h = nn.Conv(block_in, kernel_size=(3, 3, 3), strides=(1, 1, 1))(z)

        # middle
        h = ResnetBlock(out_channels=block_in)(h)
        h = AttnBlock(block_in)(h)
        h = ResnetBlock(out_channels=block_in)(h)

        # upsampling
        for i_level in reversed(range(num_resolutions)):
            block_out = self.ch * self.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                h = ResnetBlock(out_channels=block_out)(h)

                block_in = block_out
                if curr_res in self.attn_resolutions:
                    h = AttnBlock(block_in)(h)
            if i_level != 0:
                h = Upsample(block_in)(h)
                curr_res = curr_res * 2

        # end
        h = norm()(h)
        h = activ(h)
        h = nn.Conv(self.out_ch, kernel_size=(3, 3, 3), strides=(1, 1, 1))(h)
        h = nn.sigmoid(h)
        return h


def reparameterize(rng, mean, logvar, is_training=True):
    if not is_training:
        return mean
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


class VAE(nn.Module):
    z_channels: int
    ch: int
    num_res_blocks: int
    ch_mult: list
    resolution: int
    attn_resolutions: list

    def setup(self):
        self.encoder = Encoder(z_channels=self.z_channels, ch=self.ch, num_res_blocks=self.num_res_blocks, ch_mult=self.ch_mult, resolution=self.resolution, attn_resolutions=self.attn_resolutions)
        self.decoder = Decoder(z_channels=self.z_channels, ch=self.ch, num_res_blocks=self.num_res_blocks, ch_mult=self.ch_mult, resolution=self.resolution, attn_resolutions=self.attn_resolutions)

    def __call__(self, x: jnp.ndarray, z_rng, is_training=True):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar, is_training)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar
