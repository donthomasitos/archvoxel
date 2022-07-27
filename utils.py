import math, jax, pickle, os, optax
from PIL import Image
import numpy as np
from flax.training import checkpoints
import matplotlib.pyplot as plt
import binvox_rw
import jax.numpy as jnp
from model import VAE
from flax.training import train_state


def save_image(ndarray, fp, nrow=4, padding=2, pad_value=0, format=None):
    """Make a grid of images and Save it into an image file.
  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
    fp - A filename(string) or file object
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the filename extension.
      If a file object was used instead of a filename, this parameter should always be used.
  """
    ndarray = np.array(ndarray) > 0.5

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]

    # center slices for faster drawing
    posx = ndarray.shape[1] // 2

    images = []
    for n in range(nmaps):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.voxels(ndarray[n, :, :, :, 0], edgecolor=None, facecolors="red")
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(data)
        plt.close(fig)

    ndarray = np.stack(images)

    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = np.full((height * ymaps + padding, width * xmaps + padding, num_channels), pad_value, dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width] = ndarray[k]
            k = k + 1

    im = Image.fromarray(np.array(grid))
    im.save(fp, format=format)
    print("Saved", fp)


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=2)


def clear_empty_slices(v):
    assert len(v.shape) == 3
    pre_items = np.prod(v.shape)
    v = np.delete(v, np.argwhere(np.all(v == 0, axis=(1, 2))), axis=0)
    v = np.delete(v, np.argwhere(np.all(v == 0, axis=(0, 2))), axis=1)
    v = np.delete(v, np.argwhere(np.all(v == 0, axis=(0, 1))), axis=2)
    num_reduced = pre_items - np.prod(v.shape)
    if num_reduced != 0:
        print(f"Reduced {num_reduced} items, {int(100.0 * num_reduced / pre_items)}% deleted")
    return v


def load_model_data(load_pickled, min_size=None):
    model_datas = []

    # TODO use h5py
    if load_pickled:  # Shortcut to make data loading faster
        print("Loading data...")
        with open('model_datas.p', 'rb') as handle:
            model_datas = pickle.load(handle)
    else:
        for filename in os.listdir("data"):
            with open(os.path.join("data", filename), 'rb') as f:
                binvox_read = binvox_rw.read_as_3d_array(f)
            new_data = clear_empty_slices(binvox_read.data.astype(np.float32))
            model_datas.append(new_data)
            print("Voxel input size:", model_datas[-1].shape)
        with open('model_datas.p', 'wb') as handle:
            pickle.dump(model_datas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if min_size is not None:
        print("Padding...")
        for im in range(len(model_datas)):
            pads = [max(0, min_size - model_datas[im].shape[d]) for d in range(3)]
            model_datas[im] = np.pad(model_datas[im], [(0, p) for p in pads])
    return model_datas


def get_random_part(rng, model_data, resolution, use_empty_probab=0.05):
    # pads = [(ensure_modulo - (model_datas[im].shape[d] % ensure_modulo)) % ensure_modulo for d in range(3)]
    # model_datas[im] = np.pad(model_datas[im], [(p, 0) for p in pads])
    # assert np.all(np.array(model_datas[im].shape) % ensure_modulo == 0)

    x = None
    no_empty_slice = rng.random() > use_empty_probab  # sometimes allow empty slices, model must learn that too.
    while x is None or (np.sum(x) == 0.0 and no_empty_slice):
        ix, iy, iz = rng.integers([0, 0, 0], [model_data.shape[0] - resolution + 1,  # +1 bc upper range is exclusive
                                              model_data.shape[1] - resolution + 1,
                                              model_data.shape[2] - resolution + 1])
        x = model_data[ix:ix + resolution, iy:iy + resolution, iz:iz + resolution]
    # TODO only rotations, not flips that destroy R/L
    if rng.random() > 0.5:
        x = np.flip(x, 0)
    # No up/down flip. Meaningless for architecture.
    if rng.random() > 0.5:
        x = np.flip(x, 2)

    if rng.random() > 0.5:
        x = np.transpose(x, [2, 1, 0])
    return x


def get_vae_state(resolution, vae_params, rng_init, rng_sample, workdir, lr, verbose=False):
    init_data = jnp.ones((1, resolution, resolution, resolution, 1), jnp.float32)
    variables = VAE(**vae_params).init(rng_init, init_data, rng_sample)

    opt_chain = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.linear_schedule(0.0, lr, 1000)))

    state = train_state.TrainState.create(
        apply_fn=VAE(**vae_params).apply,
        params=variables['params'],
        tx=opt_chain,
    )
    state = restore_checkpoint(state, workdir)
    if verbose:
        print(VAE(**vae_params).tabulate(jax.random.PRNGKey(0), init_data, jax.random.PRNGKey(0)))
    return state


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access
        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def undo_view_as_blocks(x):
    assert len(x.shape) == 7, "3 spatial, 3 block spatial, 1 data"
    new_dim = x.shape[0] * x.shape[3]
    '''
    # the transpose digits come from this hacky code... I *think* it's correct but terribly ugly.
    import itertools
    from skimage.util.shape import view_as_blocks
    a = np.arange(16*16*16)
    a = np.reshape(a, (16,16,16))
    a_block = view_as_blocks(a, (2,2,2))
    for p in list(itertools.permutations([0, 1, 2, 3,4,5])):
        if np.all(a == np.reshape(np.transpose(a_block, p), a.shape)):
            print (p)'''
    return np.reshape(np.transpose(x, [0, 3, 1, 4, 2, 5, 6]), (new_dim, new_dim, new_dim, x.shape[-1]))
