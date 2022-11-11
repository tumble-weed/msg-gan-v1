import torch

'''
def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):
    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
    return out_fmap
'''

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, C, H, W)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)

    Returns
    -------
    - output: tensor of shape (B, C, H, W)
    """
    device = x.device
    batch_size,nchan,height,width = (x).shape

    batch_idx = torch.arange(batch_size).to(device)
    batch_idx = batch_idx.reshape((batch_size, 1, 1))
    #TODO: tile in pytorch
    b = torch.tile(batch_idx, (1, height, width))

    indices = torch.stack([b, y, x], 3)
    indices = indices.permute((0,3,1,2))
    out = torch.gather(img, indices)
    assert out.shape == batch_size,nchan,height,width
    return out 

'''
def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.

    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.

    - width: desired width of grid/output. Used
      to downsample or upsample.

    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.

    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.

    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids
'''

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, C,H, W) layout.
    # - grid: x, y which is the output of affine_grid_generator.
    grid: N,H,W,C is the flow

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H,W = img.shape[-2:]
    max_y = int(H - 1)
    max_x = int(W - 1)
    # zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    # x = float(x)
    # y = float(y)
    x = 0.5 * ((x + 1.0) * float(max_x-1))
    y = 0.5 * ((y + 1.0) * float(max_y-1))

    # grab 4 nearest corner points for each (x_i, y_i)
    #x0 = tf.cast(tf.floor(x), 'int32')
    x0 = x.floor().long()
    x1 = x0 + 1
    # y0 = tf.cast(tf.floor(y), 'int32')
    y0 = y.floor().long()
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = torch.clamp(x0, None, max_x)
    x1 = torch.clamp(x1, None, max_x)
    y0 = torch.clamp(y0, None, max_y)
    y1 = torch.clamp(y1, None, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    wa = wa.unsqueeze(1)
    wb = wb.unsqueeze(1)
    wc = wc.unsqueeze(1)
    wd = wd.unsqueeze(1)

    # compute output
    out = wa*Ia + wb*Ib + wc*Ib + wd*Id
    return out
