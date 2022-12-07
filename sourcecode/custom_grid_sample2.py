# modified from https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
import torch
import torch.nn.functional as F
import numpy as np
'''
def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):
    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
    return out_fmap
'''

def gather_nd_torch(params, indices, batch_dim=1):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.

    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.

    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x, m]
      batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.

    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x, c].

    Example:
    >>> batch_size = 5
    >>> inputs = torch.randn(batch_size, batch_size, batch_size, 4, 4, 4, 32)
    >>> pos = torch.randint(4, (batch_size, batch_size, batch_size, 12, 3))
    >>> gathered = gather_nd_torch(inputs, pos, batch_dim=3)
    >>> gathered.shape
    torch.Size([5, 5, 5, 12, 32])

    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = tf.gather_nd(inputs_tf, pos_tf, batch_dims=3)
    >>> gathered_tf.shape
    TensorShape([5, 5, 5, 12, 32])

    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)
    # indices = indices.reshape(batch_size, -1, n_pos)
    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered


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
    batch_size,height,width = (x).shape[:3]
    nchan = img.shape[1]
    batch_idx = torch.arange(batch_size).to(device)
    batch_idx = batch_idx.reshape((batch_size, 1, 1))
    #TODO: tile in pytorch
    b = torch.tile(batch_idx, (1, height, width))

    # indices = torch.stack([b, y, x], 3)
    indices = torch.stack([y, x], 3)
    # indices = indices.permute((0,3,1,2))
    # out = torch.gather(img, indices)
    '''
    out = gather_nd_torch(img.permute(0,2,3,1), indices, batch_dim=1)
    '''
    out = gather_nd_torch(img.permute(0,2,3,1), indices.flatten(start_dim=1,end_dim=2), batch_dim=1)
    out = out.reshape(batch_size,nchan,height,width)
    assert out.shape == (batch_size,nchan,height,width)
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
    '''
    torch.cuda.empty_cache()
    import gc;gc.collect()
    '''
    device = img.device
    H,W = img.shape[-2:]
    max_y = int(H - 1)
    max_x = int(W - 1)
    # zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    # x = float(x)
    # y = float(y)
    # print('check if this needs to be max_x - 1');import pdb;pdb.set_trace()
    x_pre = x
    y_pre = y
    
    x = 0.5 * ((x + 1.0) * float(max_x))
    y = 0.5 * ((y + 1.0) * float(max_y))

    D = min(H,W)
    span = (D//8)
    new_x = x.long()
    new_y = y.long()
        
    x = x + (span//2)
    y = y + (span//2)
    new_x = torch.clamp(new_x, None, max_x) + (span//2)
    new_y = torch.clamp(new_y, None, max_y) + (span//2)

    # del_x = x - new_x 
    # del_y = y - new_y 
    
    X,Y = torch.meshgrid(torch.arange(-(span//2),1+(span//2),device=device),
                         torch.arange(-(span//2),1+(span//2),device=device),
                         indexing='xy')
    assert X.ndim == 2
    assert Y.ndim == 2
    Y = Y[None,None,None,None,...]
    X = X[None,None,None,None,...]
    
    padded_img = torch.nn.functional.pad(img,((span//2),(span//2),(span//2),(span//2)),
                                          #  mode='constant',value=1)
                                          mode='reflect')
    at_Y = new_y[...,None,None] + Y
    at_X = new_x[...,None,None] + X
    assert (at_Y >= 0).all()
    assert (at_X >= 0).all()
    windows = padded_img[:,:,
                         at_Y,
                         at_X]
    # import pdb;pdb.set_trace()
    def gaussian_window(Y,X,cy,cx,sigma):
        w = torch.exp( - ((Y - cy)/sigma)**2 - ((X - cx)/sigma)**2 )
        # import pdb;pdb.set_trace()
        w = w/w.sum(dim=(-1,-2),keepdim=True)
        return w
    # grid_for_gauss = X.unsuqe
    # gaussian_window(Y,X,cy,cx,sigma)
    # X = X[0,0]
    # Y = Y[0,0]
    '''
    X = X + new_x[...,None,None].detach()
    Y = Y + new_y[...,None,None].detach()
    '''
    weights = gaussian_window(at_Y,at_X,y[...,None,None],x[...,None,None],span/3)
    out = combined = (weights * windows).sum(dim=(-1,-2))
    assert out.shape[-2:] == x.shape[-2:]
    '''
    to_fold = windows/weights
    # TODO:fold


    folded = F.fold(to_fold, output_size=x.shape[-2:], kernel_size=span, stride=1)
    out = F.fold(torch.ones_like(to_fold), output_size=x.shape[-2:], kernel_size=span, stride=1)
    print(f'folded shape = {folded.shape}')

    print(f'out shape = {out.shape}')
    '''
    # import pdb;pdb.set_trace()
    return out


