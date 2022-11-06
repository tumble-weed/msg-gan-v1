import torch
from torch.nn.functional import fold, unfold

def combine_patches(O, patch_size, stride, img_shape):
    
    assert img_shape.__len__() == 4
    assert len(patch_size) == 2
    assert len(O.shape) == 6
    assert O.shape[-2:] == patch_size
    O = O.permute((0,2,3,1,4,5))
    O = O.contiguous()
    O = O.view(-1,*O.shape[-3:])
    assert O.shape[-2:] == patch_size
    device = O.device
    channels = 3
    O = O.permute(1, 0, 2, 3).unsqueeze(0) # chan,batch_size,patch_size,patch_size
    patches = O.contiguous().view(O.shape[0], O.shape[1], O.shape[2], -1) \
        .permute(0, 1, 3, 2) \
        .contiguous().view(1, channels * patch_size[0] * patch_size[0], -1)
    
    # batch_size,chan,Ypatch_size,-1
    # batch_size,chan,Xpatch_size,Ypatch_size
    #  -> 1, channels*Xpatch_size*Ypatch_size, H*W
    # chan,batch_size,patch_size,patch_size
    combined = fold(patches, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    # normal fold matrix
    input_ones = torch.ones((1, img_shape[1], img_shape[-2], img_shape[-1]), dtype=O.dtype, device=device)
    divisor = unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = fold(divisor, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    combined =  (combined / divisor).squeeze(dim=0).permute(1, 2, 0)
    # convert from hwc to bchw format
    combined = combined.permute(2,0,1)[None,...]
    return combined
"""
# deprecated 
def sample_using_flow(flow,x,patch_size):
    #==========================================================
    # Dummy Flow for testing
    Ymargin = (patch_size//2) * 1./ x.shape[2]
    Xmargin = (patch_size//2) * 1./ x.shape[3]
    device = x.device
    dummy_flow = torch.meshgrid(torch.linspace(-1,1,patch_size[0]),torch.linspace(-1,1,patch_size[1])).to(device)
    dummy_flow[:,0,:,:] = (dummy_flow[:,0,:,:] * Ymargin)
    dummy_flow[:,1,:,:] = (dummy_flow[:,1,:,:] * Xmargin)
    flow = dummy_flow; print('setting flow to dummy flow, should recreate the original image')
    patch_size = (1,1);print('setting patch_size to (1,1)')
    #==========================================================
    device = x.device
    batch_size = flow.shape[0]
    n_patches = flow.shape[2]*flow.shape[3]
    
    Y,X = torch.meshgrid(
        torch.linspace(-1,1,flow.shape[2],device=device),
        torch.linspace(-1,1,flow.shape[3],device=device))
    YX = torch.stack([Y,X],dim=-1)
    '''
    theta = torch.zeros(batch_size*n_patches,2,3).to(device)
    # YX.shape == [H,W,2]
    theta[:,0,0] = 1
    theta[:,1,1] = 1
    theta[:,0,2] = flow[:,0,:,:].reshape(batch_size*n_patches)
    theta[:,1,2] = flow[:,1,:,:].reshape(batch_size*n_patches)
    print('see if theta is correctly shaped'); import pdb;pdb.set_trace()
    # theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(theta, x.size()) # N,H,W,2
    '''
    grid = torch.permute(flow,[0,2,3,1])
    H,W = grid.shape[[1,2]]
    patch_grid = grid[:,:,:,None,None,:] + YX[None,None,None,...]
    patch_grid = torch.permute(patch_grid,[0,1,3,2,4,5])
    patch_grid = patch_grid.reshape(batch_size,H*patch_size,W*patch_size,2)
    
    patches = F.grid_sample(x, patch_grid)
    print('see shape of patches');import pdb;pdb.set_trace()
    patches = patches.view(batch_size,H,patch_size,W,patch_size,-1)
    patches = torch.transpose(patches,(0,1,3,2,4,5))
    return patches
"""
def patch_sample(flow,img,patch_size):
#             fake_flow.shape = N,2,H,W
    device = flow.device
    # import pdb;pdb.set_trace()
    img_xy = torch.meshgrid(
        torch.arange(img.shape[3],device=device).float(),
        torch.arange(img.shape[2],device=device).float(),
        indexing = 'xy'
    )
    img_x,img_y = img_xy
    img = torch.stack([img_y,img_x],dim=0)
    
    img = torch.cat(
        [img,
        torch.zeros_like(img[:1]),],
        dim =0
    )[None,...]
    print('setting img to arange, to see how far are patches being pulled from')
    
    N,_,fH,fW = flow.shape
    H,W = img.shape[-2:]
#             flow = torch.permute(flow,(0,2,3,1))
    # assert img.shape[-2] == img.shape[-1]
    step_y = (1 - (-1))/(H -1) 
    step_x = (1 - (-1))/(W - 1)
    new_flow = torch.zeros(N,fH,patch_size,fW,patch_size,2).to(device)
    '''
    if patch_size % 2 == 1:
        # odd
        mesh = torch.meshgrid(
            torch.linspace(-step_y*(patch_size//2),step_y*(patch_size//2),patch_size),
            torch.linspace(-step_x*(patch_size//2),step_x*(patch_size//2),patch_size),
        )
    else:
        # even
        mesh = torch.meshgrid(
            torch.linspace(-step_y*(patch_size//2 - 0.5),step_y*(patch_size//2 + 0.5),patch_size),
            torch.linspace(-step_x*(patch_size//2 - 0.5),step_x*(patch_size//2 + 0.5),patch_size),
        )        
    '''
    if 'even with 0.5' and False:
        pass
        # is_even = float((patch_size % 2) == 0)
        # assert False,'meshgrid is wrong'
        # mesh = torch.meshgrid(
        #         torch.linspace(-step_y*(patch_size//2 - 0.5*is_even),step_y*(patch_size//2 - 0.5*is_even),patch_size),
        #         torch.linspace(-step_x*(patch_size//2 - 0.5*is_even),step_x*(patch_size//2 - 0.5*is_even),patch_size),
        #     )
    elif 'same for even and odd' and True:
        mesh_xy = torch.meshgrid(
                torch.linspace(-step_x*(-1 + patch_size//2) - 0.5*step_x ,step_x*(-1 + patch_size//2) + 0.5*step_x,patch_size),
                torch.linspace(-step_y*(-1 + patch_size//2) - 0.5*step_y,step_y*(-1 + patch_size//2) + 0.5*step_y,patch_size),
                indexing = 'xy'
            )        
        mesh_x,mesh_y = mesh_xy
    # import pdb;pdb.set_trace()
    # flow is (x,y) not (y,x) https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    mesh = torch.stack([mesh_x,mesh_y],dim=-1)
    mesh = mesh.to(device)
    flow = flow.permute(0,2,3,1)

    new_flow[...,0] = torch.zeros_like(flow[:,:,None,:,None,0]) + mesh[None,None,:,None,:,0]
    new_flow[...,1] = torch.zeros_like(flow[:,:,None,:,None,1]) + mesh[None,None,:,None,:,1]

#     new_flow = new_flow.permute((0,2,3,1,4,5))
#     new_flow = torch.flatten(new_flow,start_dim=0,end_dim=2)

    new_flow = torch.reshape(new_flow,(N,(fH)*patch_size,(fW)*patch_size,2))
    patches = torch.nn.functional.grid_sample(img,new_flow)
    patches = patches.reshape(N,3,(fH),patch_size,(fW),patch_size)

    patches = patches.permute(0,1,2,4,3,5)
    # if img.shape[-1] >= 256:
    #     import pdb;pdb.set_trace()
    return patches
def flow_to_rgb(flow,patch_size,stride,img):
    # patch_size = 1;print('setting patch size to 1')
    if 'heterogenous for even and odd' and True:
        if patch_size > 1:
            if patch_size % 2 == 1:
                flow = flow[:,:,patch_size//2:-(patch_size//2),patch_size//2:-(patch_size//2)]
            else:
                flow = flow[:,:,(patch_size//2):-(patch_size//2)+1,patch_size//2:-(patch_size//2) + 1]

    elif 'same for even and odd' and False:
        flow = flow[:,:,patch_size//2:-(patch_size//2),patch_size//2:-(patch_size//2)]
    fake_patches = patch_sample(flow,img[:1],patch_size = patch_size)
    # img_shape = real_cpu.shape
    img_shape = img.shape
    fake = combine_patches(fake_patches, (patch_size,patch_size), stride, img_shape)
    return fake
'''
# for making an image from fake_flow
fake_flow = netG(noise)[:,:,patch_size//2:-(patch_size//2),patch_size//2:-(patch_size//2)]
fake_patches = patch_sample(fake_flow,real_cpu[:1],patch_size = patch_size)
img_shape = real_cpu.shape
fake = combine_patches(fake_patches, (patch_size,patch_size), stride, (img_
'''