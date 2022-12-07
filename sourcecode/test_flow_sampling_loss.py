#%%
from flow_utils import get_flow_sampling
import torch
from matplotlib import pyplot as plt
import numpy as np
tensor_to_numpy = lambda t:t.detach().cpu().numpy()

#%%
device = 'cuda'
flow_shape = (1,1)
x = (0.0*torch.randn(1,2,*flow_shape)).to(device).requires_grad_(True)
# x = (-1000*torch.ones(1,2,*flow_shape)).to(device).requires_grad_(True)
# x = (0.1*torch.stack(
#     torch.meshgrid(
#         torch.arange(flow_shape[0],device=device) - (flow_shape[0]//2),
#         torch.arange(flow_shape[1],device=device)- (flow_shape[1]//2),
#         indexing = 'xy'),
#     dim=0).unsqueeze(0).float()).requires_grad_(True)

flow = torch.tanh(x)
# x.data.copy_(torch.clamp(x,-1,1))
# flow = x
dummy_img = torch.randn(1,3,101,101).to(device)
optim_x = torch.optim.Adam([x],lr=1e-2)
res_patch_size = 33
from collections import defaultdict
trends = defaultdict(list)
#%%
# optim_x.param_groups[0]['lr'] = 1e-2
for i in range(100):
    flow = torch.tanh(x)
    # flow =x
    flow_sampling,detached_flow  = get_flow_sampling(flow,dummy_img,res_patch_size,retain_graph = True
                # ,stride=1
                )
    flow_sampling.retain_grad()
    # print(flow_sampling.abs().sum())
    # print((flow_sampling!=0).sum())
    assert (flow_sampling >=0).all()
    M = max(1,flow_sampling.max().detach())
    # M = 1'
    # '''
    sampling_norm = ((flow_sampling/1)**2).sum()
    # '''
    '''
    sampling_norm = ((flow_sampling[...,1:,:]-flow_sampling[...,:-1,:])**2).sum() + ((flow_sampling[...,:,1:]-flow_sampling[...,:,:-1])**2).sum()
    '''
    (1e0*sampling_norm).backward()    
    trends['sampling_norm'].append(sampling_norm.item())
    trends['max_flow_sampling'].append(flow_sampling.max().item())
    flow_std = flow.std(dim=(-1,-2)).mean(1)
    trends['flow_std'].append(flow_std.item())
    def add_flow_norm_grad(g,
        # added_g=added_g
        detached_flow = detached_flow
        ):
        # g = g + added_g#[...,::2,::2]
        # import pdb;pdb.set_trace()
        # g = g + th.clamp(detached_flow.grad,-1.,1.)
        g2 = (detached_flow.grad/M)
        trends['flow_grad'].append(g2.norm().item())
        g = g + g2
        return g
    flow.register_hook(add_flow_norm_grad)    
    (0*flow.sum()).backward()
    optim_x.step()
    # x.data.copy_(torch.clamp(x,-1.,1.))
    if (i% 10) == 0:
        if True:
            plt.figure()
            plt.imshow(tensor_to_numpy((flow_sampling[0,1]).float()),vmin=0)
            plt.colorbar()
            plt.show()
        if False:
            plt.figure()
            plt.imshow(tensor_to_numpy(flow[0,0]),vmin=-1)
            plt.show()
        if True:
            plt.figure()
            plt.imshow(tensor_to_numpy((flow_sampling).grad[0,1]),vmin=0)
            plt.colorbar()
            plt.title('flow_sampling grad')
            plt.show()
            
            
        if True:
            plt.figure()
            plt.plot(trends['sampling_norm'])
            plt.show()
        if False:
            plt.figure()
            plt.plot(trends['max_flow_sampling'])
            plt.show()        
        if False:
            plt.figure()
            plt.plot(trends['flow_grad'])
            plt.show()                
        if False:
            plt.figure()
            plt.plot(trends['flow_std'])
            plt.show()                        
        if False:
            step = 1
            plt.quiver(np.arange(0, flow.shape[-1], step), np.arange(flow.shape[-2], 0, -step), 
                    tensor_to_numpy(flow)[0,0,::step, ::step], tensor_to_numpy(flow)[0,1,::step, ::step])
        
    # %%