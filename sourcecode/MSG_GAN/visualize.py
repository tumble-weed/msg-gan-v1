from matplotlib import pyplot as plt
import os
import numpy as np
import torch as th
import cv2
import numpy as np
from flow_utils import flow_to_rgb
import torch.nn.functional as F
tensor_to_numpy = lambda t:t.detach().cpu().numpy()

def get_res_filenames(sample_dir,reses,prefix,epoch,i):
    img_files = [os.path.join(sample_dir, res, f"{prefix}_" +
                                    str(epoch) + "_" +
                                    str(i) + ".png")
                        for res in reses]

    for img_file in img_files:
        os.makedirs(os.path.dirname(img_file), exist_ok=True)                        
    return img_files
def visualize(msg_gan,epoch,i,
              sample_dir,fixed_input,real_images,
              purge=True):
    if purge and (epoch == 1) and (i == 1):
        print(f'purging {sample_dir}')
        os.system(f'rm -rf {sample_dir}')
    # create a grid of samples and save it
    reses = [str(int(np.power(2, dep))) + "_x_"
                + str(int(np.power(2, dep)))
                for dep in range(2, msg_gan.depth + 2)[msg_gan.min_scale:]]

    gen_img_files = get_res_filenames(sample_dir,reses,'gen',epoch,i)
    flow_img_files = get_res_filenames(sample_dir,reses,'flow',epoch,i)
    # Make sure all the required directories exist
    # otherwise make them
    os.makedirs(sample_dir, exist_ok=True)

    if (i == 1) and (epoch == 1):
        real_img_files = get_res_filenames(sample_dir,reses,'real',epoch,i)
        # assert real_images[-1].shape[-2:] == (256,256)
        msg_gan.create_grid(real_images, real_img_files)

    with th.no_grad():
        flow = msg_gan.gen(fixed_input)
        assert flow[-1].shape[-2:] == real_images[-1].shape[-2:]
        #=================================================
        if False and 'hacking flow to yield original image back':
            device = fixed_input.device
            print('hacking flow to yield original image back')
            new_flow = []
            assert len(flow) == len(msg_gan.patch_sizes)
            for k,(f,ps) in enumerate(zip(flow,msg_gan.patch_sizes)):
                H,W = f.shape[-2:]
                step_y = (1. - (-1.))/(H - 1)
                # half_step_y = step_y/2.
                step_x = (1. - (-1.))/(W- 1)
                # half_step_x = step_x/2.
                assert (ps % 2 == 0), 'this expression only valid for even patch sizes'
                # X,Y = th.meshgrid(
                #     th.linspace(-step_x*(-1 + W//2) - 0.5*step_x ,step_x*(-1 + W//2) + 0.5*step_x,W),
                #     th.linspace(-step_y*(-1 + H//2) - 0.5*step_y,step_y*(-1 + H//2) + 0.5*step_y,H),
                #     indexing = 'xy'
                # )
                X,Y = th.meshgrid(
                    th.linspace(-1.,1.,W),
                    th.linspace(-1.,1.,H),
                    indexing = 'xy'
                )
                new_f = th.stack([X,Y],dim=-1)
                new_f = new_f.permute(2,0,1)[None,...]
                new_f = new_f.to(device)
                assert new_f.shape == flow[k].shape
                new_flow.append(new_f)
                # import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()
            flow = new_flow
        #=================================================
        # print('in visualize setting all ps to 2')
        # PS = 2
        # import pdb;pdb.set_trace()
        # fake_sample_256 = flow_to_rgb(flow[-1],PS,msg_gan.stride,F.interpolate(msg_gan.ref,flow[-1].shape[-2:],align_corners=True))
        #=================================================
        fake_samples = [flow_to_rgb(f,ps,msg_gan.stride,F.interpolate(msg_gan.ref,f.shape[-2:],mode='bilinear',align_corners=True),(1,3)+f.shape[-2:]) for f,ps in zip(flow,msg_gan.patch_sizes)]        
        assert fake_samples[-1].shape == real_images[-1].shape
        msg_gan.create_grid(fake_samples, gen_img_files)
        #=================================================
        '''
        flow = [visualize_optical_flow(tensor_to_numpy(f.permute(0,2,3,1))[0]) for f in flow]
        # will have to remap to tensor for create grid to work
        flow = [th.tensor(f).permute(2,0,1)[None,...] for f in flow]
        
        msg_gan.create_grid(flow, flow_img_files)
        '''

        #=================================================
        # plot losses
        # import pdb;pdb.set_trace()
        for lname in msg_gan.trends:
            
            plt.figure()
            plt.plot(msg_gan.trends[lname])
            plt.title(lname)
            plt.draw()
            plt.savefig(os.path.join(sample_dir,lname+'.png'))
            plt.close()        
    #=================================================
    sampling = [get_flow_sampling(f,img,patch_size) for (f,img,patch_size) in zip(flow,real_images,msg_gan.patch_sizes) ]
    if False:
        print('sampling grounding')
        sampling = [s/s.max() for s in sampling]
    # flow = [visualize_optical_flow(tensor_to_numpy(f.permute(0,2,3,1))[0]) for f in flow]
    msg_gan.create_grid(sampling, flow_img_files)        
    #=================================================        
    # if epoch != 1:
    #     import pdb;pdb.set_trace()
def visualize_optical_flow(flow):
    # from https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
    # Use Hue, Saturation, Value colour model 
    
    hsv = np.zeros(flow.shape[:2]+(3,), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
    # cv2.imshow("colored flow", bgr)
    # return np.array()
    return rgb

def get_flow_sampling(flow,img,patch_size):
    import flow_utils
    sampling,detached_flow = flow_utils.get_flow_sampling(flow,img,patch_size)
    '''
    (sampling*(sampling > 1).float()).sum().backward()
    #TODO: actually this is a double derivative. but according to this expression
    # we are taking the dd after flow
    # and d before it, is this correct?
    flow.backward(detached_flow.grad)
    '''
    return sampling
    
    