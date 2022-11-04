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
              sample_dir,fixed_input,real_images):
    # create a grid of samples and save it
    reses = [str(int(np.power(2, dep))) + "_x_"
                + str(int(np.power(2, dep)))
                for dep in range(2, msg_gan.depth + 2)]

    gen_img_files = get_res_filenames(sample_dir,reses,'gen',epoch,i)
    # Make sure all the required directories exist
    # otherwise make them
    os.makedirs(sample_dir, exist_ok=True)

    if (i == 1) and (epoch == 1):
        real_img_files = get_res_filenames(sample_dir,reses,'real',epoch,i)
        msg_gan.create_grid(real_images, real_img_files)

    with th.no_grad():
        flow = msg_gan.gen(fixed_input)
        #=================================================
        print('hacking flow to yield original image back')
        new_flow = []
        for k,f in enumerate(flow):
            H,W = f.shape[-2:]
            Y,X = th.meshgrid(
                th.linspace(-1,1,H),
                th.linspace(-1,1,W)
            )
            f = th.stack([X,Y],dim=-1)
            f = f.permute(2,0,1)[None,...]
            assert f.shape == flow[k].shape
            new_flow.append(f)
        flow = new_flow
        #=================================================
        fake_samples = [flow_to_rgb(f,msg_gan.patch_size,msg_gan.stride,F.interpolate(msg_gan.ref,f.shape[-2:])) for f in flow]
        msg_gan.create_grid(fake_samples, gen_img_files)
        flow = [visualize_optical_flow(tensor_to_numpy(f.permute(0,2,3,1))[0]) for f in flow]
        # will have to remap to tensor for create grid to work
        flow = [th.tensor(f).permute(2,0,1)[None,...] for f in flow]
        msg_gan.create_grid(flow, gen_img_files)
    

def visualize_optical_flow(flow):
    # from https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
    # Use Hue, Saturation, Value colour model 
    
    hsv = np.zeros(flow.shape, dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
    # cv2.imshow("colored flow", bgr)
    # return np.array()
    return rgb
