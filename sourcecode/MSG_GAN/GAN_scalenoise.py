""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""
TODO = None
import datetime
import os
import time
import timeit
import numpy as np
import torch as th
import torch.nn.functional as F
import os
from MSG_GAN.visualize import visualize
from flow_utils import flow_to_rgb
def multilevel_flow(levels,detach=False):
    out_levels = []
    base = th.zeros_like(levels[0])
    for l in levels:
        upscaled = th.nn.functional.interpolate(base,size= l.shape[-2:],mode='bilinear')
        if detach:
            upscaled = upscaled.detach()
        base = th.clamp(2*l + upscaled,-1.,1.)
        out_levels.append(base)
        
    return out_levels
        
        
        
    
class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, dilation=1, use_spectral_norm=True,min_scale = 0):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param dilation: amount of dilation to be used by the 3x3 convs
                         in the Generator module.
        :param use_spectral_norm: whether to use spectral normalization
        """
        from torch.nn import ModuleList, Conv2d
        from MSG_GAN.CustomLayers import GenGeneralConvBlock, GenInitialBlock

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.depth = depth
        self.latent_size = latent_size
        self.spectral_norm_mode = None
        self.dilation = dilation

        # register the modules required for the GAN Below ...
        # create the ToRGB layers for various outputs:
        
        def to_flow(in_channels):
            return Conv2d(in_channels, 2, (1, 1), bias=True)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([GenInitialBlock(self.latent_size)])
        self.flow_converters = ModuleList([to_flow(self.latent_size)])
        #TODO: rename all rgb to flow (flow_converters, to_rgb)
        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size,
                                            dilation=dilation)
                flow = to_flow(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    dilation=dilation
                )
                flow = to_flow(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.flow_converters.append(flow)

        # if spectral normalization is on:
        if use_spectral_norm:
            self.turn_on_spectral_norm()
        self.min_scale = min_scale
    def turn_on_spectral_norm(self):
        """
        private helper for turning on the spectral normalization
        :return: None (has side effect)
        """
        # print('not using spectral norm')
        # return 
        from torch.nn.utils import spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is False, \
                "can't apply spectral_norm. It is already applied"

        # apply the same to the remaining relevant blocks
        for module in self.layers:
            module.conv_1 = spectral_norm(module.conv_1)
            module.conv_2 = spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = True

    def turn_off_spectral_norm(self):
        """
        private helper for turning off the spectral normalization
        :return: None (has side effect)
        """
        print('not using spectral norm')
        return 
        from torch.nn.utils import remove_spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

        # remove the applied spectral norm
        for module in self.layers:
            remove_spectral_norm(module.conv_1)
            remove_spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = False

    def forward(self, xlist):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        from torch import tanh
        outputs = []  # initialize to empty list
        xlist = xlist[::-1]
        y = xlist[0]  # start the computational pipeline
        for j,(block, converter) in enumerate(zip(self.layers, self.flow_converters)):
            if j > 0:
                assert y.shape == xlist[j].shape
                y = y + 0.01*xlist[j]
            # import pdb;pdb.set_trace()
            y = block(y)

            flow = tanh(converter(y))
            # if  j == 0:
            #     import pdb;pdb.set_trace()            
            # import pdb;pdb.set_trace()
            self.mixup = 0.
            if self.mixup:
                identity = th.meshgrid(
                    
                    th.linspace(-1,1,flow.shape[-1],
                                device = flow.device
                        ),
                    th.linspace(-1,1,flow.shape[-2],
                                device = flow.device),
                    indexing = 'xy'
                    
                )
                identity = th.stack(identity,dim=0).unsqueeze(0)
                flow = self.mixup*flow + (1 -  self.mixup)*identity
            outputs.append(flow)
            
                
        # print('see effect of gan input shape');import pdb;pdb.set_trace()
        # return outputs[self.min_scale:]
        return multilevel_flow(outputs[self.min_scale:],detach=True)

'''
class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512, dilation=1, use_spectral_norm=True):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param dilation: amount of dilation to be applied to
                         the 3x3 convolutional blocks of the discriminator
        :param use_spectral_norm: whether to use spectral_normalization
        """
        from torch.nn import ModuleList
        from MSG_GAN.CustomLayers import DisGeneralConvBlock, DisFinalBlock
        from torch.nn import Conv2d

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.depth = depth
        self.feature_size = feature_size
        self.spectral_norm_mode = None
        self.dilation = dilation

        # create the fromRGB layers for various inputs:
        def from_rgb(out_channels):
            return Conv2d(3, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList([from_rgb(self.feature_size // 2)])

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([DisFinalBlock(self.feature_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    dilation=dilation
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            dilation=dilation)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 1] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # if spectral normalization is on:
        if use_spectral_norm:
            self.turn_on_spectral_norm()

    def turn_on_spectral_norm(self):
        """
        private helper for turning on the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is False, \
                "can't apply spectral_norm. It is already applied"

        # apply the same to the remaining relevant blocks
        for module in self.layers:
            module.conv_1 = spectral_norm(module.conv_1)
            module.conv_2 = spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = True

    def turn_off_spectral_norm(self):
        """
        private helper for turning off the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import remove_spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

        # remove the applied spectral norm
        for module in self.layers:
            remove_spectral_norm(module.conv_1)
            remove_spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = False

    def forward(self, inputs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 1](inputs[self.depth - 1])
        y = self.layers[self.depth - 1](y)
        for x, block, converter in \
                zip(reversed(inputs[:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = th.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        return y
'''
# from consingan
class ConvBlock(th.nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, batch_norm=False, generator=False):
        super(ConvBlock,self).__init__()
        self.add_module('conv', th.nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=1, padding=padd))
        if generator and batch_norm:
            self.add_module('norm', th.nn.BatchNorm2d(out_channel))
        # self.add_module(opt.activation, get_activation(opt))
        self.add_module('lrelu', th.nn.LeakyReLU(0.2, inplace=True))
class Discriminator(th.nn.Module):
    def __init__(self, latent_size=64,ker_size=3,padd_size=0,
                 num_layer=3,):
        super(Discriminator, self).__init__()

        # self.opt = opt
        N = int(latent_size)

        self.head = ConvBlock(3, N, ker_size, padd_size)

        self.body = th.nn.Sequential()
        for i in range(num_layer):
            block = ConvBlock(N, N, ker_size, padd_size)
            self.body.add_module('block%d'%(i),block)

        self.tail = th.nn.Conv2d(N, 1, kernel_size=ker_size, padding=padd_size)

    def forward(self,x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out

class MSG_GAN:
    """ Unconditional TeacherGAN

        args:
            depth: depth of the GAN (will be used for each generator and discriminator)
            latent_size: latent size of the manifold used by the GAN
            gen_dilation: amount of dilation for generator
            dis_dilation: amount of dilation for discriminator
            use_spectral_norm: whether to use spectral normalization to the convolutional
                               blocks.
            device: device to run the GAN on (GPU / CPU)
    """

    def __init__(self, depth=7, latent_size=512, gen_dilation=1,
                 dis_dilation=1, use_spectral_norm=True, device=th.device("cpu"),
                 patch_size = None,stride=None,ref=None):
        """ constructor for the class """
        from torch.nn import DataParallel
        self.min_scale = 3
        self.gen = Generator(depth, latent_size, dilation=gen_dilation,
                             use_spectral_norm=use_spectral_norm,min_scale=self.min_scale).to(device)
        self.dis_list = []
        for s in range(self.min_scale,depth):
            # dis = TODO
            dis = Discriminator(latent_size=64,ker_size=3,padd_size=0,
                 num_layer=3,).to(device)
            self.dis_list += [dis]

        # Create the Generator and the Discriminator
        '''
        if device == th.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)
        '''

        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.device = device

        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        for dis in self.dis_list:
            dis.eval()
        from collections import defaultdict
        self.trends = defaultdict(list)
        self.largest_patch_size = patch_size
        self.patch_sizes = list(reversed([1 + (self.largest_patch_size-1)//(2**i) for i in range(depth)]))[self.min_scale:]
        self.ref = ref
        self.stride = stride
    '''
    def generate_samples(self, num_samples):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated
        :return: generated samples tensor: list[ Tensor(B x H x W x C)]
        """
        noise = th.randn(num_samples, self.latent_size,*self.latent_spatial).to(self.device)
        generated_images = self.gen(noise)

        # reshape the generated images
        generated_images = list(map(lambda x: (x.detach().permute(0, 2, 3, 1) / 2) + 0.5,
                                    generated_images))

        return generated_images
    '''

    def optimize_discriminator(self, dis_optim, noise_list, real_batch, loss_fn):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_flow = self.gen(noise_list)
        
        fake_samples = [flow_to_rgb(flow,ps,self.stride,
        F.interpolate(self.ref,flow.shape[-2:],mode='bilinear',align_corners=True),(1,3)+flow.shape[-2:]) for flow,ps in zip(fake_flow,self.patch_sizes)]
        # print('early return from optimize_discriminator'); return 0
        fake_samples = list(map(lambda x: x.detach(), fake_samples))
        
        assert len(real_batch[self.min_scale:]) == len(fake_samples)
        loss = loss_fn.dis_loss(real_batch[self.min_scale:], fake_samples,trends=self.trends)
        # print('not stepping in dis')
        if True and 'no-grad':
            # optimize discriminator
            dis_optim.zero_grad()
            if True:
                loss.backward()
            dis_optim.step()
        return loss.item()

    def optimize_generator(self, gen_optim, noise_list, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """
        global epoch,batch,sample_dir1
        # import pdb;pdb.set_trace()
        # print('setting noise to 0');noise = th.zeros_like(noise)
        # generate a batch of samples
        # with th.no_grad():
        fake_flow = self.gen(noise_list)
        fake_samples = [flow_to_rgb(flow,ps,self.stride,F.interpolate(self.ref,flow.shape[-2:],mode='bilinear',align_corners=True),(1,3)+flow.shape[-2:]) for flow,ps in zip(fake_flow,self.patch_sizes)]
        
        assert len(real_batch[self.min_scale:]) == len(fake_samples)
        loss = loss_fn.gen_loss(real_batch[self.min_scale:], fake_samples,trends=self.trends)
        # optimize discriminator
        gen_optim.zero_grad()
        #=====================================================
        total_variance_loss = 0
        if False:
            # import pdb;pdb.set_trace()
            total_variance_loss = 0
            for j,f in enumerate(fake_flow):
                fmean = f.mean(dim=(-1,-2),keepdim=True)
                f = f - fmean
                '''
                cov = th.einsum('bcij,bdkl->bcd',f,f)
                L = th.linalg.eigvals(cov)
                # if j == 0:
                #     import pdb;pdb.set_trace()
                variance_loss = -1e-3*L.sum()
                total_variance_loss = total_variance_loss + variance_loss
                '''
                assert f.shape[1] == 2
                variance_loss = -1e-4 * f.std(dim=(0,-1,-2)).mean()
                total_variance_loss = total_variance_loss + variance_loss
                self.trends[f'sampling_norm_loss_{j}'].append(variance_loss.item())
        if True and 'sampling norm':
            from flow_utils import get_flow_sampling
            # flow diversity loss
            tv_loss = 0
            for j,(res_flow,res_img,res_patch_size) in enumerate(zip(fake_flow,real_batch[self.min_scale:],self.patch_sizes)):
                sampling_loss_factor = 1
                # if j < len(fake_flow) -1:
                #     sampling_loss_factor = 4
                #     continue
                flow_sampling,detached_flow = get_flow_sampling(res_flow,res_img,res_patch_size,retain_graph = True
                # ,stride=1
                )
                
                # added_g = detached_flow.grad
                # detached_flow = None
                # M = flow_sampling.max().detach()
                M = max(1,flow_sampling.max().detach())
                sampling_norm = th.log(((flow_sampling/1)**2).mean())
                def get_tv_loss(flow):
                    assert flow.ndim == 4
                    assert flow.shape[1] == 2
                    tv_horz = flow[...,:1] - flow[...,-1:]
                    tv_vert = flow[...,:1,:] - flow[...,-1:,:]
                    tv = (tv_horz**2).mean() + (tv_vert**2).mean()
                    return tv
                tv_loss_res = get_tv_loss(res_flow)
                tv_loss += 1e1*tv_loss_res
                '''
                # added_g = detached_flow.grad
                # detached_flow = None
                # sampling_norm = flow_sampling.norm()
                M = flow_sampling.max().detach()
                # M = flow_sampling.median().detach()
                # M = flow_sampling.mean().detach()
                assert (flow_sampling >= 0 ).all()
                sampling_norm = (( (flow_sampling/M) * (flow_sampling == flow_sampling.max()).float())**2).sum()
                sampling_norm1 = (( (flow_sampling) * (flow_sampling == flow_sampling.max()).float())**2).sum()
                '''
                """
                if flow_sampling.shape[-2] == 256:
                    import pdb;pdb.set_trace()
                """
                # sampling_norm = (flow_sampling).sum()
                # sampling_norm = (flow_sampling*th.log(flow_sampling+1e-8)).sum()
                flow_sampling = None
                # this will populate the detached_flow grad
                '''
                global counter
                if 'counter' not in globals():
                    counter = 1
                max_iters = 5000
                sampling_loss_weight = (1e0 - 1e-1) * float((counter<(max_iters//5))) + 1e-1
                '''
                sampling_loss_weight = 1
                (sampling_loss_factor*sampling_loss_weight*sampling_norm).backward()
                D = res_img.shape[-2]
                if False:
                    self.create_grid([detached_flow.grad[:,:1,...]], [os.path.join(sample_dir1, f'{D}_x_{D}', f"grad_of_flow_" +
                                    str(epoch) + "_" +
                                    str(batch) + ".png")])        
                def add_flow_norm_grad(g,
                    # added_g=added_g
                    detached_flow = detached_flow
                    ):
                    # g = g + added_g#[...,::2,::2]
                    # import pdb;pdb.set_trace()
                    # g = g + th.clamp(detached_flow.grad,-1.,1.)
                    if True:
                        g = g + (detached_flow.grad/M)
                    return g
                # if res_flow.shape[-2] == 256:
                if True:
                    res_flow.register_hook(add_flow_norm_grad)
                # self.trends[f'sampling_norm_loss_{j}'].append(th.log(sampling_norm).item())
                self.trends[f'sampling_norm_loss_{j}'].append((sampling_norm).item())
                self.trends[f'tv_loss_{j}'].append((tv_loss_res).item())
                fake_flow[j] = None
            # print('early return from optimize_generator');return 0
        #=====================================================
        # print('not using gen loss')
        (1e-1*loss + 0*total_variance_loss + 0*tv_loss).backward()
        # print('not stepping in gen')
        # th.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=.1, norm_type='inf')
        '''
        pgrads = []
        V = 0.5
        # V = None
        for p in gen_optim.param_groups[0]['params']:
            if p.grad is not None:
                if p.ndim == 4:
                    # pgrads.append(p.grad.abs().max().item())
                    pnorm = p.flatten(start_dim=1,end_dim=-1).norm(dim=-1,keepdim=True)[:,:,None,None]
                    rel_p_grad = p.grad/pnorm
                    if V is not None:
                        rel_p_grad = th.clip(rel_p_grad,-V,V)
                    # p.grad = rel_p_grad * pnorm
                    p.grad.data.copy_(rel_p_grad * pnorm )
                if p.ndim == 2:
                    # pgrads.append(p.grad.abs().max().item())
                    pnorm = p.norm(dim=(-1),keepdim=True)
                    rel_p_grad = p.grad/pnorm
                    if V is not None:
                        rel_p_grad = th.clip(rel_p_grad,-V,V)
                    p.grad.data.copy_(rel_p_grad * pnorm )
        # print(max(pgrads))
        '''
        # import pdb;pdb.set_trace()
        
        gen_optim.step()

        return loss.item()

    @staticmethod
    def create_grid(samples, img_files):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing list[Tensors]
        :param img_files: list of names of files to write
        :return: None (saves multiple files)
        """
        from torchvision.utils import save_image
        from numpy import sqrt
        
        # save the images:
        for sample, img_file in zip(samples, img_files):
            sample = th.clamp((sample.detach() / 2) + 0.5, min=0, max=1)
            save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])))

    def train(self, data, gen_optim, dis_optim, loss_fn,
              start=1, num_epochs=12, feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=64,
              log_dir=None, sample_dir="./samples",
              save_dir="./models"):

        # TODOcomplete write the documentation for this method
        # no more procrastination ... HeHe
        """
        Method for training the network
        :param data: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap this inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap this inside a Scheduler if you want to
        :param loss_fn: Object of GANLoss
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note this is absolute and not relative to start
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after these many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: path to directory for saving the loss.log file
        :param sample_dir: path to directory for saving generated samples' grids
        :param save_dir: path to directory for saving the trained models
        :return: None (writes multiple files to disk)
        """

        from torch.nn.functional import avg_pool2d
        global epoch,batch,sample_dir1
        sample_dir1 = sample_dir
        # turn the generator and discriminator into train mode
        self.gen.train()
        for dis in self.dis_list:
            dis.train()

        assert isinstance(gen_optim, th.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, th.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        print("Starting the training process ... ")

        # create fixed_input for debugging
        # fixed_input = th.randn(num_samples, self.latent_size,*self.latent_spatial).to(self.device)
        fixed_input = [
                    th.randn(
                    num_samples,*s[1:]).to(self.device) for s in self.latent_spatials[::-1]
                    ][::-1]
        if False:
            print('setting fixed input to 0')
            fixed_input = th.zeros_like(fixed_input)
        # create a global time counter
        global_time = time.time()

        for epoch in range(start, num_epochs + 1):
            epoch = epoch
            start = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            total_batches = len(iter(data))

            limit = int((data_percentage / 100) * total_batches)
            # import pdb;pdb.set_trace()
            if epoch > 1:
                print(i)
            # for (i, batch) in enumerate(data, 1):
            batch = next(iter(data))
            images = batch.to(self.device)
            extracted_batch_size = images.shape[0]

            # create a list of downsampled images from the real images:
            images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                                    for i in range(1, self.depth)]
            images = list(reversed(images))
            n_batches = 100
            for i in range(n_batches):
                batch = i
                # import pdb;pdb.set_trace()
                '''
                # extract current batch of data for training
                images = batch.to(self.device)
                extracted_batch_size = images.shape[0]

                # create a list of downsampled images from the real images:
                images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                                     for i in range(1, self.depth)]
                images = list(reversed(images))
                '''
                

                gan_input = [
                    th.randn(
                    extracted_batch_size,*s[1:]).to(self.device) for s in self.latent_spatials[::-1]
                    ][::-1]
                '''
                torch.Size([1, 64, 1, 4])
                torch.Size([1, 64, 4, 7])
                torch.Size([1, 64, 8, 14])
                torch.Size([1, 64, 16, 28])
                torch.Size([1, 64, 32, 56])
                torch.Size([1, 32, 64, 112])
                torch.Size([1, 16, 128, 224])
                '''
                

                if False:
                    print('setting gan input to fixed_input')
                    gan_input =fixed_input
                # gan_input = gan_input[...,None,None]
                # optimize the discriminator:
                if True:
                    dis_loss = self.optimize_discriminator(dis_optim, gan_input,
                                                       images, loss_fn)
                else:
                    dis_loss = 0 
                # print('early return from train loop');continue
                # optimize the generator:
                # resample from the latent noise
                gan_input = [
                    th.randn(
                    extracted_batch_size,*s[1:]).to(self.device) for s in self.latent_spatials[::-1]
                    ][::-1]
                if False:
                    print('setting gan input to fixed_input')
                    gan_input = fixed_input
                
                '''
                gan_input = gan_input[...,None,None]
                gan_input = th.zeros(gan_input.shape[:2] + (1,2)).to(gan_input.device); print('setting the shape of gan_input to ', gan_input.shape)
                # 1,:,1,2 -> 1,3,256,320
                # 1,:,1,3 -> 1,3,256,384
                outputs = self.gen(gan_input)
                print('see effect of gan input shape');import pdb;pdb.set_trace()
                '''
                if True and 'optimize generator':
                    gen_loss = self.optimize_generator(gen_optim, gan_input,
                                                   images, loss_fn)
                else:
                    gen_loss = 0.

                
                # provide a loss feedback
                # import pdb;pdb.set_trace()
                # if i % int(limit / feedback_factor) == 0 or i == 1:
                if  i >100:
                    import pdb;pdb.set_trace()
                if (i == (n_batches - 1)) or (i == 1):
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] batch: %d  d_loss: %f  g_loss: %f"
                          % (elapsed, i, dis_loss, gen_loss))

                    # also write the losses to the log file:
                    if log_dir is not None:
                        log_file = os.path.join(log_dir, "loss.log")
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, "a") as log:
                            log.write(str(dis_loss) + "\t" + str(gen_loss) + "\n")

                    # create a grid of samples and save it
                    reses = [str(int(np.power(2, dep))) + "_x_"
                             + str(int(np.power(2, dep)))
                             for dep in range(2, self.depth + 2)]
                    # if True and 'dont visualize':
                    visualize(self,epoch,i,sample_dir,fixed_input,images[self.min_scale:])
                    """
                    gen_img_files = [os.path.join(sample_dir, res, "gen_" +
                                                  str(epoch) + "_" +
                                                  str(i) + ".png")
                                     for res in reses]

                    # Make sure all the required directories exist
                    # otherwise make them
                    os.makedirs(sample_dir, exist_ok=True)

                    if i == 1:
                        # import pdb;pdb.set_trace()
                        os.makedirs(os.path.join(sample_dir,'real'), exist_ok=True)
                        real_img_files = [os.path.join(sample_dir,'real', res, "real_" +
                                                                        str(epoch) + "_" +
                                                                        str(i) + ".png")
                                                            for res in reses]
                        for real_img_file in real_img_files:
                            os.makedirs(os.path.dirname(real_img_file), exist_ok=True)                        

                        self.create_grid(images, real_img_files)

                    for gen_img_file in gen_img_files:
                        os.makedirs(os.path.dirname(gen_img_file), exist_ok=True)

                    dis_optim.zero_grad()
                    gen_optim.zero_grad()
                    with th.no_grad():
                        self.create_grid(self.gen(fixed_input), gen_img_files)
                    """
                # if i > limit:
                #     break

            # calculate the time required for the epoch
            stop = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(save_dir,
                                                   "GAN_GEN_OPTIM_" + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(save_dir,
                                                   "GAN_DIS_OPTIM_" + str(epoch) + ".pth")

                th.save(self.gen.state_dict(), gen_save_file)
                import itertools
                th.save([dis.state_dict() for dis in self.dis_list], dis_save_file)
                th.save(gen_optim.state_dict(), gen_optim_save_file)
                th.save(dis_optim.state_dict(), dis_optim_save_file)

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        for dis in self.dis_list:
            dis.eval()
