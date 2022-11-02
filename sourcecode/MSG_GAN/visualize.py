from matplotlib import pyplot as plt
import os
def visualize(msg_gan,i,sample_dir,fixed_input):
    # create a grid of samples and save it
    reses = [str(int(np.power(2, dep))) + "_x_"
                + str(int(np.power(2, dep)))
                for dep in range(2, msg_gan.depth + 2)]


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

        msg_gan.create_grid(images, real_img_files)

    for gen_img_file in gen_img_files:
        os.makedirs(os.path.dirname(gen_img_file), exist_ok=True)

    with th.no_grad():
        msg_gan.create_grid(msg_gan.gen(fixed_input), gen_img_files)
