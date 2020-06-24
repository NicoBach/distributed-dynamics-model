import torch
import numpy as np


def save_network(network, save_path):
    """ Load Control Points """

    layer = network.get_weights()

    w0 = layer[0]
    w1 = layer[2]
    w2 = layer[4]
    w3 = layer[6]
    w4 = layer[8]
    w5 = layer[10]

    b0 = layer[1]
    b1 = layer[3]
    b2 = layer[5]
    b3 = layer[7]
    b4 = layer[9]
    b5 = layer[11]

    for i in range(4):
        w0[i].astype(np.float32).tofile(save_path + 'W0_%03i.bin' % i)
        w1[i].astype(np.float32).tofile(save_path + 'W1_%03i.bin' % i)
        w2[i].astype(np.float32).tofile(save_path + 'W2_%03i.bin' % i)

        b0[i].astype(np.float32).tofile(save_path + 'b0_%03i.bin' % i)
        b1[i].astype(np.float32).tofile(save_path + 'b1_%03i.bin' % i)
        b2[i].astype(np.float32).tofile(save_path + 'b2_%03i.bin' % i)

    w3.astype(np.float32).tofile(save_path + 'W3_0.bin')
    w4.astype(np.float32).tofile(save_path + 'W3_1.bin')
    w5.astype(np.float32).tofile(save_path + 'W3_2.bin')

    b3.astype(np.float32).tofile(save_path + 'b3_0.bin')
    b4.astype(np.float32).tofile(save_path + 'b3_1.bin')
    b5.astype(np.float32).tofile(save_path + 'b3_2.bin')


def save_checkpoint(model_att, model_cluster, savepath, suffix):
    filename1 = "{logdir}/checkpoint_att.{suffix}.pth.tar".format(
            logdir=savepath, suffix=suffix)
    filename2 = "{logdir}/checkpoint_cl.{suffix}.pth.tar".format(
        logdir=savepath, suffix=suffix)
    torch.save(model_att, filename1)
    torch.save(model_cluster, filename2)


def load_checkpoint(savepath, suffix):
    filename1 = "{logdir}/checkpoint_att.{suffix}.pth.tar".format(
        logdir=savepath, suffix=suffix)
    filename2 = "{logdir}/checkpoint_cl.{suffix}.pth.tar".format(
        logdir=savepath, suffix=suffix)
    model_att = torch.load(filename1)
    model_cl = torch.load(filename2)
    return model_att, model_cl
