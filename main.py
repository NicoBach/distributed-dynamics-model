import numpy as np
import torch
import math
from networks import AttentionNetwork, NetworkCluster
import algorithm as tr
from torch.utils.data import DataLoader, TensorDataset
from utils import *

rng = np.random.RandomState(23457)

""" Load Data """
save_path = './models/mountaincar_test'

database = np.load('mountaincar_data.npz', allow_pickle=True)
database = np.reshape(database['arr_0'], (int(len(database['arr_0']) / 5), 5))
# 4 input space X_n -Y_n --> output X_n+1
X = np.array([np.append(i[0], i[1]) for i in database])
Y = np.stack(database[:, -2]).squeeze(axis=1)

""" Normalize Data """
min_position = -1.2
max_position = 0.6
max_speed = 0.07


X[:, 0] = (X[:, 0] - min_position) / (max_position - min_position)
X[:, 1] = (X[:, 1] + max_speed) / (max_speed + max_speed)

idx1 = np.arange(len(X), dtype='int32')
idx2 = np.array(X[:, 2], dtype='int32')

actions = np.zeros_like(X, dtype='float32')
actions[idx1, idx2] = 1

X = np.append(X[:, :2], actions, axis=1)

Y[:, 0] = (Y[:, 0] - min_position) / (max_position - min_position)
Y[:, 1] = (Y[:, 1] + max_speed) / (max_speed + max_speed)

print(X.shape, Y.shape)

""" Build Model and Algorithm """

X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()
state_size = Y.shape[1]

num_models = 4
batch_size = 1024
I = np.arange(len(X))

att = AttentionNetwork([X.shape[1], 128, 128, 64, num_models])
cluster = NetworkCluster([X.shape[1], 128, 128, state_size], num_models)
# att, cluster = load_checkpoint(save_path, 14)

algo = tr.TrainAlgorithm(att, cluster)

""" Start Training """

for me in range(20):

    # rng.shuffle(I)

    print('\n[MacroEpoch] %03i' % me)
    print('\n Cluster Training')
    for bi in range(100):

        print('\n[Epoch] %03i' % bi)

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size)
        epoch_batch = []
        for i, batch in enumerate(loader):
            epoch_batch += [batch]
        sorted_epoch_batch = algo.sort_epoch_batch(epoch_batch)

        algo.train_cluster(sorted_epoch_batch)

    print('\n Attention Training')
    for bi in range(100):

        print('\n[Epoch] %03i' % bi)

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size)
        epoch_batch = []
        for i, batch in enumerate(loader):
            epoch_batch += [batch]
        sorted_epoch_batch = algo.sort_epoch_batch_attention(epoch_batch)

        algo.train_attention(sorted_epoch_batch)
    save_checkpoint(att, cluster, save_path, me)

print('well done')
