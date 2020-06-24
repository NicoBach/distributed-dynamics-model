import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('tkagg')
rng = np.random.RandomState(2345234)


def sort_samples(loss):
    loss_ratio = loss[:, 0] / loss[:, 1]
    _, idx1 = torch.sort(loss_ratio, dim=0, descending=False)
    return idx1


class TrainAlgorithm:

    def __init__(self, attention, cluster, batch_size=1024, capacity=2 ** 11, lr=1e-4, betas=(0.9, 0.999)):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._device = torch.device("cpu")

        assert batch_size < capacity
        self.batch_size = batch_size
        self.capacity = capacity

        self.attention = attention.to(self._device)
        self.cluster = cluster
        self.num_models = len(self.cluster)
        # multiple separators
        self.separator = np.zeros((self.num_models - 1, self.num_models))
        for i in range(self.num_models - 1):
            self.separator[i, 0] = -1

        self.plot_2D = False
        self.plot_3D = False
        self.nu = 1e-5
        self.nu_inc = 5e-1
        self.idx = 0

        self.rec = []

        for i in range(self.num_models):
            cluster[i].to(self._device)
        self.cluster_optimizer = []
        self.activities = torch.zeros(self.num_models, device=self._device)
        self.stored_activities = torch.ones(self.num_models, device=self._device)
        self.loss = torch.nn.SmoothL1Loss(reduction='none')
        self.loss_update = torch.nn.SmoothL1Loss()

        self.attention_optimizer = torch.optim.Adam(self.attention.parameters(), lr=lr, betas=betas)
        for c in range(self.num_models):
            self.cluster_optimizer += [torch.optim.Adam(self.cluster[c].parameters(), lr=lr, betas=betas)]

    def sort_epoch_batch(self, epoch_batch):

        """

            1. Compute losses of whole batch for sub networks from clusters
            2. create batches size of (batch / self.num_models) for each model
            3. optimize

        """
        epoch_batch_return = []
        statistics_indexes = []
        statistics_actions = []
        statistics_losses = []
        for batch in epoch_batch:
            x, y = batch

            x = x.to(self._device)
            y = y.to(self._device)

            """ Compute loss of batches """
            logits = self.cluster.forward(x)
            losses = [torch.mean(self.loss(logit, y), 1) for logit in logits]
            losses = torch.transpose(torch.stack(losses, dim=0), 0, 1).cpu().detach().numpy()

            """ create model-specific batches """
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            batch_size = int(x.shape[0] / self.num_models)
            models_without_batch = np.arange(self.num_models)
            indexes_not_used = np.arange((losses.shape[0]))
            xy = [[np.empty((0, x.shape[1])), np.zeros((0, y.shape[1]))]for _ in range(self.num_models)]
            # recorded_lossses = [np.empty((0, self.num_models))for _ in range(self.num_models)]
            # recorded_indexes = [np.empty((0, self.num_models))for _ in range(self.num_models)]
            # idxs = np.zeros(self.num_models)

            for model_no in range(self.num_models):

                loss_ratios = dict()
                loss_pool = losses[indexes_not_used]

                for i in models_without_batch:
                    loss_per_model = []
                    for j in range(int(self.num_models - 1)):
                        loss_per_model.append((indexes_not_used[np.argsort(loss_pool[:, i] / loss_pool[:, (i + j + 1) % self.num_models])])[:int(losses.shape[0] / 2)])
                    loss_ratios[i] = loss_per_model

                gathered_indexes = np.empty(0, dtype='int32')

                for i in models_without_batch:
                    lr = loss_ratios[i]
                    ind_lr = lr[0]
                    for j in range(len(lr) - 1):
                        ind_lr = ind_lr[np.in1d(ind_lr, lr[j + 1])][:(batch_size - xy[i][0].shape[0])]
                    # losses = np.delete(indexes_not_used, ind_lr, axis=0)
                    xy[i][0] = np.append(xy[i][0], x[ind_lr], axis=0)
                    xy[i][1] = np.append(xy[i][1], y[ind_lr], axis=0)
                    gathered_indexes = np.append(gathered_indexes, ind_lr, axis=0)

                for i in models_without_batch:
                    if xy[i][0].shape[0] >= batch_size:
                        models_without_batch = models_without_batch[models_without_batch != i]
                        # if model_no == 0:
                        #     idxs[i] = 1

                if len(models_without_batch) == 0:
                    break

                indexes_not_used = np.setdiff1d(indexes_not_used, gathered_indexes)
            # gather actions

            # recorded_indexes = np.append(recorded_indexes, idxs)

            # a1 = x[:, 2]
            # a2 = x[:, 3]
            # a3 = x[:, 4]
            # action1 = losses[a1 != 0]
            # action2 = losses[a2 != 0]
            # action3 = losses[a3 != 0]
            #
            # statistics_actions += [[action1, action2, action3]]
            #
            # statistics_indexes += [recorded_indexes]
            #
            # statistics_losses += [recorded_lossses]

            if self.plot_3D:

                # plot the surface
                plt3d = plt.figure().gca(projection='3d')
                ax = plt.gca()

                l0 = losses[gathered_indexes[0]]
                l1 = losses[gathered_indexes[1]]
                l2 = losses[gathered_indexes[2]]


                ax.scatter(l0[:, 0], l0[:, 1], l0[:, 2], c='green')
                ax.scatter([0.3], [0], [0], c='green')

                ax.scatter(l1[:, 0], l1[:, 1], l1[:, 2], c='black')
                ax.scatter([0], [0.3], [0], c='black')

                ax.scatter(l2[:, 0], l2[:, 1], l2[:, 2], c='red')
                ax.scatter([0], [0], [0.3], c='red')

                ax.set_xlim(0.0, 0.5)
                ax.set_ylim(0.0, 0.5)
                ax.set_zlim(0.0, 0.5)

                ax.set_xlabel('Model 1 - ' + str(l0.shape[0]) + 'samples')
                ax.set_ylabel('Model 2 - ' + str(l1.shape[0]) + 'samples')
                ax.set_zlabel('Model 3 - ' + str(l2.shape[0]) + 'samples')
                # fig.savefig('./plots/plot_separation_plot3D/img' + str(loss_idx) + '-' + str(i) + '.png')

            epoch_batch_return += [xy]

        return epoch_batch_return #,  statistics_indexes, statistics_actions, statistics_losses

    def sort_epoch_batch_attention(self, epoch_batch):

        epoch_batch_return = []

        for batch in epoch_batch:

            x, y = batch

            x = x.to(self._device)
            y = y.to(self._device)

            """ Compute loss of batches """
            logits = self.cluster.forward(x)
            losses = [torch.mean(self.loss(logit, y), 1) for logit in logits]
            losses = torch.transpose(torch.stack(losses, dim=0), 0, 1).cpu().detach().numpy()

            """ create model-specific batches """
            x = x.cpu().detach().numpy()

            batch_size = int(x.shape[0] / self.num_models)
            models_without_batch = np.arange(self.num_models)
            indexes_not_used = np.arange((losses.shape[0]))
            x_att = [np.empty((0, x.shape[1])) for _ in range(self.num_models)]

            for model_no in range(self.num_models):

                loss_ratios = dict()
                loss_pool = losses[indexes_not_used]

                for i in models_without_batch:
                    loss_per_model = []
                    for j in range(int(self.num_models - 1)):
                        loss_per_model.append((indexes_not_used[
                            np.argsort(loss_pool[:, i] / loss_pool[:, (i + j + 1) % self.num_models])])[
                                              :int(losses.shape[0] / 2)])
                    loss_ratios[i] = loss_per_model

                gathered_indexes = np.empty(0, dtype='int32')

                for i in models_without_batch:
                    lr = loss_ratios[i]
                    ind_lr = lr[0]
                    for j in range(len(lr) - 1):
                        ind_lr = ind_lr[np.in1d(ind_lr, lr[j + 1])][:(batch_size - x_att[i].shape[0])]
                    x_att[i] = np.append(x_att[i], x[ind_lr], axis=0)
                    gathered_indexes = np.append(gathered_indexes, ind_lr, axis=0)

                for i in models_without_batch:
                    if x_att[i].shape[0] >= batch_size:
                        models_without_batch = models_without_batch[models_without_batch != i]

                if len(models_without_batch) == 0:
                    break

                indexes_not_used = np.setdiff1d(indexes_not_used, gathered_indexes)

            length_batches = []
            length_batches_1 = 0
            # length_batches_2 = x_att[0].shape[0]

            for i in range(self.num_models):
                length_batches.append([length_batches_1, length_batches_1 + x_att[i].shape[0]])
                length_batches_1 += x_att[i].shape[0]

            x_att = np.concatenate(x_att, axis=0)
            y_att = torch.zeros(x_att.shape[0], self.num_models, device=self._device, dtype=torch.float)
            for i in range(self.num_models):
                y_att[length_batches[i][0]:length_batches[i][1], i] = 1

            assert x_att.shape[0] == y_att.shape[0]

            epoch_batch_return.append([torch.tensor(x_att, device=self._device, dtype=torch.float), y_att])

        return epoch_batch_return

    """ Update networks """
    def train_cluster(self, epoch_batch):

        for batch in epoch_batch:

            for i in range(self.num_models):
                x, y = batch[i]
                self.cluster_update(torch.tensor(x, device=self._device, dtype=torch.float), torch.tensor(y, device=self._device, dtype=torch.float), i)

        # self.rec += [[sample_loss], [loss_att1, loss_att2]]

    def train_attention(self, epoch_batch):

        for batch in epoch_batch:
            x_att, y_att = batch
            self.attention_update(x_att, y_att)

    def attention_update(self, x, y):
        self.attention.zero_grad()
        self.attention_optimizer.zero_grad()
        out = self.attention(x)
        loss = self.loss_update(out, y)
        loss.backward()
        self.attention_optimizer.step()

    def cluster_update(self, x, y, idx):
        self.cluster[idx].zero_grad()
        self.cluster_optimizer[idx].zero_grad()
        out = self.cluster[idx](x)
        loss = self.loss_update(out, y)
        loss.backward()
        self.cluster_optimizer[idx].step()
