import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

rng = np.random.RandomState(2345234)


def sort_samples(loss):
    loss_ratio = loss[:, 0] / loss[:, 1]
    _, idx1 = torch.sort(loss_ratio, dim=0, descending=False)
    return idx1


class TrainAlgorithm:

    def __init__(self, attention, cluster, batch_size=1024, capacity=2 ** 11, lr=1e-4, betas=(0.9, 0.999), ess=1024):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._device = torch.device("cpu")

        assert batch_size < capacity
        self.batch_size = batch_size
        self.capacity = capacity
        self.logger = SummaryWriter('./logs')

        self.attention = attention.to(self._device)
        self.cluster = cluster
        self.num_models = len(self.cluster)
        # multiple separators
        self.separator = np.zeros((self.num_models - 1, self.num_models))
        for i in range(self.num_models - 1):
            self.separator[i, 0] = -1

        self.plot_2D = False
        self.plot_3D = False
        self.idx = 0
        self.epoch_sorting_size = ess
        self.attention_step = 0
        self.cluster_step = 0
        self.model_pointer_cluster = 0
        self.model_pointer_attention = 0
        self.rec = []
        self.overall_mean_loss = np.empty((0))

        for i in range(self.num_models):
            cluster[i].to(self._device)
        self.cluster_optimizer = []
        self.activities = torch.zeros(self.num_models, device=self._device)
        self.stored_activities = torch.ones(self.num_models, device=self._device)
        self.loss = torch.nn.L1Loss(reduction='none')
        self.loss_update = torch.nn.L1Loss()

        self.attention_optimizer = torch.optim.Adam(self.attention.parameters(), lr=lr, betas=betas)
        for c in range(self.num_models):
            self.cluster_optimizer += [torch.optim.Adam(self.cluster[c].parameters(), lr=lr, betas=betas)]

    def sort_epoch_batch(self, X, Y):

        """

            1. Compute losses of whole batch for sub networks from clusters
            2. create batches size of (batch / self.num_models) for each model
            3. optimize

        """

        epoch_batch_return = []
        statistics_indexes = []
        statistics_actions = []
        statistics_losses = []

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, 1024)
        epoch_batch = np.empty((0, self.num_models))
        batch_x = np.empty((0, X.shape[1]))
        batch_y = np.empty((0, Y.shape[1]))

        """ Compute loss of batches """

        for i, batch in enumerate(loader):
            x, y = batch
            x = x.to(self._device)
            y = y.to(self._device)
            logits = self.cluster.forward(x)
            losses = [torch.mean(self.loss(logit, y), 1) for logit in logits]
            losses = torch.transpose(torch.stack(losses, dim=0), 0, 1).cpu().detach().numpy()
            epoch_batch = np.append(epoch_batch, losses, axis=0)
            batch_x = np.append(batch_x, x.cpu().detach().numpy(), axis=0)
            batch_y = np.append(batch_y, y.cpu().detach().numpy(), axis=0)

        for b in range(int(X.shape[0]/self.epoch_sorting_size)):
            x = batch_x[b*self.epoch_sorting_size:(b + 1)*self.epoch_sorting_size]
            y = batch_y[b*self.epoch_sorting_size:(b + 1)*self.epoch_sorting_size]

            losses = epoch_batch[b*self.epoch_sorting_size:(b + 1)*self.epoch_sorting_size]
            """ create model-specific batches """

            batch_size = int(x.shape[0] / self.num_models)
            models_without_batch = np.arange(self.num_models)
            xy = [[np.empty((0, x.shape[1])), np.zeros((0, y.shape[1]))]for _ in range(self.num_models)]
            # recorded_lossses = [np.empty((0, self.num_models))for _ in range(self.num_models)]
            # recorded_indexes = [np.empty((0, self.num_models))for _ in range(self.num_models)]
            # idxs = np.zeros(self.num_models)

            loss_radii = np.sqrt(np.sum(np.power(losses, 2), axis=1))
            sorted_losses = np.argsort(loss_radii)[::-1]
            loss_indexes = np.argsort(losses[sorted_losses], axis=1)

            for l_i in zip(sorted_losses, loss_indexes):
                idx, loss_index = l_i
                loss_index = [i for i in loss_index if i in models_without_batch][0]

                xy[loss_index][0] = np.append(xy[loss_index][0], [x[idx]], axis=0)
                xy[loss_index][1] = np.append(xy[loss_index][1], [y[idx]], axis=0)

                for i in models_without_batch:
                    if xy[i][0].shape[0] >= batch_size:
                        models_without_batch = models_without_batch[models_without_batch != i]
                if len(models_without_batch) == 0:
                    break
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

            # if self.plot_3D:
            #
            #     # plot the surface
            #     plt3d = plt.figure().gca(projection='3d')
            #     ax = plt.gca()
            #
            #     l0 = losses[loss_ratios[0]]
            #     l1 = losses[loss_ratios[1]]
            #     l2 = losses[loss_ratios[2]]
            #
            #
            #     ax.scatter(l0[:, 0], l0[:, 1], l0[:, 2], c='green')
            #     ax.scatter([0.3], [0], [0], c='green')
            #
            #     ax.scatter(l1[:, 0], l1[:, 1], l1[:, 2], c='black')
            #     ax.scatter([0], [0.3], [0], c='black')
            #
            #     ax.scatter(l2[:, 0], l2[:, 1], l2[:, 2], c='red')
            #     ax.scatter([0], [0], [0.3], c='red')
            #
            #     ax.set_xlim(0.0, 0.5)
            #     ax.set_ylim(0.0, 0.5)
            #     ax.set_zlim(0.0, 0.5)
            #
            #     ax.set_xlabel('Model 1 - ' + str(l0.shape[0]) + 'samples')
            #     ax.set_ylabel('Model 2 - ' + str(l1.shape[0]) + 'samples')
            #     ax.set_zlabel('Model 3 - ' + str(l2.shape[0]) + 'samples')
            #     # fig.savefig('./plots/plot_separation_plot3D/img' + str(loss_idx) + '-' + str(i) + '.png')
            epoch_batch_return += [xy]

        return epoch_batch_return #,  statistics_indexes, statistics_actions, statistics_losses

    def prioritize_samples(self, sorted_batch):

        epoch_batch_return = []

        for i, modelbatch in enumerate(sorted_batch[0]):

            X, Y = modelbatch

            X = torch.tensor(X, device=self._device, dtype=torch.double)
            Y = torch.tensor(Y, device=self._device, dtype=torch.double)
            dataset = TensorDataset(X, Y)
            loader = DataLoader(dataset, 1024)
            epoch_batch = np.empty(0)
            batch_x = np.empty((0, X.shape[1]))
            batch_y = np.empty((0, Y.shape[1]))

            """ Compute loss of batches """

            for i, batch in enumerate(loader):
                x, y = batch
                x = x.to(self._device)
                y = y.to(self._device)
                logits = self.cluster.forward(x, idx=i)
                losses = torch.mean(self.loss(logits, y), 1).cpu().detach().numpy()
                epoch_batch = np.append(epoch_batch, losses, axis=0)
                batch_x = np.append(batch_x, x.cpu().detach().numpy(), axis=0)
                batch_y = np.append(batch_y, y.cpu().detach().numpy(), axis=0)

            median_loss = np.median(epoch_batch)
            X = X[epoch_batch > median_loss].cpu().detach().numpy()
            Y = Y[epoch_batch > median_loss].cpu().detach().numpy()

            for i, xy in enumerate(zip(X, Y)):

                x, y = xy
                multiplicator = int(median_loss / epoch_batch[i])
                multiplicator = np.clip(multiplicator, 0, 4)

                for i in range(multiplicator):
                    X = np.append(X, [x], axis=0)
                    Y = np.append(Y, [y], axis=0)

            epoch_batch_return += [[X, Y]]

        return [epoch_batch_return]

    def sort_epoch_batch_attention(self, X, Y):

        epoch_batch_return = []

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, 1024)
        epoch_batch = np.empty((0, self.num_models))
        """ Compute loss of batches """
        batch_x = np.empty((0, X.shape[1]))

        for i, batch in enumerate(loader):
            x, y = batch
            x = x.to(self._device)
            y = y.to(self._device)
            logits = self.cluster.forward(x)
            losses = [torch.mean(self.loss(logit, y), 1) for logit in logits]
            losses = torch.transpose(torch.stack(losses, dim=0), 0, 1).cpu().detach().numpy()
            epoch_batch = np.append(epoch_batch, losses, axis=0)
            batch_x = np.append(batch_x, x.cpu().detach().numpy(), axis=0)

        for b in range(int(X.shape[0] / self.epoch_sorting_size)):
            x = batch_x[b * self.epoch_sorting_size: (b + 1) * self.epoch_sorting_size]

            losses = epoch_batch[b * self.epoch_sorting_size:(b + 1) * self.epoch_sorting_size]

            batch_size = int(x.shape[0] / self.num_models)
            models_without_batch = np.arange(self.num_models)
            x_att = [np.empty((0, x.shape[1])) for _ in range(self.num_models)]

            loss_radii = np.sqrt(np.sum(np.power(losses, 2), axis=1))
            sorted_losses = np.argsort(loss_radii)[::-1]
            loss_indexes = np.argsort(losses[sorted_losses], axis=1)

            for l_i in zip(sorted_losses, loss_indexes):
                idx, loss_index = l_i
                loss_index = [i for i in loss_index if i in models_without_batch][0]

                x_att[loss_index] = np.append(x_att[loss_index], [x[idx]], axis=0)

                for i in models_without_batch:
                    if x_att[i].shape[0] >= batch_size:
                        models_without_batch = models_without_batch[models_without_batch != i]
                if len(models_without_batch) == 0:
                    break

            length_batches = []
            length_batches_1 = 0
            # length_batches_2 = x_att[0].shape[0]

            for i in range(self.num_models):
                length_batches.append([length_batches_1, length_batches_1 + x_att[i].shape[0]])
                length_batches_1 += x_att[i].shape[0]

            x_att = np.concatenate(x_att, axis=0)
            y_att = np.zeros((x_att.shape[0], self.num_models))
            for i in range(self.num_models):
                y_att[length_batches[i][0]:length_batches[i][1], i] = 1

            assert x_att.shape[0] == y_att.shape[0]
            self.model_pointer_attention = (self.model_pointer_attention + 1) % self.num_models
            epoch_batch_return.append([x_att, y_att])

        return epoch_batch_return

    """ Update networks """
    def train_cluster(self, epoch_batch):

        mean_loss = np.empty((0))
        for sorted_epoch_batch in epoch_batch:
            self.cluster_step += 1
            for i in range(self.num_models):
                x, y = sorted_epoch_batch[i]
                x = torch.tensor(x, device=self._device, dtype=torch.double)
                y = torch.tensor(y, device=self._device, dtype=torch.double)
                dataset = TensorDataset(x, y)
                loader = DataLoader(dataset, self.batch_size, shuffle=True)
                for step, batch in enumerate(loader):
                    x1, y1 = batch
                    loss = self.cluster_update(x1, y1, i)
                    mean_loss = np.append(mean_loss, [loss.cpu().detach().numpy()], axis=0)

        mean_loss = np.mean(mean_loss)
        self.overall_mean_loss = np.append(self.overall_mean_loss, mean_loss)
        print('Mean Loss: ', mean_loss)

        # self.rec += [[sample_loss], [loss_att1, loss_att2]]

    def train_attention(self, epoch_batch):

        mean_loss = np.empty((0))
        for sorted_epoch_batch in epoch_batch:
            self.attention_step += 1
            x, y = sorted_epoch_batch
            x = torch.tensor(x, device=self._device, dtype=torch.double)
            y = torch.tensor(y, device=self._device, dtype=torch.double)
            dataset = TensorDataset(x, y)
            loader = DataLoader(dataset, self.batch_size, shuffle=True)
            for i, batch in enumerate(loader):
                x_att, y_att = batch
                loss = self.attention_update(x_att, y_att)
                mean_loss = np.append(mean_loss, [loss.cpu().detach().numpy()])

        mean_loss = np.mean(mean_loss)
        self.overall_mean_loss = np.append(self.overall_mean_loss, mean_loss)
        print('Mean Loss: ', mean_loss)

    def attention_update(self, x, y):
        self.attention.zero_grad()
        self.attention_optimizer.zero_grad()
        out = self.attention(x)
        loss = self.loss_update(out, y)
        loss.backward()
        self.attention_optimizer.step()
        return loss

    def cluster_update(self, x, y, idx):
        self.cluster[idx].zero_grad()
        self.cluster_optimizer[idx].zero_grad()
        out = self.cluster[idx](x)
        loss = self.loss_update(out, y)
        loss.backward()
        self.cluster_optimizer[idx].step()
        return loss
