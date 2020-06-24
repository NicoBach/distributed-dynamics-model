import math
from time import sleep
import gym
import custom_gym
import numpy as np
import torch
from pynput import keyboard
from utils import load_checkpoint

LINK_LENGTH_1 = 1.  # [m]
LINK_LENGTH_2 = 1.  # [m]
LINK_MASS_1 = 1.  #: [kg] mass of link 1
LINK_MASS_2 = 1.  #: [kg] mass of link 2
LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
LINK_MOI = 1.  #: moments of inertia for both links

MAX_VEL_1 = np.pi * 4  # * 4
MAX_VEL_2 = np.pi * 9  # * 9

book_or_nips = "book"

dt = .02

AVAIL_TORQUE = [-1., 0., +1]


def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)

class Listener:

    def __init__(self, a):
        self.action = a
        self.lis = keyboard.Listener(on_press=self.on_press)
        self.lis.start()
        # self.lis.join()

    def on_press(self, key):
        try:
            k = key.char  # single-char keys
        except:
            k = key.name  # other keys
        if key == keyboard.Key.esc: return False  # stop listener

        if k == 'left':
            # Push left
            self.action *= 0
            self.action[0] = 1
        if k == 'down':
            # no push
            self.action *= 0
            self.action[1] = 1
        if k == 'right':
            # push right
            self.action *= 0
            self.action[2] = 1

    # def on_release(self, key):
    #     self.action *= 0
    #     self.action[1] = 1


rng = np.random.RandomState(23456)
save_path = './models/acrobot_8D_dt0.02-new-alg-tanh-WORKING'
num_models = 8

norm_data = np.load(save_path + '/norm_data.npz')

Xmean, Xstd = norm_data['arr_0']
Ymean, Ystd = norm_data['arr_1']

att, cluster = load_checkpoint(save_path, 19)
# att, _ = load_checkpoint('./models/acrobot_8D_dt0.02-new-alg-tanh', 19)
att = att.cpu()
for c in range(len(cluster)):
    cluster[c].cpu()

rec = []
rec2 = []
velocities = []
att_list = []

env = gym.make('abEnv-v0')
start_obs = env.reset()

sampled_action = env.action_space.sample()

action = np.zeros(3)
action[1] = 1

listener = Listener(action)

obs = start_obs
# state = start_obs
state = [0., 0., -0., -0.]
# obs, _, _, _ = env.step(np.argmax(action))
sample = np.zeros(54)
x_j1 = []
x_j2 = []
x_vels = []
x_actions = []

gathered_model_selection = np.zeros((1000, num_models))

for _ in range(1):
    for i in range(10000):

        # COMPARISON TO OUTPUT OF ENVIRONMENT


        # x1 = np.append(math.cos(obs[0]), math.sin(obs[0]))
        # x_j1 = [x1] + x_j1
        # x_j1 = x_j1[:6]
        # x2 = np.append(math.cos(obs[1]), math.sin(obs[1]))
        # x_j2 = [x2] + x_j2
        # x_j2 = x_j2[:6]
        # x_vel = np.append(obs[2], obs[3])
        # x_vels = [x_vel] + x_vels
        # x_vels = x_vels[:6]
        # x_actions = [action] + x_actions
        # x_actions = x_actions[:6]

        # AUTOREGRESSIVE

        x1 = np.append(math.cos(state[0]), math.sin(state[0]))
        x_j1 = [x1] + x_j1
        x_j1 = x_j1[:6]
        x2 = np.append(math.cos(state[1]), math.sin(state[1]))
        x_j2 = [x2] + x_j2
        x_j2 = x_j2[:6]
        x_vel = np.append(state[2], state[3])
        x_vels = [x_vel] + x_vels
        x_vels = x_vels[:6]
        x_actions = [action] + x_actions
        x_actions = x_actions[:6]

        stop = i - 5 if i - 6 < 0 else 0
        for j in range(6 + stop):
            sample[j * 2:(j + 1) * 2] = x_j1[j]
            sample[j + 12] = x_vels[j][0]
            sample[j * 2 + 18: (j + 1) * 2 + 18] = x_j2[j]
            sample[j + 30] = x_vels[j][1]
            sample[j * 3 + 36:(j + 1) * 3 + 36] = x_actions[j]
        X = sample
        """ Normalize input """
        # X[:, 4] = (2 * (X[:, 4] - low_x[0])) / (high_x[0] - low_x[0]) - 1
        # X[:, 5] = (2 * (X[:, 5] - low_x[1])) / (high_x[1] - low_x[1]) - 1
        # X[:, 6] = (2 * (X[:, 6] - low_x[2])) / (high_x[2] - low_x[2]) - 1
        # X[:, 7] = (2 * (X[:, 7] - low_x[3])) / (high_x[3] - low_x[3]) - 1
        # X[:, 8] = (2 * (X[:, 8] - low_x[4])) / (high_x[4] - low_x[4]) - 1
        X1 = (X - Xmean) / Xstd

        X1 = torch.tensor([X1], dtype=torch.float64)

        """ FORWARD STEP """
        Y_pred_all = cluster.forward(X1)
        Y_pred_all = [(y_pred.detach().numpy().flatten() * Ystd) + Ymean for y_pred in Y_pred_all]

        # attention = torch.softmax(att.forward(X1), dim=1).detach().numpy().flatten()
        # num_models = np.arange(0, attention.shape[0])
        # mean_output = np.empty((0, 4))
        # for i in num_models:
        #     mean_output = np.append(mean_output, [Y_pred_all[i] * attention[i]], axis=0)
        # Y_pred_att_argmax = np.sum(mean_output, axis=0)

        attention = np.argmax(att.forward(X1).detach().numpy())

        gathered_model_selection[i % 1000, :] *= 0
        gathered_model_selection[i % 1000, attention] = 1
        gms = [np.count_nonzero(gathered_model_selection[:, i] == 1) for i in range(num_models)]
        print('model selection in the last 100 steps: ' + str(gms))
        # attention = torch.softmax(att.forward(X1), dim=1).detach().numpy().flatten()
        # num_models = np.arange(0, attention.shape[0])
        # attention = np.random.choice(num_models, p=attention)
        # attention = torch.tensor(attention, dtype=torch.int32)
        Y_pred_att_argmax = Y_pred_all[attention]
        Y_pred_att_argmax = Y_pred_att_argmax
        """ PREDICTION: COS(J1), SIN(J1), COS(J2), SIN(J2)"""
        Y_pred_att_argmax = np.clip(Y_pred_att_argmax, -1, 1)
        y_output1, y_output2 = np.array_split(Y_pred_att_argmax, 2)

        """ PREDICTION: ANGLE-COS(J1), ANGLE-SIN(J1), ANGLE-COS(J2), ANGLE-SIN(J2)"""

        y_output = np.append([math.acos(y_output1[0]), math.asin(y_output1[1])],
                             [math.acos(y_output2[0]), math.asin(y_output2[1])])
        y_output_neg = np.append([math.acos(-y_output1[0]), math.asin(-y_output1[1])],
                                 [math.acos(-y_output2[0]), math.asin(-y_output2[1])])
        """ COMPUTE WHICH QUADRANT OF CIRCLE, SMOOTH INTERFERENCE BETWEEN COSINE AND SINE ANGLE """

        min_dist_j1 = np.argmin(
            [y_output[0] - y_output[1], y_output[0] - y_output_neg[1],
             y_output_neg[0] - y_output_neg[1], y_output_neg[0] - y_output[1]])

        if min_dist_j1 == 0:
            w = Y_pred_att_argmax[0] ** 2 / (Y_pred_att_argmax[0] ** 2 + Y_pred_att_argmax[1] ** 2)
            y_out_1 = y_output[1] * w + y_output[0] * (1 - w)
            # y_out_1 = (y_output[1] + y_output[0])/2

        elif min_dist_j1 == 1:
            w = Y_pred_att_argmax[0] ** 2 / (Y_pred_att_argmax[0] ** 2 + Y_pred_att_argmax[1] ** 2)
            y_out_1 = -y_output_neg[1] * w - y_output[0] * (1 - w)
            # y_out_1 = (-y_output_neg[1] - y_output[0])/2
        elif min_dist_j1 == 2:
            # w = Y_pred_att_argmax[0] ** 2 / (Y_pred_att_argmax[0] ** 2 + Y_pred_att_argmax[1] ** 2)
            y_out_1 = -np.pi + y_output_neg[1] * w + y_output_neg[0] * (1 - w)
            # y_out_1 = (-2*np.pi + y_output_neg[1] + y_output_neg[0])/2
        elif min_dist_j1 == 3:
            w = Y_pred_att_argmax[0] ** 2 / (Y_pred_att_argmax[0] ** 2 + Y_pred_att_argmax[1] ** 2)
            y_out_1 = np.pi - y_output[1] * w - y_output_neg[0] * (1 - w)
            # y_out_1 = (2*np.pi - y_output[1] - y_output_neg[0])/2


        # print("min dist: ", min_dist_j1, 'weight: ', w)

        min_dist_j2 = np.argmin(
            [y_output[2] - y_output[3], y_output[2] - y_output_neg[3],
             y_output_neg[2] - y_output_neg[3], y_output_neg[2] - y_output[3]])

        if min_dist_j2 == 0:
            w = Y_pred_att_argmax[2] ** 2 / (Y_pred_att_argmax[2] ** 2 + Y_pred_att_argmax[3] ** 2)
            y_out_2 = y_output[3] * w + y_output[2] * (1 - w)
            # y_out_2 = (y_output[3] + y_output[2])/2
        elif min_dist_j2 == 1:
            w = Y_pred_att_argmax[2] ** 2 / (Y_pred_att_argmax[2] ** 2 + Y_pred_att_argmax[3] ** 2)
            y_out_2 = -y_output_neg[3] * w - y_output[2] * (1 - w)
            # y_out_2 = (-y_output_neg[3] - y_output[2])/2
        elif min_dist_j2 == 2:
            w = Y_pred_att_argmax[2] ** 2 / (Y_pred_att_argmax[2] ** 2 + Y_pred_att_argmax[3] ** 2)
            y_out_2 = (-np.pi + y_output_neg[3] * w) + y_output_neg[2] * (1 - w)
            # y_out_2 = (2*-np.pi + y_output_neg[3] + y_output_neg[2]) /2
        elif min_dist_j2 == 3:
            w = Y_pred_att_argmax[2] ** 2 / (Y_pred_att_argmax[2] ** 2 + Y_pred_att_argmax[3] ** 2)
            y_out_2 = np.pi - y_output[3] * w - y_output_neg[2] * (1 - w)
            # y_out_2 = (2*np.pi - y_output[3] - y_output_neg[2])/2

        # y_out_1 = y_output[1]
        # y_out_2 = 0

        """ COMPUTE AND CLIP VELOCITIES """
        if np.abs(y_out_1 - state[0]) > np.pi:
            # s0 = np.pi - np.abs(obs[0])
            s0 = np.pi - np.abs(state[0])
            s1 = np.pi - np.abs(y_out_1)
            if y_out_1 > state[0]:
                vel1 = ((s0 + s1) * -1) / dt
            else:
                vel1 = (s0 + s1) / dt
        else:
            # vel1 = (y_out_1 - obs[0]) / dt
            vel1 = (y_out_1 - state[0]) / dt

        if np.abs(y_out_2 - state[1]) > np.pi:
            # s0 = np.pi - np.abs(obs[1])
            s0 = np.pi - np.abs(state[1])
            s1 = np.pi - np.abs(y_out_2)
            if y_out_2 > state[1]:
                vel2 = ((s0 + s1) * -1) / dt
            else:
                vel2 = (s0 + s1) / dt
        else:
            # vel2 = (y_out_2 - obs[1]) / dt
            vel2 = (y_out_2 - state[1]) / dt

        vel = np.append(vel1, vel2)
        vel[0] = bound(vel[0], -MAX_VEL_1, MAX_VEL_1)
        vel[1] = bound(vel[1], -MAX_VEL_2, MAX_VEL_2)
        y_out_1 = wrap(y_out_1, -np.pi, np.pi)
        y_out_2 = wrap(y_out_2, -np.pi, np.pi)

        """ STATE: ANGLE J1, ANGLE J2, A-VELOCITY J1, A-VELOCITY J2 """

        y_o = np.append(y_out_1, y_out_2)
        state = np.append(y_o, vel, axis=0)

        # Set env
        env.set_state(state)
        # obs, _, _, _ = env.step(np.argmax(action))

        env.render()

        """ RECORD """
        print("Prediction of state ", attention, "\n", state)
        # rec += [np.append([obs], [np.array([y_out_1, y_out_2, 0, 0])], axis=0)]
        # velocities += [np.append([state], [Y_pred_att_argmax], axis=0)]
        # att_list += [attention]
        # rec2 += [obs, action]
    env.reset()
# np.savez('additional_training.npz', rec2)
# np.savez('./acrobot_data5.npz', rec, velocities, att_list)
env.close()
