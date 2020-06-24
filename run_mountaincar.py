from utils import load_checkpoint
import numpy as np
import gym
import custom_gym

from gym import wrappers
import torch
from pynput import keyboard
from time import sleep


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
            print('Key pressed: ' + k)
            self.action *= 0
            self.action[0] = 1
            # print(self.action)
        if k == 'down':
            # no push
            print('Key pressed: ' + k)
            self.action *= 0
            self.action[1] = 1
            # print(self.action)
        if k == 'right':
            # push right
            print('Key pressed: ' + k)
            self.action *= 0
            self.action[2] = 1
            # print(self.action)


rng = np.random.RandomState(23456)
save_path = './models/mountaincar_3D'

action_sequence = np.load('./action_sequence2.npz')['arr_0']
# action_sequence = np.reshape(action_sequence,(1001,3))

att, cluster = load_checkpoint(save_path, 8)

min_position = -1.2
max_position = 0.6
max_speed = 0.07

att = att.cpu()
for c in range(len(cluster)):
    cluster[c].cpu()


env = gym.make('mcEnv-v0')
# env = gym.wrappers.Monitor(env, './video', force=True)
start_obs = env.reset()

start_action = np.zeros(3)
start_action[env.action_space.sample()] = 1
action = np.zeros(3)
action[1] = 1

actions = []

# for i in action_sequence:
#     actions += [np.argmax(i)]
# action = action_sequence[0]

listener = Listener(action)

X = torch.tensor([np.append(start_obs, start_action)], dtype=torch.float)
X[:, 0] = (X[:, 0] - min_position) / (max_position - min_position)
X[:, 1] = (X[:, 1] + max_speed) / (max_speed + max_speed)


Y_pred_all = cluster.forward(X)
attention = att.forward(X)

Y_pred_att_argmax = Y_pred_all[torch.argmax(attention)]
Y_pred_att_argmax = Y_pred_att_argmax.detach().numpy().flatten()

# denormalization
Y_pred_att_argmax[0] = (Y_pred_att_argmax[0]) * (max_position - min_position) + min_position
Y_pred_att_argmax[1] = (Y_pred_att_argmax[1]) * (max_speed + max_speed) - max_speed
# clipping
Y_pred_att_argmax[0] = np.clip(Y_pred_att_argmax[0], min_position, max_position)
Y_pred_att_argmax[1] = np.clip(Y_pred_att_argmax[1], -max_speed, max_speed)
action_list = action
obs_real, _, _, _ = env.step(np.argmax(start_action))
recorded_actions = np.empty((0, 3))

for i in range(4001):

    """ Comment this to use hotkeys for controlling the car"""

    action = action_sequence[i]
    # AUTOREGRESSIVE
    X = torch.tensor([np.append(Y_pred_att_argmax, action)], dtype=torch.float)
    # normalization
    X[:, 0] = (X[:, 0] - min_position) / (max_position - min_position)
    X[:, 1] = (X[:, 1] + max_speed) / (max_speed + max_speed)
    Y_pred_all = cluster.forward(X)

    attention = att.forward(X)
    # print(torch.argmax(attention))
    Y_pred_att_argmax = Y_pred_all[torch.argmax(attention)]
    Y_pred_att_argmax = Y_pred_att_argmax.detach().numpy().flatten()
    Y_pred_att_argmax[0] = Y_pred_att_argmax[0] * (max_position - min_position) + min_position
    Y_pred_att_argmax[1] = Y_pred_att_argmax[1] * (max_speed + max_speed) - max_speed

    Y_pred_att_argmax[0] = np.clip(Y_pred_att_argmax[0], min_position, max_position)
    Y_pred_att_argmax[1] = np.clip(Y_pred_att_argmax[1], -max_speed, max_speed)

    # env.set_state(Y_pred_att_argmax)
    env.step(np.argmax(action))

    env.render()
    # recorded_actions = np.append(recorded_actions, [action], axis=0)


# np.savez('action_sequence2.npz', recorded_actions)
env.close()
