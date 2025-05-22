#!/usr/bin/env python3
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from OperantGym import OperantLearning

# --------------------
# 1) Q‐Network (DQN)
# --------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )
    def forward(self, x):
        return self.net(x)

# --------------------
# 2) Replay Buffer
# --------------------
Transition = namedtuple("Transition", ("s", "a", "r", "s2", "done"))
class ReplayBuffer:
    def __init__(self, cap): 
        self.buf, self.cap = deque(maxlen=cap), cap
    def push(self, *args): 
        self.buf.append(Transition(*args))
    def sample(self, n): 
        return random.sample(self.buf, n)
    def __len__(self): 
        return len(self.buf)

# --------------------
# 3) Hyperparameters
# --------------------
pairing          = 'reward'
num_trials       = 50
batch_size       = 32
gamma            = 0.9
lr               = 1e-3
buffer_cap       = 10000
eps_start        = 0.5
eps_end          = 0.01
eps_decay        = 0.995
target_upd_every = 10
pre_steps        = 20    # 1 s @ 100 ms
post_steps       = 30    # 5 s @ 100 ms
max_trial_steps  = pre_steps + post_steps

omission_prob    = 0.1
enl_duration     = (1.0, 3.0)  # seconds
action_cost      = 0.1
enl_penalty      = 0.1

# --------------------
# 4) Setup
# --------------------
env = OperantLearning(
    pairing=pairing,
    omission_prob=omission_prob,
    enl_duration=enl_duration,
    action_cost=action_cost,
    enl_penalty=enl_penalty,
    detection_delay=1,
)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()
memory = ReplayBuffer(buffer_cap)

epsilon = eps_start

# Logs
reward_history = []
loss_history   = []
all_licks      = []
all_tds        = []

# --------------------
# 5) Training Loop
# --------------------
trial = 0
episode = 0

print("Training...")
while trial < num_trials:
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)
    prev_phase = obs[0]
    trial_losses = []

    # Pre-cue circular buffers
    buf_actions = deque([0]*pre_steps, maxlen=pre_steps)
    buf_tds     = deque([0.]*pre_steps, maxlen=pre_steps)
    recording = False
    trial_actions = []
    trial_tds = []
    total_reward = 0.0

    while True:
        # ε-greedy action
        if random.random() < epsilon:
            a = random.randrange(action_dim)
        else:
            with torch.no_grad():
                qvals = policy_net(state.unsqueeze(0))
                a = qvals.argmax().item()

        next_obs, r, _, _, info = env.step(a)
        # print(f"Action: {a}\tReward: {r}\tInfo: {info}")
        next_state = torch.tensor(next_obs, dtype=torch.float32)
        done = info.get("done")
        total_reward += r

        # compute TD error
        with torch.no_grad():
            q_sa = policy_net(state)[a]
            q_next = target_net(next_state).max().item()
            td_err = r + gamma * (0 if done else q_next) - q_sa.item()

        # detect cue onset
        if not recording and next_obs[0] == 1 and prev_phase == 0:
            recording = True
            trial_actions = list(buf_actions)
            trial_tds = list(buf_tds)
        # update pre-cue buffers
        else:
            buf_actions.append(a)
            buf_tds.append(td_err)

    
        if recording:
            trial_actions.append(a)
            trial_tds.append(td_err)

        # store & learn
        memory.push(state, a, r, next_state, done)
        if len(memory) >= batch_size:
            batch = memory.sample(batch_size)
            trans = Transition(*zip(*batch))
            S_b  = torch.stack(trans.s)
            A_b  = torch.tensor(trans.a)
            R_b  = torch.tensor(trans.r, dtype=torch.float32)
            S2_b = torch.stack(trans.s2)
            D_b  = torch.tensor(trans.done, dtype=torch.float32)

            curr_q = policy_net(S_b).gather(1, A_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next = target_net(S2_b).max(1)[0]
                target_q = R_b + gamma * max_next * (1 - D_b)
            loss = loss_fn(curr_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss_history.append(loss.item())
            trial_losses.append(loss.item())

        state = next_state
        prev_phase = next_obs[0]

        if done:
            break

    # end trial: pad to uniform length
    L = len(trial_actions)
    if L < max_trial_steps:
        trial_actions += [0] * (max_trial_steps - L)
        trial_tds    += [0.0] * (max_trial_steps - L)

    all_licks.append(trial_actions[:max_trial_steps])
    all_tds.append(trial_tds[:max_trial_steps])
    reward_history.append(total_reward)
    if trial_losses:
        loss_history.append(np.mean(trial_losses))
    else:
        loss_history.append(0.0)

    trial += 1
    episode += 1
    if episode % target_upd_every == 0:
        target_net.load_state_dict(policy_net.state_dict())
    epsilon = max(eps_end, epsilon * eps_decay)
    print(f"Trial {trial}/{num_trials} | Reward: {total_reward:.2f}")

# --------------------
# 6) Convert & Plot Summary Figure
# --------------------
licks = np.array(all_licks)
tds   = np.array(all_tds)

# time axis from -1s to +2s at 0.1s steps
dt = 0.1
t_axis = np.linspace(-pre_steps*dt, (max_trial_steps-pre_steps)*dt, max_trial_steps)

def plotSEM(x, y, label=None, color=None, ax=None, alpha=0.2):
    """Plot with shaded error margin."""
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    if label is None:
        label = ax._get_lines.get_next_label()
    
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color,
                     edgecolor='None', label='_nolegend_')
    

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# assume: reward_history, loss_history, licks, tds, t_axis, num_trials, plotSEM are defined

fig = plt.figure(figsize=(18, 10))
gs  = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0.2, hspace=0.3)

# top‐left: Reward per trial
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(reward_history)
ax0.set_title("Reward per trial")
ax0.set_xlabel("Trial")
ax0.set_ylabel("Total Reward")

# top‐middle: Loss per trial
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(loss_history)
ax1.set_title("Loss per trial")
ax1.set_xlabel("Trial")
ax1.set_ylabel("MSE Loss")

# bottom‐left: Lick raster (scatter)
ax2 = fig.add_subplot(gs[1, 0])
for i in range(num_trials):
    lick_times = t_axis[licks[i] == 1]
    ax2.scatter(lick_times, np.ones_like(lick_times)*(i+1),
                color='tab:pink', s=20, marker='o', alpha=0.8, edgecolor='none')
# mark cue window
ax2.fill_betweenx([0, num_trials+1], 0, 0.5, color='tab:orange', alpha=0.2, edgecolor='None')
ax2.set_title("Lick raster")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Trial")
ax2.set_xlim(t_axis[0], t_axis[-1])
ax2.set_ylim(0.5, num_trials+1)

# bottom‐middle: Mean + individual TD
ax3 = fig.add_subplot(gs[1, 1])
plotSEM(t_axis, tds, label="TD Error", color='tab:blue', ax=ax3, alpha=0.2)
for i in range(num_trials):
    ax3.plot(t_axis, tds[i], color='tab:blue', alpha=0.1)
# shade cue
ymin, ymax = ax3.get_ylim()
ax3.fill_betweenx([ymin, ymax], 0, 0.5, color='tab:orange', alpha=0.2, edgecolor='None')
ax3.set_ylim(ymin, ymax)
ax3.set_title("TD error vs time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("TD Error")

# the new heatmap, spanning both rows in column 3
ax_heat = fig.add_subplot(gs[:, 2])
im = ax_heat.imshow(
    tds,
    aspect='auto',
    interpolation='nearest',
    extent=[t_axis[0], t_axis[-1], num_trials, 1],
    cmap='coolwarm',
)
ax_heat.set_title("TD Error Heatmap")
ax_heat.set_xlabel("Time (s)")
ax_heat.set_ylabel("Trial")
ax_heat.invert_yaxis()   # so Trial 1 is at top
cbar = fig.colorbar(im, ax=ax_heat, orientation='vertical', label='TD Error')

# Tighten layout and show
fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.07)
plt.show()
