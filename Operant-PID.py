#!/usr/bin/env python3
import os, sys, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
# from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer as PIDReplayBuffer
from TabularPID.Agents.DQN.DQN import PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

from OperantGym import OperantLearning
from plotfunctions import plot_figure, get_traces
from recorder import SessionRecorder, TrialLimitCallback


# Replay buffer to do online updates
class OnlineReplayBuffer(PIDReplayBuffer):
    """
    A replay buffer that only ever returns the most recent transition
    (i.e. batch_size = 1, sample = last added element).
    """
    def sample(self, batch_size: int = 1, env=None):
        # ignore batch_size; we always return the single last transition
        # pos points at the *next* insertion index, so last = (pos - 1) mod buffer_size
        last_idx = (self.pos - 1) % self.buffer_size
        # single env => env_index = 0
        batch_inds = np.array([last_idx])
        return self._get_samples(batch_inds, env=env)


# --------------------
# Hyperparameters
# --------------------
pairing          = 'reward'
num_trials       = 200
pre_steps        = 20    # 1 s @ 100 ms
post_steps       = 30    # 5 s @ 100 ms
max_trial_steps  = pre_steps + post_steps

omission_prob    = 0.0
enl_duration     = (2.0, 4.0)  # seconds
action_cost      = 0.05
enl_penalty      = 0.1

tau_on  = 0.01   # 10 ms
tau_off = 0.1    # 100 ms

batch_training = False
batch_size = 64 if batch_training else 1
buffer_size = 100000 if batch_training else 1
replaybuffer = PIDReplayBuffer if batch_training else OnlineReplayBuffer


# PID-DQN parameters
pid_params = {
    "kp": 1.0,                  # proportional gain
    "ki": 0.0,                  # integral gain
    "kd": 0.0,                  # derivative gain
    "alpha": 0.05,              # meta-learning rate for gains
    "beta": 0.95,               # momentum term for meta updates
    "d_tau": 1e-3,              # time constant for D component
    "tabular_d": False,         # use tabular D vs function-approx D

    "learning_rate": 1e-3,      # LR for value network
    "replay_memory_size": buffer_size,
    "batch_size": batch_size,
    "tau": 1e-3,                # Polyak update coefficient
    "gamma": 0.9,              # discount factor
    "gradient_steps": 0,
    "train_freq": int(1e9),
    "target_update_interval": 1000,

    'meta_lr': 1e-3,           # meta-learning rate for gains
    'epsilon_gain': 0.1,          # exploration rate for gains

    "initial_eps": 0.5,
    "exploration_fraction": 0.1,
    "minimum_eps": 0.05,
    "learning_starts": 1000,

    "inner_size": 64,           # hidden layer size
    "dump_buffer": False,
    "is_double": False,
    "policy_evaluation": False,
    "seed": 42,
}

# --------------------
# Setup
# --------------------
env = OperantLearning(
    pairing=pairing,
    omission_prob=omission_prob,
    enl_duration=enl_duration,
    action_cost=action_cost,
    enl_penalty=enl_penalty,
    detection_delay=1,
)
# env = Monitor(env)  

# Gain adapter
gain_adapter = SingleGainAdapter(
    kp=pid_params["kp"],
    ki=pid_params["ki"],
    kd=pid_params["kd"],
    alpha=pid_params["alpha"],
    beta=pid_params["beta"],
    meta_lr=pid_params["meta_lr"],
    epsilon=pid_params["epsilon_gain"],
)

# Define policy kwargs (network architecture + optimizer)
policy_kwargs = dict(
    net_arch=[pid_params["inner_size"], pid_params["inner_size"]],
    optimizer_class=optim.Adam,
    with_RNN_layer = True,
)

# Prevent CUDA from being used (patch)
import TabularPID.Agents.DQN.DQN_policy as _dp
# find whatever class has jump_start_cuda and override it
for _name in dir(_dp):
    cls = getattr(_dp, _name)
    if isinstance(cls, type) and hasattr(cls, "jump_start_cuda"):
        cls.jump_start_cuda = lambda self: None


# Set up the model
model = PID_DQN(
    # PID-specific arguments
    pid_params['d_tau'],
    pid_params['tabular_d'],
    gain_adapter,

    # standard SB3/DQN args
    policy="MlpPolicy",
    env=env,
    learning_rate=pid_params['learning_rate'],
    buffer_size=pid_params['replay_memory_size'],
    batch_size=pid_params['batch_size'],
    tau=pid_params['tau'],
    gamma=pid_params['gamma'],
    gradient_steps=pid_params['gradient_steps'],
    train_freq=pid_params['train_freq'],
    target_update_interval=pid_params['target_update_interval'],
    exploration_fraction=pid_params['exploration_fraction'],
    exploration_initial_eps=pid_params['initial_eps'],
    exploration_final_eps=pid_params['minimum_eps'],
    optimize_memory_usage=False,
    learning_starts=pid_params['learning_starts'],
    tensorboard_log=None,
    policy_kwargs=policy_kwargs,
    seed=pid_params['seed'],
    device="cpu",
    dump_buffer=pid_params['dump_buffer'],
    is_double=pid_params['is_double'],
    optimal_model=None,
    policy_evaluation=pid_params['policy_evaluation'],

    # Use authors replay buffer
    replay_buffer_class=replaybuffer,
)
orig_buffer = model.replay_buffer # save the original big replay buffer

# Link adapter to model
gain_adapter.set_model(model)

# Set up logging
new_logger = configure(None, ["stdout"])
model.set_logger(new_logger)
recorder = SessionRecorder()

# epsilon decay params
eps_start    = pid_params["initial_eps"]
eps_end      = pid_params["minimum_eps"]
decay_trials = int(pid_params["exploration_fraction"] * num_trials)

# --------------------
# Training Loop
# --------------------

obs, _ = env.reset()
trial_idx = 0

while trial_idx < num_trials:
    print(f"Trial {trial_idx + 1}/{num_trials}")
    # compute trial-based epsilon
    frac = min(1.0, trial_idx / max(1, decay_trials))
    eps  = eps_start + frac * (eps_end - eps_start)
    model.exploration_rate = eps
    model.logger.record("rollout/exploration_rate", eps)
    print(f"Trial {trial_idx+1}/{num_trials}, ε={eps:.3f}")

    # 1) roll out one trial
    trial_transitions = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, _, _, info = env.step(action)
        done = info["done"]
        outcome = info["outcome"]
        # print(f"Reward: {reward}, Info: {info}")
        # store transition
        trial_transitions.append((obs, action, next_obs, reward, done, info))
        obs = next_obs
        # record every timestep in the session trace
        recorder.record_env_step(action, reward, next_obs, info, model=model)

    # print(info) # Print trial summary
    if outcome != "enl_break": trial_idx += 1 # update trial index

    # 2) build a tiny buffer for exactly this trial
    N = len(trial_transitions)
    trial_buf = PIDReplayBuffer(
        buffer_size=N,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )
    # fill it
    for (s, a, s2, r, done, info) in trial_transitions:
        trial_buf.add(obs=s, next_obs=s2, 
                      action=np.array([a]), reward=np.array([r]), done=np.array([done]), infos=[info])

    # 3) swap in, do a *single* batch-update on the whole trial
    model.replay_buffer = trial_buf

    # 4) do a single training step
    model.train(batch_size=N, gradient_steps=1)
    # record the *actual* PID-DQN update signal for this trial
    recorder.record_train(model)

    # 5) restore original buffer so you keep accumulating long-term experience
    model.replay_buffer = orig_buffer

# --------------------
# Plot Summary Figure
# --------------------

td_errors = np.array(recorder.td_errors)
licks     = np.array(recorder.licks)
tones     = np.array(recorder.tones)

rewards = np.array(recorder.rewards)
losses  = np.array(recorder.losses)
dones   = np.array(recorder.dones)

# print(recorder.p.shape)
p_history = np.array(recorder.p)
d_history = np.array(recorder.d)
i_history = np.array(recorder.i)
kp_history = np.array(recorder.kp)
ki_history = np.array(recorder.ki)
kd_history = np.array(recorder.kd)
update_history = kp_history * p_history + ki_history * i_history + kd_history * d_history

# Align licks and TD errors to the cue
error = td_errors
cue_licks = get_traces(licks, tones, pre_steps, post_steps)
cue_error   = get_traces(error, tones, pre_steps, post_steps)

# Get reward and loss history
trial_ends = np.where(dones)[0]
trial_starts = np.concatenate(([0], trial_ends[:-1] + 1))
reward_history = [rewards[s : e + 1].sum() for s, e in zip(trial_starts, trial_ends)]
loss_history = [losses[s : e + 1].mean() for s, e in zip(trial_starts, trial_ends)]

plot_figure(cue_licks, cue_error, reward_history, loss_history,
            dt=0.1,
            pre_steps=pre_steps, post_steps=post_steps,
            tau_on=tau_on, tau_off=tau_off)
