#!/usr/bin/env python3
import os, sys, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
# from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN # not working for me
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer
from TabularPID.Agents.DQN.DQN import PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

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
from recorder import SessionRecorder
from types import SimpleNamespace
from stable_baselines3.common.utils import zip_strict

import pickle

class OnlineReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: torch.device,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        # parallel arrays for d & z
        self.ds = np.zeros((buffer_size, 1), dtype=np.float32)
        self.zs = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: bool,
        infos: list,
        d: np.ndarray,
        z: np.ndarray,
    ) -> None:
        # write obs/next_obs/etc.
        super().add(obs, next_obs, action, reward, done, infos)
        idx = (self.pos - 1) % self.buffer_size
        self.ds[idx, 0] = float(np.asarray(d).flatten()[0])
        self.zs[idx, 0] = float(np.asarray(z).flatten()[0])

    def sample(
        self,
        batch_size: int,
        env=None,
        seq_len: int = None
    ):
        if seq_len is None:
            # Generate indices manually
            idxs = np.random.randint(0, self.size(), size=batch_size)
            base_batch = self._get_samples(idxs, env=env)
        else:
            # truncated BPTT style: take the first `seq_len` of your buffer
            n    = self.size()
            L    = min(seq_len, n)
            idxs = np.arange(L, dtype=int)
            base_batch = self._get_samples(idxs, env=env)

        # Slice d & z using the sampled indices
        batch_ds = torch.as_tensor(self.ds[idxs], device=self.device)
        batch_zs = torch.as_tensor(self.zs[idxs], device=self.device)

        # Turn the base namedtuple into a dict
        data = { field: getattr(base_batch, field) for field in base_batch._fields }
        # Inject your custom tensors
        data['ds'] = batch_ds
        data['zs'] = batch_zs

        data['indices'] = idxs

        # If using sequence mode, reshape accordingly
        if seq_len is not None:
            for field in ('observations','next_observations'):
                arr = data[field]
                if arr.ndim == 2:
                    data[field] = arr[np.newaxis, ...]
            for field in ('actions','rewards','dones','ds','zs'):
                arr = data[field]
                if arr.ndim == 1:
                    data[field] = arr[np.newaxis, :]
                elif arr.ndim == 2:
                    data[field] = arr[np.newaxis, ...]
        return SimpleNamespace(**data)
    def update(self, indices, zs=None, ds=None):
        if zs is not None:
            self.zs[indices] = zs.cpu().numpy()
        if ds is not None:
            self.ds[indices] = ds.cpu().numpy()


# --------------------
# Hyperparameters
# --------------------
pairing          = 'reward'
num_trials       = 200
pre_steps        = 10    # 1 s @ 100 ms
post_steps       = 40    # 5 s @ 100 ms
max_trial_steps  = pre_steps + post_steps

omission_prob    = 0.05
enl_duration     = (2.0, 4.0)  # seconds
action_cost      = 0.05
enl_penalty      = 0.05

tau_on  = 0.01   # 10 ms
tau_off = 0.1    # 100 ms

gradient_steps = 10  # how many gradient steps to do per trial
gamma = 0.95  # discount factor for the DQN
n_step_td = 20  # n-step TD learning

batch_training = False
batch_size = 64 if batch_training else 1
buffer_size = 100000 if batch_training else 1
replaybuffer = OnlineReplayBuffer



# PID-DQN parameters
pid_params = {
    "kp": 1.0,                  # proportional gain
    "ki": 0.0,                  # integral gain
    "kd": 1.0,                  # derivative gain
    "alpha": 0.05,              # meta-learning rate for gains
    "beta": 0.95,               # momentum term for meta updates
    "d_tau": 1e-3,              # time constant for D component
    "tabular_d": False,         # use tabular D vs function-approx D

    "learning_rate": 1e-3,      # LR for value network
    "replay_memory_size": buffer_size,
    "batch_size": batch_size,
    "tau": 0.5,                   # Polyak update coefficient
    "gamma": gamma,              # discount factor
    "gradient_steps": 1,
    "train_freq": 1,
    "target_update_interval": 10,

    'meta_lr': 1e-3,           # meta-learning rate for gains
    'epsilon_gain': 0.1,          # exploration rate for gains

    "initial_eps": 0.7,
    "exploration_fraction": 0.07,
    
    "minimum_eps": 0.07,
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
    with_RNN_layer=True,
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
# max_num_iters = 40000
max_num_iters = 20000

decay_trials = int(pid_params["exploration_fraction"] * max_num_iters)

# Set up buffer
model.replay_buffer = OnlineReplayBuffer(
    buffer_size=10_000, # hold the last 10 000 steps (ie 1000 seconds)
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=model.device,
    optimize_memory_usage=False,
    handle_timeout_termination=True,
)

# --------------------
# Training Loop
# --------------------

obs, _ = env.reset()
trial_idx = 0
iter_count = 1

while trial_idx < num_trials:
    print(f"Trial {trial_idx + 1}/{num_trials}")
    if trial_idx > 50: # make omission prob higher after 50 trials
        env.omission_prob = 0.2

    # reset the LSTM here so it doesn’t leak from the last trial
    z_prev = 0.0
    model.policy.q_net.reset_hidden(batch_size=batch_size)

    # 1) roll out one trial
    trial_transitions = []
    done = False

    while not done:
        # compute step-based epsilon
        frac = min(1.0, iter_count / max(1, decay_trials))
        eps  = eps_start + frac * (eps_end - eps_start)
        model.exploration_rate = eps
        model.logger.record("rollout/exploration_rate", eps)
        print(f"Trial {trial_idx+1}/{num_trials}, ε={eps:.3f}")

        # act
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, _, _, info = env.step(action)
        done = info["done"]
        outcome = info["outcome"]

        # hard sync
        if iter_count % model.target_update_interval == 0:
            if iter_count > model.learning_starts:
                update_size = min(50000, model.replay_buffer.size())
                replay_data = model.replay_buffer.sample(update_size, env=model._vec_normalize_env)  # type: ignore[union-attr]
                model.gain_adapter.adapt_gains(replay_data)
                print("kd gain:", model.kd)

            model.policy.d_net.load_state_dict(model.policy.q_net_target.state_dict())
            model.policy.q_net_target.load_state_dict(model.policy.q_net.state_dict())
        iter_count += 1

        

        # self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        # if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            # Update the gains here once we have started training
            # if self._n_calls > self.learning_starts:
            #     #self.gain_adapter.apply_weight_decay(self.replay_buffer)
            #     update_size = 50000
            #     update_size = min(update_size, self.replay_buffer.size())

            #     replay_data = self.replay_buffer.sample(update_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                
            #     self.gain_adapter.adapt_gains(replay_data)
            # update d net
            # self.d_net.load_state_dict(self.q_net_target.state_dict())
            # Update the target network
            # self.q_net_target.load_state_dict(self.q_net.state_dict())

        # store transition
        trial_transitions.append((obs, action, next_obs, reward, done, info))
        
        # calculate d and z updates
        with torch.no_grad():
            # make observation tensor
            obs_t = torch.tensor(obs, device=model.device, dtype=torch.float32).unsqueeze(0) # [1, obs_dim]
            next_t = torch.tensor(next_obs, device=model.device, dtype=torch.float32).unsqueeze(0) # [1, obs_dim]
            
            # get the D update
            if model.tabular_d:
                d_update = model.gain_adapter.get_d_update(obs_t, next_t)
            else:
                d_out = model.d_net(obs_t) # [1, n_actions]
                d_update = d_out[0, action].item()  # get the D update for the action taken
            # get your PID gains α, β (and kp,ki,kd if you want)
            action_scalar = int(action)  
            a_t = torch.tensor([[action_scalar]], dtype=torch.long, device=model.device)
            kp, ki, kd, alpha, beta = model.gain_adapter.get_gains(obs_t, a_t, None)
            
            # d) Q-values for TD‐error
            q_curr = model.policy.q_net(obs_t)[0, action].item()
            q_next = model.policy.q_net_target(next_t).max(dim=1)[0].item()
            pd_target = q_curr + d_update
            target = q_curr
            
            td_err = reward + (0.0 if done else model.gamma * q_next) - target  # BRₜ
            td_error_PD = reward + (0.0 if done else model.gamma * q_next) - pd_target  # BRₜ without D
            print("target no d:", target)
            print("td err no d:", td_err)

            print("d update:", d_update)
            print("pd target with d:", pd_target)
            
            print("td_error_PD:", td_error_PD)
            # print("Q current:", q_curr, "Q next:", q_next, "TD Error:", td_err, "Reward:", reward, "Done:", done, "Action:", action_scalar)

            # e) integrator update
            z_update = beta * z_prev + alpha * td_err

            q_cue    = model.policy.q_net(obs_t)[0, action_scalar].item()
            q_reward = model.policy.q_net(next_t)[0, action_scalar].item()
        
        # add to the replay buffer
        model.replay_buffer.add(obs=obs, next_obs=next_obs,
                                action=np.array([action]),
                                reward=np.array([reward], dtype=np.float32),
                                done=done, infos=[info],
                                d=np.array([d_update], dtype=np.float32), 
                                z=np.array([z_update], dtype=np.float32),
                                )
        z_prev = z_update
        # record every timestep in the session trace
        recorder.record_env_step(action, reward, next_obs, info, model=model)
        # update obs
        obs = next_obs

    if outcome != "enl_break": trial_idx += 1 # update trial index

    # 2) build a tiny buffer for exactly this trial
    N = len(trial_transitions)

    # 4) do a single training step
    model.train(batch_size=batch_size, seq_len=N, gradient_steps=gradient_steps)
    # record the *actual* PID-DQN update signal for this trial
    recorder.record_train(model)

    # 5) restore original buffer so you keep accumulating long-term experience
    model.replay_buffer = orig_buffer

# --------------------
# Plot Summary Figure
# --------------------

plot_figure(recorder,
            dt=0.1, pre_steps=pre_steps, post_steps=post_steps,
            tau_on=tau_on, tau_off=tau_off)
