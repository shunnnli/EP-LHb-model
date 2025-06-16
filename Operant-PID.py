#!/usr/bin/env python3
import os, sys, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
# from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN # not working for me
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer
from TabularPID.Agents.DQN.DQN import PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

import numpy as np
import gymnasium as gym
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from OperantGym import OperantLearning
from plotfunctions import plot_figure
from recorder import SessionRecorder

# --------------------
# Hyperparameters
# --------------------
pairing          = 'reward'
num_trials       = 200
pre_steps        = 10           # 1 s @ 100 ms
post_steps       = 40           # 5 s @ 100 ms
max_trial_steps  = pre_steps + post_steps

omission_prob    = 0.1
enl_duration     = (2.0, 4.0)  # seconds
action_cost      = 0.05
enl_penalty      = 0.1         # for individual licks during ENL

enl_threshold   = 200          # for accumulated and consecutive ENL licks
enl_punish_scale = 1.5         # scale for ENL punish

tau_on  = 0.01                 # 10 ms
tau_off = 0.1                  # 100 ms

gradient_steps = 10            # how many gradient steps to do per trial
gamma = 0.95                   # discount factor for the DQN
n_step_td = 20                 # n-step TD learning

batch_training = False
batch_size = 64 if batch_training else 1
buffer_size = 100000 if batch_training else 1
replaybuffer = OnlineReplayBuffer


# PID-DQN parameters
pid_params = {
    "kp": 1.0,                  # proportional gain
    "ki": 0.0,                  # integral gain
    "kd": 0.0,                  # derivative gain
    'meta_lr': 0,               # meta-learning rate for gains
    'epsilon_gain': 0.1,        # exploration rate for gains
    "alpha": 0.05,              # i update coefficient
    "beta": 0.95,               # i update coefficient
    "d_tau": 1,                 # time constant for D component
    "tabular_d": False,         # use tabular D vs function-approx D

    "learning_rate": 1e-3,      # LR for value network
    "replay_memory_size": buffer_size,
    "batch_size": batch_size,
    "tau": 1,                   # Polyak update coefficient
    "gamma": gamma,              # discount factor
    "gradient_steps": 1,
    "train_freq": 1,
    "target_update_interval": 10,

    "initial_eps": 0.1,
    "exploration_fraction": 0.01, # smaller fraction for fast decay
    
    "minimum_eps": 0.05,
    "learning_starts": 1000,

    "inner_size": 64,           # hidden layer size
    "dump_buffer": False,
    "is_double": False,
    "policy_evaluation": False,
    "seed": 26,
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
eps_start   = pid_params["initial_eps"]
eps_end     = pid_params["minimum_eps"]
max_num_iters = 40000
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
eps = eps_start  # start with high exploration rate
enl_count = 0

while trial_idx < num_trials:
    print(f"Trial {trial_idx+1}/{num_trials}, ε={eps:.3f}")

    # reset the network here so it doesn’t leak from the last trial
    z_prev = 0.0
    model.policy.q_net.reset_hidden(batch_size=batch_size)

    # 1) roll out one trial
    trial_timesteps = 0
    done = False
    

    while not done:
        # Set exploration rate
        model.exploration_rate = eps
        model.logger.record("rollout/exploration_rate", eps)

        # act
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, _, _, info = env.step(action)
        done = info["done"]
        outcome = info["outcome"]

        # punish if stuck in ENL for > 200 steps
        enl_count = enl_count + 1 if outcome and "enl" in outcome else 0
        reward -= max(enl_count - enl_threshold, 0) * enl_punish_scale

        # update gains and sync networks
        model._on_step()
        trial_timesteps += 1
        
        # calculate d and z updates for replay buffer
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
            # print("Gains:", kp.item(), ki.item(), kd.item())
            
            # d) Q-values for TD‐error
            q_curr = model.policy.q_net(obs_t)[0, action].item()
            q_next = model.policy.q_net_target(next_t).max(dim=1)[0].item()
            td_err = reward + (0.0 if done else model.gamma * q_next) - q_curr  # BRₜ
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
        recorder.record_env_step(trial_idx, action, reward, next_obs, info, model=model)
        # update obs
        obs = next_obs

    if outcome != "enl_break": 
        trial_idx += 1 # update trial index
        # compute step-based epsilon
        frac = min(1.0, trial_idx / max(1, decay_trials))
        eps  = eps_start + frac * (eps_end - eps_start)

    # 4) do a single training step
    model.train(batch_size=batch_size, seq_len=trial_timesteps, gradient_steps=gradient_steps)
    # record the *actual* PID-DQN update signal for this trial
    # recorder.record_train(model)

    # 5) restore original buffer so you keep accumulating long-term experience
    model.replay_buffer = orig_buffer


# --------------------
# Plot Summary Figure
# --------------------
plot_figure(recorder,
            dt=0.1, pre_steps=pre_steps, post_steps=post_steps,
            tau_on=tau_on, tau_off=tau_off)
