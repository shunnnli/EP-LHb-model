#!/usr/bin/env python3
import os, sys, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
# from TabularPID.AgentBuilders.DQNBuilder import build_PID_DQN # not working for me
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer
from TabularPID.Agents.DQN.DQN import EPLHb_DQN, PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

import numpy as np
import gymnasium as gym
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from recorder import SessionRecorder

# --------------------
# Configurable parameters
# --------------------
ENV_NAME = "CliffWalking-v0"  # Change to "CliffWalking-v0" or any Gymnasium env
NUM_EPISODES = 200
MAX_STEPS = 500
WARMUP_STEPS = 10000
TRAIN_EVERY_N_STEPS = 1000  # You can adjust this value
render_mode = "none"  # Use "human" for rendering, or None for no rendering

# PID-DQN parameters (customize as needed)
pid_params = {
    "learning_rate": 1e-3,
    "eplhb_lr": 1e-2,
    "coeff_lr": 0.0,
    "initial_eplhb_coeff": -0.00,
    "rnn_type": "GRU",  # Options: "RNN", "GRU", "LSTM"
    "l2_lambda": 1e-6,
    "buffer_size": 100_000,
    "batch_size": 32,
    "tau": 1,
    "gamma": 0.95,
    "gradient_steps": 1,
    "train_freq": 1,
    "target_update_interval": 10,
    "initial_eps": 1,
    "exploration_fraction": 0.8,
    "minimum_eps": 0.05,
    "learning_starts": 10,
    "inner_size": 64,
    "dump_buffer": False,
    "is_double": False,
    "policy_evaluation": False,
    "seed": 42,
    "kp": 1.0,
    "ki": 0.0,
    "kd": 0.0,
    "meta_lr": 0,
    "epsilon_gain": 0.1,
    "alpha": 0.05,
    "beta": 0.95,
    "d_tau": 1,
    "tabular_d": False,
}

# Set up environment
env = gym.make(ENV_NAME, render_mode=render_mode)  # Use "human" for rendering
replaybuffer = OnlineReplayBuffer

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

# Policy kwargs
policy_kwargs = dict(
    net_arch=[pid_params["inner_size"], pid_params["inner_size"]],
    optimizer_class=optim.Adam,
    with_RNN_layer=True,
    rnn_type=pid_params["rnn_type"],
    features_extractor_kwargs=dict(
        initial_eplhb_coeff=pid_params["initial_eplhb_coeff"],
    ),
)

optimizer_kwargs = dict(
    eplhb_lr=pid_params["eplhb_lr"],
    coeff_lr=pid_params["coeff_lr"],
)

# Prevent CUDA from being used (patch)
import TabularPID.Agents.DQN.DQN_policy as _dp
# find whatever class has jump_start_cuda and override it
for _name in dir(_dp):
    cls = getattr(_dp, _name)
    if isinstance(cls, type) and hasattr(cls, "jump_start_cuda"):
        cls.jump_start_cuda = lambda self: None

# Set up the model
model = EPLHb_DQN(
    pid_params['d_tau'],
    pid_params['tabular_d'],
    gain_adapter,
    policy="MlpPolicy",
    env=env,
    learning_rate=pid_params['learning_rate'],
    buffer_size=pid_params['buffer_size'],
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
    optimizer_kwargs=optimizer_kwargs,
    seed=pid_params['seed'],
    device="cpu",
    dump_buffer=pid_params['dump_buffer'],
    is_double=pid_params['is_double'],
    optimal_model=None,
    policy_evaluation=pid_params['policy_evaluation'],

    # Use authors replay buffer
    replay_buffer_class=replaybuffer,
)
model.l2_lambda = pid_params["l2_lambda"]

# Link adapter to model
gain_adapter.set_model(model)

# Set up logging
model.set_logger(configure(None, []))
# model.set_logger(configure(None, ["stdout"]))
recorder = SessionRecorder()

# --------------------
# Warm-up phase: fill replay buffer with random actions
# --------------------
print("Warm up: fill the replay buffer with random actions...")
# Create a separate environment for warm-up with no rendering
warmup_env = gym.make(ENV_NAME, render_mode=None)
obs, _ = warmup_env.reset()
for _ in tqdm(range(WARMUP_STEPS), desc="Warm-up steps"):
    action = warmup_env.action_space.sample()
    next_obs, reward, terminated, truncated, info = warmup_env.step(int(action))
    done = terminated or truncated
    def to_obs_array(o):
        return o if isinstance(o, np.ndarray) else np.array([o])
    model.replay_buffer.add(
        obs=to_obs_array(obs),
        next_obs=to_obs_array(next_obs),
        action=np.array([action]),
        reward=np.array([reward], dtype=np.float32),
        done=done,
        infos=[info],
        d=np.array([0.0], dtype=np.float32),  # Not used in standard gym
        z=np.array([0.0], dtype=np.float32),  # Not used in standard gym
    )
    obs = next_obs
    if done:
        obs, _ = warmup_env.reset()
# Close warm-up environment
warmup_env.close()


# Training loop
print(f"Train: train the network every {TRAIN_EVERY_N_STEPS} steps...")
eps = pid_params["initial_eps"]
eps_start   = pid_params["initial_eps"]
eps_end     = pid_params["minimum_eps"]
max_num_iters = NUM_EPISODES
decay_trials = int(pid_params["exploration_fraction"] * max_num_iters)

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    # For per-step reward tracking
    step_rewards = []

    while not done and step < MAX_STEPS:
        # Set exploration rate
        model.exploration_rate = eps
        model.logger.record("rollout/exploration_rate", eps)

        action, _ = model.predict(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_reward += reward
        step_rewards.append(reward)

        # update gains and sync networks
        model._on_step()

        # Add to replay buffer
        def to_obs_array(o):
            return o if isinstance(o, np.ndarray) else np.array([o])
        model.replay_buffer.add(
            obs=to_obs_array(obs),
            next_obs=to_obs_array(next_obs),
            action=np.array([action]),
            reward=np.array([reward], dtype=np.float32),
            done=done,
            infos=[info],
            d=np.array([0.0], dtype=np.float32),  # Not used in standard gym
            z=np.array([0.0], dtype=np.float32),  # Not used in standard gym
        )
        # Record step
        recorder.record_env_step(episode, action, reward, next_obs, info, model=model)
        obs = next_obs
        step += 1

        # Train every n steps if buffer is sufficiently filled
        if step % TRAIN_EVERY_N_STEPS == 0 and model.replay_buffer.size() > pid_params["learning_starts"]:
            model.train(batch_size=pid_params["batch_size"], seq_len=step, gradient_steps=pid_params["gradient_steps"])

    # Store per-episode and per-step rewards for plotting
    if episode == 0:
        all_total_rewards = []
    all_total_rewards.append(total_reward)

     # compute step-based epsilon
    frac = min(1.0, episode / max(1, decay_trials))
    eps  = eps_start + frac * (eps_end - eps_start)
    
    print(f"Episode {episode+1}/{NUM_EPISODES}: Total Reward: {total_reward}, eps: {eps:.2f}")

# --------------------
# Plot summary metrics after training
# --------------------
import matplotlib.pyplot as plt

# Plot total rewards per episode
plt.figure(figsize=(12, 6))
plt.plot(all_total_rewards, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()
plt.show()

