#!/usr/bin/env python3
import os, sys, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer
from TabularPID.Agents.DQN.DQN import EPLHb_DQN, PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from OperantGym import OperantLearning
from plotfunctions import plot_figure
from recorder import SessionRecorder
import matplotlib.pyplot as plt
import random

from trainfuntions import set_global_seeds, setup_model, setup_buffer, train_operant_environment, train_gym_environment, transfer_weights

seed = 12242

# ============================================================================
# OPERANT ENVIRONMENT PARAMETERS (exactly matching EPLHb-Operant.py)
# ============================================================================
operant_session_params = {
    "pairing":          'reward',
    "num_trials":       1000,
    "pre_steps":        10,           # 1 s @ 100 ms
    "post_steps":       40,           # 5 s @ 100 ms
    "enl_duration":     (2.0, 4.0),   # seconds
    "tau_on":           0.01,         # 10 ms
    "tau_off":          0.1,          # 100 ms

    "omission_prob":    0.05,
    "action_cost":      0.1,
    "enl_penalty":      0.2,
    "enl_threshold":    200,          # for accumulated & consecutive ENL licks
    "enl_punish_scale": 0.1,

    "dt":               0.1,          # 100 ms
}

operant_pid_params = {
    "learning_rate": 1e-3,
    "eplhb_lr": 1e-2,
    "coeff_lr": 0.0,
    "initial_eplhb_coeff": -0.3,

    "rnn_type": "GRU",  # Options: "RNN", "GRU", "LSTM"
    "l2_lambda": 0.0,  # L2 regularization strength for EPLHb weights

    "batch_training": False,
    "batch_size": 64 if False else 1,
    "buffer_size": 100_000 if False else 1,
    "tau": 1,
    "gamma": 0.95,
    "gradient_steps": 10,
    "train_freq": 1,
    "target_update_interval": 10,
    
    "initial_eps": 0.1,
    "exploration_fraction": 0.5,
    "minimum_eps": 0.05,
    "learning_starts": 1,
    "inner_size": 64,
    "dump_buffer": False,
    "is_double": False,
    "policy_evaluation": False,
    "seed": seed,

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

# ============================================================================
# GYM ENVIRONMENT PARAMETERS (exactly matching EPLHb-Gym.py)
# ============================================================================
gym_env_params = {
    "env_name": "CliffWalking-v0",  # Change to "CliffWalking-v0" or any Gymnasium env
    "num_episodes": 200,
    "max_steps": 500,
    "warmup_steps": 10000,
    "train_every_n_steps": 100,  # You can adjust this value
    "render_mode": "none",  # Use "human" for rendering, or None for no rendering
}

gym_pid_params = {
    "learning_rate": 1e-3,
    "eplhb_lr": 1e-2,
    "coeff_lr": 0.0,
    "initial_eplhb_coeff": -1.0,
    
    "rnn_type": "GRU",  # Options: "RNN", "GRU", "LSTM"
    "l2_lambda": 0.0,
    
    "buffer_size": 100_000,
    "batch_size": 1,
    "tau": 1,
    "gamma": 0.99,
    "gradient_steps": 10,
    "train_freq": 1,
    "target_update_interval": 10,
    
    "initial_eps": 1,
    "exploration_fraction": 0.8,
    "minimum_eps": 0.05,
    "learning_starts": 1000,
    "inner_size": 64,
    "dump_buffer": False,
    "is_double": False,
    "policy_evaluation": False,
    "seed": seed,

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

# ============================================================================
# TRANSFER LEARNING PARAMETERS
# ============================================================================
transfer_params = {
    "source_env": "operant",  # "operant" or "gym"
    "target_env": "gym",      # "operant" or "gym"
    "fix_source_weights": 10,
}

# ============================================================================
# ENVIRONMENT SELECTION AND SETUP
# ============================================================================
def setup_environment(env_type):
    """Setup environment based on type (operant or gym)"""
    if env_type == "operant":
        env = OperantLearning(
            pairing=operant_session_params["pairing"],
            omission_prob=operant_session_params["omission_prob"],
            enl_duration=operant_session_params["enl_duration"],
            action_cost=operant_session_params["action_cost"],
            enl_penalty=operant_session_params["enl_penalty"],
            reward_decay=True,
            reward_decay_time=1.0,
            print_status=False,
        )
        return env, operant_session_params, operant_pid_params
    elif env_type == "gym":
        env = gym.make(gym_env_params["env_name"], render_mode=gym_env_params["render_mode"])
        return env, gym_env_params, gym_pid_params
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

# ============================================================================
# Weight transfer functions
# ============================================================================
def save_and_plot_results(env_type, env_params, pid_params,
                        recorder=None, reward_history=None, 
                        save=True, plot=True):
    """Save results and generate plots"""
    print(f"\n{'='*60}")
    print("Saving results and plotting")
    print(f"{'='*60}")
    
    if save:
        # Save the recorder and reward data with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save source environment results
        if recorder:
            import pickle
            source_filename = f"{timestamp}-{env_type}_results.pkl"
            # Store both params and recorder
            results = {
                "session_params": env_params,
                "pid_params":     pid_params,
                "recorder":       recorder,
                "seed":           pid_params["seed"],
            }

            # Save everything
            with open(f"PID-results/{timestamp}-{env_type}_results.pkl", "wb") as f:
                pickle.dump(results, f)
            print(f"Saved environment results to {source_filename}")
        
        # Save target environment results
        if reward_history is not None:
            # Save as pickle file
            results = {
                "session_params": env_params,
                "pid_params":     pid_params,
                "recorder":       None,
                "seed":           pid_params["seed"],
            }
            with open(f"PID-results/{timestamp}-{env_type}_results.pkl", "wb") as f:
                pickle.dump(results, f)
            print(f"Saved environment results to {timestamp}-{env_type}_results.pkl")
    
    if plot:
        # Plot source environment results
        if env_type == 'operant' and recorder:
            print("\n--- Plotting results from Operant Task ---")
            plot_figure(recorder, td_error_type='internal', dt=env_params["dt"], show=True,
                    pre_steps=env_params["pre_steps"], post_steps=env_params["post_steps"])
        
        # Plot target environment results
        if reward_history is not None:
            print("\n--- Plotting results from Target Environment ---")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(reward_history, label='Total Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(f'{env_type.title()} Transfer Learning Performance')
            plt.legend()
            plt.grid(True)
            plt.show()

# ============================================================================
# MAIN TRANSFER LEARNING FUNCTION
# ============================================================================
def run_transfer_learning():
    """Main function to run transfer learning between environments"""
    print("=" * 60)
    print("EPLHb Transfer Learning")
    print("=" * 60)
    print(f"Source Environment: {transfer_params['source_env']}")
    print(f"Target Environment: {transfer_params['target_env']}")
    print("=" * 60)
    
    # Setup source environment and model
    print(f"\nSetting up source environment: {transfer_params['source_env']}")
    source_env, source_env_params, source_pid_params = setup_environment(transfer_params['source_env'])
    
    # Setup target environment and model
    print(f"\nSetting up target environment: {transfer_params['target_env']}")
    target_env, target_env_params, target_pid_params = setup_environment(transfer_params['target_env'])
    
    print("\nTransfer learning setup complete!")
    
    # ============================================================================
    # PHASE 1: TRAIN ON SOURCE ENVIRONMENT
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training on {transfer_params['source_env']} environment")
    print(f"{'='*60}")
    
    source_orig_buffer = setup_buffer(source_model, transfer_params['source_env'], source_env)
    if transfer_params['source_env'] == 'operant':
        retrain = True
        while retrain:
            # Set global seed for reproducibility
            new_seed = random.randint(0, 10000)
            set_global_seeds(new_seed)
            source_pid_params["seed"] = new_seed

            source_model, _ = setup_model(source_env, source_pid_params)
            source_recorder, retrain = train_operant_environment(source_model, source_env, source_env_params, source_pid_params, source_orig_buffer)
    else:
        source_recorder = None
        print("Source environment training skipped (not operant)")

    # Plot results from source environment
    save_and_plot_results(transfer_params['source_env'], source_env_params, source_pid_params, recorder=source_recorder, save=False, plot=True)
    
    # ============================================================================
    # PHASE 2: TRANSFER WEIGHTS TO TARGET MODEL
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: Transferring weights to {transfer_params['target_env']} model")
    print(f"{'='*60}")
    
    # Setup target model
    target_model, _ = setup_model(target_env, target_pid_params)

    # Use projection layer for weight transfer to handle different observation space sizes
    transfer_weights(source_model, target_model, use_projection=True)
    
    # ============================================================================
    # PHASE 3: TRAIN ON TARGET ENVIRONMENT
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 3: Training on {transfer_params['target_env']} environment")
    print(f"{'='*60}")
    
    _ = setup_buffer(target_model, transfer_params['target_env'], target_env)    
    if transfer_params['target_env'] == 'gym':
        all_total_rewards = train_gym_environment(
            target_model, target_env, target_env_params, target_pid_params, transfer_params['fix_source_weights']
        )
    else:
        all_total_rewards = None
        print("Target environment training skipped (not gym)")

    # Plot results from target environment
    save_and_plot_results(transfer_params['target_env'], target_env_params, target_pid_params, reward_history=all_total_rewards, save=False, plot=True)
    
    # ============================================================================
    # PHASE 4: SAVE RESULTS AND PLOT
    # ============================================================================
    # Save source environment results
    save_and_plot_results(transfer_params['source_env'], source_env_params, source_pid_params, recorder=source_recorder, save=True, plot=False)
    # Save target environment results
    save_and_plot_results(transfer_params['target_env'], target_env_params, target_pid_params, reward_history=all_total_rewards, save=True, plot=False)
    
    print("\n🎉 Transfer learning complete!")
    print(f"Successfully transferred from {transfer_params['source_env']} to {transfer_params['target_env']}")
    if transfer_params['fix_source_weights'] > 0:
        print(f"Fixed transferred weights for first {transfer_params['fix_source_weights']} episodes")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_transfer_learning()

