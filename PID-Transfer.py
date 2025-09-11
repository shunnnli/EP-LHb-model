# to-do
# implement matrix functionality to test kd, omit, max_b, num_recent, etc
# implement saving/plot functionality



#!/usr/bin/env python3
import os, sys, copy, pickle, itertools, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer, ExtendedReplayBuffer
from TabularPID.Agents.DQN.DQN import EPLHb_DQN, PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

import numpy as np
import pandas as pd
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
import datetime

from trainfuntions import set_global_seeds, setup_model, setup_buffer, train_PID_operant_environment, train_gym_environment, transfer_weights

seed = 12242
# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
experiment_params = {
    # Define sweep grid
    "kd_values":        [0, 0.1],  # PID derivative gain values
    "omission_probs":   [0, 0.1],
    "repeats":          1,  # Number of repeats for each combination

    "max_batch_sizes":  [1, 5],  # Different batch sizes to test
    "num_recents":      [1, 5],   # Different num_recent values to test
}

# ============================================================================
# OPERANT ENVIRONMENT PARAMETERS (exactly matching PID-Operant.py)
# ============================================================================
operant_session_params = {
    "pairing":          'reward',
    "num_trials":       10,
    "pre_steps":        10,
    "post_steps":       40,
    "enl_duration":     (2.0, 4.0),
    "tau_on":           0.01,
    "tau_off":          0.1,
    "omission_prob":    0.2,
    "action_cost":      0.1,
    "enl_penalty":      0.2,
    "enl_threshold":    200,
    "enl_punish_scale": 0.5,
    "gradient_steps":   10,
    "gamma":            0.95,
    "batch_training":   False,
    "batch_size":       1, 
    "max_batch_size":   5,   # max replay buffer space
    "num_recent":       1,   # number of consecutive recent trials to fill replay buffer. ex. 5 num_recent, means 5 random old trials in size 10 replay buffer
    "buffer_size":      1,
    "dt":               0.1,
    "continual_learning": True,
    "change_start":     200,
    "change_interval":  50,
}

operant_pid_params = {
    "kp":                   1.0,
    "ki":                   0.0,
    "kd":                   0,
    "meta_lr":              0,
    "meta_lr_d":            0,
    "epsilon_gain":         0.1,
    "alpha":                0.05,
    "beta":                 0.95,
    "d_tau":                1,
    "tabular_d":            False,
    "learning_rate":        1e-3,
    "replay_memory_size":   operant_session_params["buffer_size"],
    "batch_size":           operant_session_params["batch_size"],
    "tau":                  1,
    "gamma":                operant_session_params["gamma"],
    "gradient_steps":       1,
    "train_freq":           1,
    "target_update_interval": 10,
    "initial_eps":          0.1,
    "exploration_fraction": 0.001,
    "minimum_eps":          0.05,
    "learning_starts":      1000,
    "inner_size":           64,
    "dump_buffer":          False,
    "is_double":            False,
    "policy_evaluation":    False,
    "seed":                 26,
    "rnn_type": "GRU",  # Options: "RNN", "GRU", "LSTM". Change as needed.
    "l2_lambda": 1e-6,  # L2 regularization strength for EPLHb weights
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
                        recorder=None, stuck_counts=None, reward_history=None, 
                        save=True, plot=True):
    """Save results and generate plots"""
    print(f"\n{'='*60}")
    print("Saving results and plotting")
    print(f"{'='*60}")
    
    if save:
        # Save the recorder and reward data with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save source environment results
        if recorder:
            source_filename = f"{timestamp}-{env_type}_results.pkl"
            # Store both params and recorder
            results[(kd, omit, max_b, num_r, r)] = {
                "session_params": env_params,
                "pid_params":     env_params,
                "recorder":       recorder,
                "seed":           pid_params["seed"],
                "stuck_counts":   stuck_counts
            }

            # Save everything
            with open(f"PID-results/{timestamp}-{env_type}_results.pkl", "wb") as f:
                pickle.dump(results, f)
            print(f"Saved environment results to {source_filename}")
        
        # Save target environment results
        if reward_history is not None:
            # Save as pickle file
            results[(kd, omit, max_b, num_r, r)] = {
                "session_params": env_params,
                "pid_params":     env_params,
                "recorder":       None,
                "seed":           pid_params["seed"],
                "stuck_counts":   stuck_counts
            }
            with open(f"PID-results/{timestamp}-{env_type}_results.pkl", "wb") as f:
                pickle.dump(results, f)
            print(f"Saved environment results to {timestamp}-{env_type}_results.pkl")

            # Save everything
            result_file = f"results_Kd_{kd}_omit_{omit}_maxB_{max_b}_numR_{num_r}.pkl"
            with open(os.path.join(save_dir, result_file), "wb") as f:
                pickle.dump(results, f)
    
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
    
    source_model, _ = setup_model(source_env, source_pid_params, model_type="PID")
    source_orig_buffer = setup_buffer(source_model, transfer_params['source_env'], source_env)

    if transfer_params['source_env'] == 'operant':
        for r in range(experiment_params["repeats"]):
            stuck_counts = 0
            retrain = True
            while retrain:
                # Set global seed for reproducibility
                new_seed = random.randint(0, 10000)
                set_global_seeds(new_seed)
                source_pid_params["seed"] = new_seed

                # Train once with the current parameters
                print(f"Training with kd={kd}, omit={omit}, "
                        f"max_batch={max_b}, num_recent={num_r}, "
                        f"(repeat {r + 1}/{repeats})")

                source_model, _ = setup_model(source_env, source_pid_params, model_type="PID")
                source_recorder, retrain, got_stuck = train_PID_operant_environment(source_model, source_env, source_env_params, source_pid_params)
                if got_stuck:
                    stuck_counts += 1
            
            # Plot and save summary figure
            save_name = f"kd_{kd}_omit_{omit}_maxB_{max_b}_numR_{num_r}_seed_{new_seed}.png"
            plot_figure(source_recorder, dt=source_env_params["dt"], pre_steps=source_env_params["pre_steps"], post_steps=source_env_params["post_steps"],
                        save=True, save_path=os.path.join(save_dir, save_name))

            # Store both params and recorder
            results[(kd, omit, max_b, num_r, r)] = {
                "session_params": source_env_params,
                "pid_params":     source_pid_params,
                "recorder":       source_recorder,
                "seed":           source_pid_params["seed"],
                "stuck_counts":   stuck_counts
            }

        # Save everything
        result_file = f"results_Kd_{kd}_omit_{omit}_maxB_{max_b}_numR_{num_r}.pkl"
        with open(os.path.join(save_dir, result_file), "wb") as f:
            pickle.dump(results, f)
            

            
        # Plot results from source environment
        # save_and_plot_results(transfer_params['source_env'], source_env_params, source_pid_params, recorder=source_recorder, stuck_counts=stuck_counts, save=True, plot=False)

    else:
        source_recorder = None
        print("Source environment training skipped (not operant)")

    
    
    

    # # ============================================================================
    # # PHASE 2: TRANSFER WEIGHTS TO TARGET MODEL
    # # ============================================================================
    # print(f"\n{'='*60}")
    # print(f"PHASE 2: Transferring weights to {transfer_params['target_env']} model")
    # print(f"{'='*60}")
    
    # # Setup target model
    # target_model, _ = setup_model(target_env, target_pid_params, model_type="PID")

    # # Use projection layer for weight transfer to handle different observation space sizes
    # transfer_weights(source_model, target_model, use_projection=True)
    
    # # ============================================================================
    # # PHASE 3: TRAIN ON TARGET ENVIRONMENT
    # # ============================================================================
    # print(f"\n{'='*60}")
    # print(f"PHASE 3: Training on {transfer_params['target_env']} environment")
    # print(f"{'='*60}")
    
    # _ = setup_buffer(target_model, transfer_params['target_env'], target_env)    
    # if transfer_params['target_env'] == 'gym':
    #     all_total_rewards = train_gym_environment(
    #         target_model, target_env, target_env_params, target_pid_params, transfer_params['fix_source_weights']
    #     )
    # else:
    #     all_total_rewards = None
    #     print("Target environment training skipped (not gym)")

    # # Plot results from target environment
    # save_and_plot_results(transfer_params['target_env'], target_env_params, target_pid_params, reward_history=all_total_rewards, save=False, plot=True)
    
    # # ============================================================================
    # # PHASE 4: SAVE RESULTS AND PLOT
    # # ============================================================================
    # # Save source environment results
    # save_and_plot_results(transfer_params['source_env'], source_env_params, source_pid_params, recorder=source_recorder, save=True, plot=False)
    # # Save target environment results
    # save_and_plot_results(transfer_params['target_env'], target_env_params, target_pid_params, reward_history=all_total_rewards, save=True, plot=False)
    
    # print("\n🎉 Transfer learning complete!")
    # print(f"Successfully transferred from {transfer_params['source_env']} to {transfer_params['target_env']}")
    # if transfer_params['fix_source_weights'] > 0:
    #     print(f"Fixed transferred weights for first {transfer_params['fix_source_weights']} episodes")

    

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Define sweep grid
    kd_values        = experiment_params["kd_values"]
    omission_probs   = experiment_params["omission_probs"]
    repeats          = experiment_params["repeats"]

    max_batch_sizes  = experiment_params["max_batch_sizes"]
    num_recents      = experiment_params["num_recents"]

    # Save results settings
    batch_name = 'kd_omission_sweep'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(script_dir, "PID-results-ext_buffer")

    os.makedirs(results_root, exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y%m%d")
    save_dir = os.path.join(results_root, f"{today}-{batch_name}")
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # --- New sweep loop ---
    for max_b, num_r in zip(max_batch_sizes, num_recents):
        print(f"\n=== Testing max_batch_size={max_b}, num_recent={num_r} ===")

        for kd, omit in itertools.product(kd_values, omission_probs):
            operant_session_params["omission_prob"] = omit
            operant_session_params["max_batch_size"] = max_b
            operant_session_params["num_recent"] = num_r
            operant_pid_params["kd"] = kd
            operant_pid_params["meta_lr_d"] = min(kd, 0.1)

            print(f"\n--- Running sweep: kd={kd}, omission_prob={omit} ---")
            run_transfer_learning()

