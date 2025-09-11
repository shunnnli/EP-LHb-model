#!/usr/bin/env python3
import os, sys, importlib
repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import OnlineReplayBuffer, ExtendedReplayBuffer
from TabularPID.Agents.DQN.DQN import EPLHb_DQN, PID_DQN
from TabularPID.Agents.DQN.DQN_gain_adapter import NoGainAdapter, SingleGainAdapter, DiagonalGainAdapter, NetworkGainAdapter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import random
from recorder import SessionRecorder

from OperantGym import OperantLearning
from plotfunctions import plot_figure
from trainfuntions import set_global_seeds, setup_model, setup_buffer, train_operant_environment

# ============================================================================
# OPERANT ENVIRONMENT PARAMETERS
# ============================================================================
operant_session_params = {
    "pairing":          'reward',
    "num_trials":       2,
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

    "continual_learning": True,
    "change_start": 200,
    "change_interval": 50,
}

operant_pid_params = {
    "learning_rate": 1e-3,
    "eplhb_lr": 1e-2,
    "coeff_lr": 1e-5,
    "initial_eplhb_coeff": -1.0,

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
    "seed": 2236,

    "kp": 1.0,
    "ki": 0.0,
    "kd": 0.0,
    "meta_lr": 0,
    "epsilon_gain": 0.1,
    "alpha": 0.05,
    "beta": 0.95,
    "d_tau": 1,
    "tabular_d": False,

    # Batch sampling parameters (like PID-Operant-Batch.py)
    "max_batch_size":   5,            # max replay buffer space
    "num_recent":       1,            # number of consecutive recent trials to fill replay buffer
}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
def setup_operant_environment():
    """Setup operant learning environment"""
    env = OperantLearning(
        pairing=operant_session_params["pairing"],
        omission_prob=operant_session_params["omission_prob"],
        enl_duration=operant_session_params["enl_duration"],
        action_cost=operant_session_params["action_cost"],
        enl_penalty=operant_session_params["enl_penalty"],
        continual_learning=operant_session_params["continual_learning"],
        change_start=operant_session_params["change_start"],
        change_interval=operant_session_params["change_interval"],
        print_status=False,
    )
    return env

# ============================================================================
# RESULTS SAVING AND PLOTTING
# ============================================================================
def save_and_plot_results(recorder, save=True, plot=True):
    """Save results and generate plots"""
    print(f"\n{'='*60}")
    print("Saving results and plotting")
    print(f"{'='*60}")
    
    if save:
        # Save the recorder and reward data with timestamp
        import datetime
        import pickle
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save results
        results = {
            "session_params": operant_session_params,
            "pid_params":     operant_pid_params,
            "recorder":       recorder,
            "seed":           operant_pid_params["seed"],
        }

        # Save everything
        with open(f"PID-results/{timestamp}-operant_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print(f"Saved operant results to {timestamp}-operant_results.pkl")
    
    if plot:
        # Plot results
        print("\n--- Plotting results from Operant Task ---")
        plot_figure(recorder, td_error_type='internal', dt=operant_session_params["dt"], show=True,
                pre_steps=operant_session_params["pre_steps"], post_steps=operant_session_params["post_steps"])

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def run_operant_training():
    """Main function to run operant learning training"""
    print("=" * 60)
    print("EPLHb Operant Learning Training")
    print("=" * 60)
    
    # Set global seed for reproducibility
    set_global_seeds(operant_pid_params["seed"])
    
    # Setup environment
    print(f"\nSetting up operant environment...")
    env = setup_operant_environment()
    
    print("\nTraining setup complete!")
    
    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"Training on operant environment")
    print(f"{'='*60}")
    
    # Training with retry logic for ENL breaks
    retrain = True
    first_training = True

    while retrain:
        if first_training:
            new_seed = operant_pid_params["seed"]
            first_training = False
        else:
            new_seed = random.randint(0, 10000)
        
        set_global_seeds(new_seed)
        operant_pid_params["seed"] = new_seed

        # Setup model with new seed
        print(f"\nSetting up EPLHb model with seed {new_seed}...")
        model, gain_adapter = setup_model(env, operant_pid_params, device="cpu", model_type="EPLHb")
        
        # Setup buffer for this model
        print(f"\nSetting up replay buffer...")
        orig_buffer = setup_buffer(model, "operant", env)
        
        # Train on operant environment
        recorder, retrain, got_stuck = train_operant_environment(
            model, env, operant_session_params, operant_pid_params, orig_buffer=orig_buffer,
            print_status=False
        )
        
        if retrain:
            print(f"Retraining with new seed: {new_seed}")
    
    # ============================================================================
    # SAVE RESULTS AND PLOT
    # ============================================================================
    save_and_plot_results(recorder, save=False, plot=True)
    
    print("\n🎉 Operant learning training complete!")
    print(f"Successfully trained for {operant_session_params['num_trials']} trials")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_operant_training()
