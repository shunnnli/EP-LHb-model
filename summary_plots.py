import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from plotfunctions import load_recorder_data

def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def scan_batches(root_dir):
    """
    Scan folders under root_dir to load batches of data.
    Uses image filename pattern to create batch keys.
    """
    batches = {}
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        result_path = os.path.join(folder_path, "results.pkl")

        if os.path.isdir(folder_path) and os.path.isfile(result_path):
            try:
                image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
                if not image_files:
                    print(f"No images in {folder_path}, skipping.")
                    continue

                match = re.match(r"(.+?)_repeat_\d+\.png", image_files[0])
                if not match:
                    print(f"No matching image pattern in {folder_path}, skipping.")
                    continue

                batch_label = match.group(1)  # e.g., "kd_0.3_omit_0.2"
                batch_data = load_data(result_path)
                batches[batch_label] = batch_data

            except Exception as e:
                print(f"Error loading {folder_path}: {e}")

    return batches

def extract_batch_data(batches):
    """
    From batch data dictionary, extract combined reward lists and TD amplitudes.
    """
    all_batch_rewards = {}
    all_batch_td = {}

    for batch_name, batch_data in batches.items():
        all_rewards = []
        td_amplitudes = []  # List of np.arrays (one per session)

        for session, session_data in batch_data.items():
            session_params = session_data['session_params']
            pre_steps = session_params['pre_steps']
            post_steps = session_params['post_steps']
            dt = session_params['dt']
            recorder = session_data['recorder']

            (
                num_trials,
                reward_history,
                *_,
                trial_TD_amplitude,
                t_axis,
                trial_axis
            ) = load_recorder_data(recorder, dt=dt, pre_steps=pre_steps, post_steps=post_steps)

            all_rewards.extend(reward_history)
            td_amplitudes.append(np.array(trial_TD_amplitude))

        all_batch_rewards[batch_name] = all_rewards
        all_batch_td[batch_name] = td_amplitudes

    return all_batch_rewards, all_batch_td

def plot_pid_results(root_dir="PID-results"):
    batches = scan_batches(root_dir)
    all_batch_rewards, all_batch_td = extract_batch_data(batches)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(batches)))

    # Plot 1: Reward boxplot per batch
    ax1.boxplot(
        all_batch_rewards.values(),
        labels=all_batch_rewards.keys(),
        widths=0.5
    )
    ax1.set_title("Combined Reward Distribution Per Batch")
    ax1.set_ylabel("Reward")
    ax1.grid(True)

    # Plot 2: TD amplitude traces
    for i, (batch_name, td_sessions) in enumerate(all_batch_td.items()):
        color = colors[i]
        all_td_array = np.stack(td_sessions)  # shape: (num_sessions, num_trials)

        # Pale individual session traces
        for session_td in all_td_array:
            ax2.plot(session_td, color=color, alpha=0.3)

        # Average trace
        mean_td = np.mean(all_td_array, axis=0)
        ax2.plot(mean_td, color=color, linewidth=2.5, label=batch_name)

    ax2.set_title("TD Amplitude During Cue Across Trials")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("TD Amplitude")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


plot_pid_results("PID-results")
