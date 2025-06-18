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
    reward_counts_above_2 = {}
    all_batch_cue_errors = {}

    for batch_name, batch_data in batches.items():
        all_rewards = []
        td_amplitudes = []  # List of np.arrays (one per session)
        counts_per_session = []
        cue_errors = []  # List of np.arrays (one per session)

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
                cue_error,
                _,
                _,
                trial_TD_amplitude,
                t_axis,
                trial_axis
            ) = load_recorder_data(recorder, dt=dt, pre_steps=pre_steps, post_steps=post_steps)

            all_rewards.extend(reward_history)
            td_amplitudes.append(np.array(trial_TD_amplitude))
            count = np.sum(np.array(reward_history) > 2)
            counts_per_session.append(count)
            cue_error = np.array(cue_error)  # shape: (num_trials, time_steps)
            cue_errors.append(cue_error)

        all_batch_rewards[batch_name] = all_rewards
        all_batch_td[batch_name] = td_amplitudes
        reward_counts_above_2[batch_name] = counts_per_session
        all_batch_cue_errors[batch_name] = cue_errors

    return all_batch_rewards, all_batch_td, reward_counts_above_2, all_batch_cue_errors, t_axis

def plot_pid_results(root_dir="PID-results"):
    batches = scan_batches(root_dir)
    all_batch_rewards, all_batch_td, reward_counts_above_2, all_batch_cue_errors, t_axis = extract_batch_data(batches)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(batches)))

    # Access subplots
    ax1 = axs[0, 0]  # Reward boxplot
    ax2 = axs[0, 1]  # TD amplitude traces
    ax3 = axs[1, 0]  # Reward count > 2
    ax4 = axs[1, 1]  # Average TD error

    # Plot 1: Reward boxplot per batch
    ax1.boxplot(
        all_batch_rewards.values(),
        labels=all_batch_rewards.keys(),
        widths=0.5
    )
    ax1.set_title("Combined Reward Distribution Per Batch")
    ax1.set_ylabel("Reward")

    # Plot 2: TD amplitude traces
    for i, (batch_name, td_sessions) in enumerate(all_batch_td.items()):
        color = colors[i]
        all_td_array = np.stack(td_sessions)  # shape: (num_sessions, num_trials)

        for session_td in all_td_array:
            ax2.plot(session_td, color=color, alpha=0.3)

        mean_td = np.mean(all_td_array, axis=0)
        ax2.plot(mean_td, color=color, linewidth=2.5, label=batch_name)

    ax2.set_title("TD Amplitude During Cue Across Trials")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("TD Amplitude")
    ax2.legend()

    # Plot 3: Reward count > 2 per batch
    batch_labels = list(reward_counts_above_2.keys())
    means = [np.mean(reward_counts_above_2[batch]) for batch in batch_labels]
    stds = [np.std(reward_counts_above_2[batch]) for batch in batch_labels]

    x = np.arange(len(batch_labels))
    bar_colors = plt.cm.tab10(np.linspace(0, 1, len(batch_labels)))

    ax3.bar(x, means, yerr=stds, color=bar_colors, capsize=5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(batch_labels, rotation=45, ha="right")
    ax3.set_ylabel("Rewards (avg over sessions)")
    ax3.set_title("Success Trials (Big Outcome) per Batch")

    # Plot 4: Average TD error trace per batch
    for i, (batch_name, cue_error_list) in enumerate(all_batch_cue_errors.items()):
        stacked_errors = np.concatenate(cue_error_list, axis=0)
        mean_error = stacked_errors.mean(axis=0)

        color = colors[i]
        ax4.plot(t_axis, mean_error, color=color, label=batch_name, linewidth=2)

    ymin, ymax = ax4.get_ylim()
    ax4.fill_betweenx([ymin, ymax], 0, 0.5, color='tab:orange', alpha=0.2)
    ax4.set_ylim(ymin, ymax)

    ax4.set_title("Average TD Error Trace Per Batch")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Average TD Error")
    ax4.legend()

    plt.tight_layout()
    plt.show()


plot_pid_results("PID-results")
