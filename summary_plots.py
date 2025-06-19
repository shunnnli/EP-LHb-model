import os
import re
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from plotfunctions import load_recorder_data


def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

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

def get_latest_run_folder(root_dir: str) -> str:
    """
    Scan root_dir for subfolders named like YYYYMMDD-*, parse the date,
    and return the path to the one with the newest date.
    """
    date_dirs = []
    pattern = re.compile(r"^(\d{8})-")
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if not os.path.isdir(full):
            continue
        m = pattern.match(name)
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            continue
        date_dirs.append((dt, full))

    if not date_dirs:
        raise FileNotFoundError(f"No dated folders (YYYYMMDD-*) found in {root_dir!r}")

    _, latest_folder = max(date_dirs, key=lambda x: x[0])
    return latest_folder
def plot_pid_results(root_dir="PID-results"):
    latest = get_latest_run_folder(root_dir)

    # Step 1: Combine all .pkl files inside the latest folder
    raw_results = {}
    for fname in os.listdir(latest):
        if fname.endswith(".pkl"):
            path = os.path.join(latest, fname)
            with open(path, "rb") as f:
                data = pickle.load(f)
                raw_results.update(data)  # Merge dictionaries

    # group by (kd, omit)
    batches = {}
    for (kd, omit, repeat), data in raw_results.items():
        batches.setdefault((kd, omit), {})[repeat] = data

    # extract data
    all_rewards, all_td, counts, cue_errors, t_axis = extract_batch_data(batches)

    # ------- plot results -------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(batches)))

    # Rewards
    ax1 = axs[0, 0]
    labels = [f"kd={k},omit={o}" for (k, o) in all_rewards]
    ax1.boxplot(all_rewards.values(), labels=labels)
    ax1.set_title("Combined Reward Distribution Per Batch")
    ax1.set_ylabel("Reward")

    # TD traces
    ax2 = axs[0, 1]
    for i, ((k, o), td_sessions) in enumerate(all_td.items()):
        arr = np.stack(td_sessions)
        for session_td in arr:
            ax2.plot(session_td, color=colors[i], alpha=0.3, linewidth=0.5)
        ax2.plot(arr.mean(0), color=colors[i], linewidth=1.5, label=f"kd={k},omit={o}")
    ax2.set_title("TD Amplitude During Cue Across Trials")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("TD Amplitude")
    ax2.legend()

    # Reward‐count >2
    ax3 = axs[1, 0]
    means = [np.mean(counts[b]) for b in counts]
    stds  = [np.std(counts[b])  for b in counts]
    x = np.arange(len(counts))
    ax3.bar(x, means, yerr=stds, color=colors, capsize=5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha="right")
    ax3.set_ylabel("Rewards (avg over sessions)")
    ax3.set_title("Success Trials (Big Outcome) per Batch")

    # Average TD‐error
    ax4 = axs[1, 1]
    for i, ((k, o), errs) in enumerate(cue_errors.items()):
        stacked = np.concatenate(errs, axis=0)
        ax4.plot(t_axis, stacked.mean(0), color=colors[i], linewidth=1, label=f"kd={k},omit={o}")
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
