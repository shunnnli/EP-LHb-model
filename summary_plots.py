import os,sys
import re
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from plotfunctions import load_recorder_data, plotSEM, plotScatterBar, plotLine

repo_path = os.path.abspath("./PID-Accelerated-TD-Learning")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)


def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def extract_batch_data(batches):
    """
    From batch data dictionary, extract combined reward lists and TD amplitudes.
    """
    all_batch_rewards = {}
    all_batch_td = {}
    success_trials = {}
    all_batch_cue_errors = {}

    # Sort batches by (kd, omit) for consistent plotting
    sorted_items = sorted(
        batches.items(),
        key=lambda kv: (kv[0][1],  # omit
                        kv[0][0])  # kd
    )
    batch_names = [f"kd={kd},omit={omit}" for (kd, omit), _ in sorted_items]

    for idx, ((kd, omit), batch_data) in enumerate(sorted_items):
        all_rewards = []
        td_amplitudes = []  # List of np.arrays (one per session)
        success_per_session = []
        cue_errors = []  # List of np.arrays (one per session)
        batch_name = (kd, omit)  # Use tuple as batch name
        print(f"Processing batch: {batch_name}")

        for session, session_data in batch_data.items():
            session_params = session_data['session_params']
            pre_steps = session_params['pre_steps']
            post_steps = session_params['post_steps']
            dt = session_params['dt']
            recorder = session_data['recorder']
            (
                _,
                reward_history,
                *_,
                cue_error,
                _,
                trial_anticipatory_licks,
                trial_TD_amplitude,
                t_axis,
                _,
            ) = load_recorder_data(recorder, dt=dt, pre_steps=pre_steps, post_steps=post_steps)

            all_rewards.extend(reward_history)
            td_amplitudes.append(np.array(trial_TD_amplitude))
            count = np.sum(np.array(trial_anticipatory_licks) > 2)
            success_per_session.append(count)
            cue_error = np.array(cue_error)  # shape: (num_trials, time_steps)
            cue_errors.append(cue_error)

        all_batch_rewards[batch_name] = all_rewards
        all_batch_td[batch_name] = td_amplitudes
        success_trials[batch_name] = success_per_session
        all_batch_cue_errors[batch_name] = cue_errors

    return all_batch_rewards, all_batch_td, success_trials, all_batch_cue_errors, t_axis


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
    print(f"Loading results from: {latest}")
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
    print("Extracting data from batches...")
    all_rewards, all_td, success_trials, cue_errors, t_axis = extract_batch_data(batches)

    # ------- plot results -------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2 = axs[0]      # top row
    ax3, ax4 = axs[1]      # bottom row

    # Set up colors and x-axis
    # --- 1) figure out your ordering (omit first, then kd) ---
    sorted_items = sorted(
        all_td.items(), 
        key=lambda kv: (kv[0][1], kv[0][0])   # (omit, kd)
    )

    # --- 2) extract the unique omits in order and pick a base color for each ---
    unique_omits = sorted({omit for (kd, omit) in all_td.keys()})
    base_rgbs    = plt.cm.tab10(np.linspace(0, 1, len(unique_omits)))
    omit_to_rgb  = dict(zip(unique_omits, base_rgbs))  # { omit → (r,g,b,…) }

    # --- 3) group the kd’s under each omit so you can space their alphas ---
    kd_groups = {
        omit: sorted(k for (k, o) in all_td.keys() if o == omit)
        for omit in unique_omits
    }

    # --- 4) build a parallel list of RGBA tuples in your plotting order ---
    plot_colors = []
    for (kd, omit), _ in sorted_items:
        grp = kd_groups[omit]
        idx = grp.index(kd)
        # linearly space alpha between 0.3 and 1.0 for this group
        alphas = np.linspace(0.3, 1.0, len(grp))
        r, g, b, _ = omit_to_rgb[omit]
        plot_colors.append((r, g, b, alphas[idx]))
    colors = [tuple(c) for c in plot_colors]  # convert to tuples for matplotlib

    # Prepare x axis and labels
    trial_axis = np.arange(len(next(iter(all_td.values()))[0]))  # Assuming all sessions have same length

    # Rewards
    print("Plotting reward distributions...")
    labels = [f"kd={k},omit={o}" for (k, o) in all_rewards]
    plotScatterBar(all_rewards.values(),labels=labels, colors=colors, style='box', ax=ax1)
    ax1.set_ylabel("Reward")
    ax1.set_title("Combined Reward Distribution Per Batch")

    # TD Amplitudes: last 25% of trials per session
    print("Plotting TD amplitudes (last 25%)...")
    td_flat_data = []
    for (kd, omit), td_sessions in sorted_items:
        trimmed = [arr[int(len(arr) * 0.75):] for arr in td_sessions]  # last 25% of each session
        td_flat_data.append(np.concatenate(trimmed))
    plotScatterBar(td_flat_data, labels=labels, colors=colors, style='box', ax=ax2)
    ax2.set_ylabel("TD Amplitude")
    ax2.set_title("TD Amplitude (Last 25% of Trials)")

    # Reward‐count >2
    print("Plotting success trials...")
    plotScatterBar(success_trials.values(),labels=labels, colors=colors, style='bar', ax=ax3)
    ax3.set_ylabel("Rewards (avg over sessions)")
    ax3.set_title("Success Trials (Big Outcome) per Batch")

    # Average TD‐error
    # print("Plotting average TD error traces...")
    # for i, ((k, o), errs) in enumerate(cue_errors.items()):
    #     stacked = np.concatenate(errs, axis=0)
    #     # plotSEM(t_axis, stacked, color=colors[i], ax=ax4, label=f"kd={k},omit={o}")
    #     ax4.plot(t_axis, stacked.mean(0), color=colors[i], linewidth=1, label=f"kd={k},omit={o}")
    # ymin, ymax = ax4.get_ylim()
    # ax4.fill_betweenx([ymin, ymax], 0, 0.5, color='tab:orange', alpha=0.2)
    # ax4.set_ylim(ymin, ymax)
    # ax4.set_title("Average TD Error Trace Per Batch")
    # ax4.set_xlabel("Time (s)")
    # ax4.set_ylabel("Average TD Error")


    # success/performance by omission level
    # currently defining performance as success_trials
    print("Plotting success by omisssion level")
    plotLine(unique_omits=unique_omits, performance=success_trials, ax=ax4)
    ax4.set_xticks(np.arange(len(unique_omits)))
    ax4.set_xticklabels([f"omit={o}" for o in unique_omits])
    ax4.set_ylabel("Avg Success Trials")
    ax4.set_title("Success Trials by Omission Level")
    ax4.legend(
    title="KD",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    frameon=False
    )
    

    # Maximize figuer to fit the whole screen (unfinished)
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized()        # Qt backends
    except AttributeError:
        try:
            mng.window.state('zoomed')    # TkAgg on Windows
        except AttributeError:
            mng.full_screen_toggle()

    # Save the figure
    plt.tight_layout()
    fig_path = os.path.join(latest, "PID-results.png")
    fig.savefig(fig_path, dpi=300)
    print(f"Figure saved to: {fig_path}")
    plt.show()


plot_pid_results("PID-results")
