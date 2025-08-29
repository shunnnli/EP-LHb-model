import os,sys
import re
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotfunctions import load_recorder_data, plotSEM, plotScatterBar, plotLine
from collections import defaultdict

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
    all_batch_td_pid = {}
    success_trials = {}
    all_batch_cue_errors = {}
    all_batch_cue_kd_d = {}
    all_batch_ant_licks = {}
    all_full_errors = {}
    all_batch_stuck_counts = {}

    # Sort batches by (kd, omit) for consistent plotting
    sorted_items = sorted(
        batches.items(),
        key=lambda kv: (kv[0][3], # num recent
                        kv[0][2], # max batch size
                        kv[0][1],  # omit
                        kv[0][0])  # kd
    )

    for idx, ((kd, omit, max_b, num_r), batch_data) in enumerate(sorted_items):
        all_rewards = []
        td_amplitudes = []  # List of np.arrays (one per session)
        td_pid_amplitudes = []
        success_per_session = []
        cue_errors = []  # List of np.arrays (one per session)
        cue_kd_ds = []
        ant_licks = []
        full_errors = []
        all_stuck_counts = []
        batch_name = (kd, omit, max_b, num_r)  # Use tuple as batch name
        print(f"Processing batch: {batch_name}")

        for session, session_data in batch_data.items():
            session_params = session_data['session_params']
            pre_steps = session_params['pre_steps']
            post_steps = session_params['post_steps']
            dt = session_params['dt']
            stuck_counts = session_data["stuck_counts"]
            recorder = session_data['recorder']
            (
            _,
            reward_history,
            p_kp_history,
            i_ki_history,
            d_kd_history,
            kp_history,
            ki_history,
            kd_history,
            cue_licks,
            cue_error,
            cue_pid_error,
            cue_kd_d,
            cue_omissions,
            cue_levels,
            trial_anticipatory_licks,
            trial_TD_amplitude,
            trial_pid_TD_amplitude,
            t_axis,
            trial_axis,
            error
        ) = load_recorder_data(recorder, dt=dt, pre_steps=pre_steps, post_steps=post_steps)

            all_rewards.extend(reward_history)
            td_amplitudes.append(np.array(trial_TD_amplitude))
            td_pid_amplitudes.append(np.array(trial_pid_TD_amplitude))
            reward_arr = np.array(reward_history)[200:]
            count = np.sum(reward_arr > 2)
            success_per_session.append(count)
            cue_error = np.array(cue_error)  # shape: (num_trials, time_steps)
            cue_errors.append(cue_error)
            cue_kd_ds.append(np.array(cue_kd_d))
            ant_licks.append(np.array(trial_anticipatory_licks))
            full_errors.append(np.array(error))
            all_stuck_counts.append(stuck_counts)

        all_batch_rewards[batch_name] = all_rewards
        all_batch_td[batch_name] = td_amplitudes
        all_batch_td_pid[batch_name] = td_pid_amplitudes
        success_trials[batch_name] = success_per_session
        all_batch_cue_errors[batch_name] = cue_errors
        all_batch_cue_kd_d[batch_name] = cue_kd_ds
        all_batch_ant_licks[batch_name] = ant_licks
        all_full_errors[batch_name] = full_errors
        all_batch_stuck_counts[batch_name] = all_stuck_counts



    return all_batch_rewards, all_batch_td, all_batch_td_pid, success_trials, all_batch_cue_errors, all_batch_cue_kd_d, all_batch_ant_licks, t_axis, all_full_errors, all_batch_stuck_counts


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
    for (kd, omit, max_b, num_r, repeat), data in raw_results.items():
        batches.setdefault((kd, omit, max_b, num_r), {})[repeat] = data

    # extract data
    print("Extracting data from batches...")
    all_rewards, all_td, all_td_pid, success_trials, cue_errors, cue_kd_d, ant_licks, t_axis, full_errors, stuck_counts = extract_batch_data(batches)

    # ------- plot results -------
    fig = plt.figure(figsize=(26, 8))  # make figure a bit wider
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[0, 2])

    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1:3])

    # Set up colors and x-axis
    # --- 1) figure out your ordering (omit first, then kd) ---
    sorted_items = sorted(
        all_td.items(), 
        key=lambda kv: (kv[0][3], kv[0][2], kv[0][1], kv[0][0])   # (num_r, max_b, omit, kd)
    )

    sorted_items_pid = sorted(
        all_td_pid.items(),
        key=lambda kv: (kv[0][3], kv[0][2], kv[0][1], kv[0][0])   # (num_r, max_b, omit, kd)
    )

    sorted_licks = sorted(
        ant_licks.items(),
        key=lambda kv: (kv[0][3], kv[0][2], kv[0][1], kv[0][0])   # (num_r, max_b, omit, kd)
    )

    sorted_errors = sorted(
        full_errors.items(),
        key=lambda kv: (kv[0][3], kv[0][2], kv[0][1], kv[0][0])   # (num_r, max_b, omit, kd)
    )

    # --- 2) extract the unique omits in order and pick a base color for each ---
    unique_omits = sorted({omit for (kd, omit, max_b, num_r) in all_td.keys()})
    base_rgbs    = plt.cm.tab10(np.linspace(0, 1, len(unique_omits)))
    omit_to_rgb  = dict(zip(unique_omits, base_rgbs))  # { omit → (r,g,b,…) }

    # --- 3) group the kd’s under each omit so you can space their alphas ---
    kd_groups = {
        omit: sorted(k for (k, o, m, n) in all_td.keys() if o == omit)
        for omit in unique_omits
    }

    # --- 4) build a parallel list of RGBA tuples in your plotting order ---
    plot_colors = []
    for (kd, omit, max_b, num_r), _ in sorted_items:
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
    labels = [f"kd={k}, omit={o}, max_b={m}, num_r={n}" for (k, o, m, n) in all_rewards]
    plotScatterBar(all_rewards.values(),labels=labels, colors=colors, style='box', ax=ax1)
    ax1.set_ylabel("Reward")
    ax1.set_title("Combined Reward Distribution Per Batch")

    # TD Amplitudes: last 25% of trials per session
    print("Plotting TD amplitudes...")
    td_flat_data = []
    for (kd, omit, max_b, num_r), td_sessions in sorted_items:
        trimmed = [arr[:int(len(arr) * 1)] for arr in td_sessions]  # last 25% of each session
        td_flat_data.append(np.concatenate(trimmed))
    td_pid_flat_data = []
    for (kd, omit, max_b, num_r), td_sessions in sorted_items_pid:
        trimmed = [arr[:int(len(arr) * 1)] for arr in td_sessions]  # last 25% of each session
        td_pid_flat_data.append(np.concatenate(trimmed))
    
    plotScatterBar(td_flat_data, labels=labels, colors=colors, style='box', ax=ax2)
    # plotScatterBar(td_pid_flat_data, labels=labels, colors=colors, style='box', ax=ax2, pos_offset=0.15, fill=False)
    # ax2.plot([], [], color='white', label="TD Error", linewidth=1)
    # ax2.plot([], [], color='black', label="PID TD Error", linewidth=1)
    # ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
    ax2.set_ylabel("TD Amplitude")
    ax2.set_title("TD Amplitude")


    

    # Success trials
    print("Plotting success trials...")
    plotScatterBar(success_trials.values(),labels=labels, colors=colors, style='bar', ax=ax3)
    ax3.set_ylabel("Success trials (reward > 2)")
    ax3.set_title("Success trials after start swapping (trial 200)")
    
    
    # # Success/performance by omission level
    # # currently defining performance as success_trials
    # print("Plotting performance delta by omisssion level...")
    # plotLine(unique_omits=unique_omits, performance=success_trials, ax=ax4)
    # # ax4.hlines(0, color='gray', linestyle='--', linewidth=0.5)
    # ax4.set_ylabel("Δ Avg Success Trials\n(from kd=0)")
    # ax4.set_title("Δ in Performance from Kd=0")
    # ax4.legend(title="Kd levels", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,frameon=False)


    # for (kd, omit), ant_licks_sessions in sorted_items:
    #     ant_licks_sessions_array = np.stack(ant_licks_sessions)  # shape: (10, 300)
    #     avg = np.mean(ant_licks_sessions_array, axis=0, keepdims=True)
    #     ax4.plot(trial_axis, avg.squeeze(), label=f"kd={kd}")

    # map keys to colors in the same order as sorted_items
    key_to_color = {}
    for color, ((kd, omit, max_b, num_r), _) in zip(colors, sorted_items):
        key_to_color[(kd, omit, max_b, num_r)] = color

    for (kd, omit, max_b, num_r), ant_licks_sessions in sorted_licks:
        standard_length = 600
        fixed_sessions = [np.pad(s, (0, standard_length - len(s)), mode='constant') if len(s) < standard_length else s[:standard_length] for s in ant_licks_sessions]
        ant_licks_sessions_array = np.stack(fixed_sessions)
        avg = np.mean(ant_licks_sessions_array, axis=0)
        sem = np.std(ant_licks_sessions_array, axis=0) / np.sqrt(ant_licks_sessions_array.shape[0])

        x = np.arange(1, avg.shape[0] + 1)
        c = key_to_color.get((kd, omit, max_b, num_r), (0.5, 0.5, 0.5, 0.7))  # fallback gray
        ax4.plot(x, avg, label=f"kd={kd}, omit={omit}, max_b={max_b}, num_r={num_r}", color=c)
        ax4.fill_between(x, avg - sem, avg + sem, alpha=0.3, color=c)

    ax4.set_ylabel("Anticipatory Licks (avg)")
    ax4.set_xlabel("Trial Num")
    ax4.set_title("Anticipatory Licks")
    ax4.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)

    # Stuck Counts
    combos = list(stuck_counts.keys())
    means = [np.mean(stuck_counts[c]) for c in combos]
    sems = [(np.std(stuck_counts[c], ddof=1) / np.sqrt(len(stuck_counts[c]))) if len(stuck_counts[c]) > 1 else 0 for c in combos]
    labels_stuck = [f"kd={kd},omit={omit}, max_b={max_b}, num_r={num_r}" for (kd, omit, max_b, num_r) in combos]
    ax5.bar(labels_stuck, means, yerr=sems, capsize=5, color=colors)
    ax5.set_ylabel("Stuck Counts (mean ± SEM)")
    ax5.set_title("Average Stuck Counts Per Repeat")
    ax5.set_xticks(range(len(labels_stuck)))
    ax5.set_xticklabels(labels_stuck, rotation=45, ha='right')


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


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(script_dir, "PID-results-ext_buffer")
    plot_pid_results(results_root)