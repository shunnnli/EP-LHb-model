import os,sys
import re
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotfunctions import load_recorder_data, plotSEM, plotScatterBar, plotLine, model_name, short_label


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
    all_batch_ant_licks = {}
    all_batch_stuck_counts = {}
    # Dynamic aggregation for any additional outputs
    all_batch_extras = {}

    # Sort batches by (kd, omit, max_b, num_r) for consistent plotting
    sorted_items = sorted(
        batches.items(),
        key=lambda kv: (kv[0][3], # num_r
                        kv[0][2], # max_b
                        kv[0][1], # omit
                        kv[0][0]) # kd
    )

    for idx, ((kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign), batch_data) in enumerate(sorted_items):
        all_rewards = []
        td_amplitudes = []  # List of np.arrays (one per session)
        success_per_session = []
        cue_errors = []  # List of np.arrays (one per session)
        batch_name = (kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign)  # Use tuple as batch name
        ant_licks = []
        all_stuck_counts = []
        batch_extras = {}
        print(f"Processing batch: {batch_name}")

        for session, session_data in batch_data.items():
            session_params = session_data['session_params']
            pre_steps = session_params['pre_steps']
            post_steps = session_params['post_steps']
            dt = session_params['dt']
            recorder = session_data['recorder']
            stuck_counts = session_data["stuck_counts"]
            output_dict = load_recorder_data(recorder, dt=dt, pre_steps=pre_steps, post_steps=post_steps)
            reward_history = output_dict['reward_history']

            p_history = output_dict['p_history']
            i_history = output_dict['i_history']
            d_history = output_dict['d_history']
            kp_history = output_dict['kp_history']
            ki_history = output_dict['ki_history']
            kd_history = output_dict['kd_history']
            p_kp_history = output_dict['p_kp_history']
            i_ki_history = output_dict['i_ki_history']
            d_kd_history = output_dict['d_kd_history']

            update_step = output_dict['update_step']
            td_errors = output_dict['td_errors']
            td_pid_errors = output_dict['td_pid_errors']
            DAs = output_dict['DAs']
            cue_licks = output_dict['cue_licks']
            cue_error = output_dict['cue_error']
            cue_omissions = output_dict['cue_omissions']
            cue_levels = output_dict['cue_levels']
            cue_d_update_step = output_dict['cue_d_update_step']
            trial_anticipatory_licks = output_dict['trial_anticipatory_licks']
            trial_TD_amplitude = output_dict['trial_TD_amplitude']
            trial_pid_TD_amplitude = output_dict['trial_pid_TD_amplitude']
            
            trial_animal_sign_index = output_dict['trial_animal_sign_index']
            eplhb_out = output_dict['eplhb_out']
            eplhb_coeff = output_dict['eplhb_coeff']
            cue_EPLHb_output = output_dict['cue_EPLHb_output']
            cue_EPLHb_coeff = output_dict['cue_EPLHb_coeff']

            t_axis = output_dict['t_axis']
            trial_axis = output_dict['trial_axis']

            all_rewards.extend(reward_history)
            td_amplitudes.append(np.array(trial_TD_amplitude))
            ant_licks.append(np.array(trial_anticipatory_licks))
            reward_arr = np.array(reward_history)#[session_params['change_start']:] only for task switching
            count = np.sum(reward_arr > 2)
            success_per_session.append(count)
            cue_error = np.array(cue_error)  # shape: (num_trials, time_steps)
            cue_errors.append(cue_error)
            all_stuck_counts.append(stuck_counts)

            # Collect any additional keys from output_dict dynamically
            for k, v in output_dict.items():
                if k in {'reward_history', 'trial_TD_amplitude', 'trial_pid_TD_amplitude', 'trial_anticipatory_licks', 'cue_error', 't_axis', 'trial_axis'}:
                    continue
                if k not in batch_extras:
                    batch_extras[k] = []
                try:
                    batch_extras[k].append(np.array(v))
                except Exception:
                    batch_extras[k].append(v)

        all_batch_rewards[batch_name] = all_rewards
        all_batch_td[batch_name] = td_amplitudes
        success_trials[batch_name] = success_per_session
        all_batch_cue_errors[batch_name] = cue_errors
        all_batch_ant_licks[batch_name] = ant_licks
        all_batch_stuck_counts[batch_name] = all_stuck_counts
        # Move batch extras into global container
        for k, per_session in batch_extras.items():
            if k not in all_batch_extras:
                all_batch_extras[k] = {}
            all_batch_extras[k][batch_name] = per_session

    return all_batch_rewards, all_batch_td, success_trials, all_batch_cue_errors, t_axis, all_batch_ant_licks, all_batch_stuck_counts, all_batch_extras


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

def load_batches(root_dir, 
                filter_kd=None, 
                filter_omit=None, 
                filter_max_b=None, 
                filter_num_r=None,
                filter_fixed_sign=None,
                filter_eplhb_fixed_sign=None,
                filter_repeat=None,
                ):
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

    # group by (kd, omit, max_b, num_r)
    batches = {}
    for (kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign, repeat), data in raw_results.items():
        batches.setdefault((kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign), {})[repeat] = data
    
    # Apply filters based on input parameters
    if any([filter_kd is not None, filter_omit is not None, filter_max_b is not None, 
            filter_num_r is not None, filter_fixed_sign is not None, filter_eplhb_fixed_sign is not None, filter_repeat is not None, ]):
        
        filtered_batches = {}
        for (kd, omit, max_b, num_r, eplhb_fixed_sign, fixed_sign), batch_data in batches.items():
            # Check if this batch matches all specified filters
            if filter_kd is not None and kd not in filter_kd:
                continue
            if filter_omit is not None and omit not in filter_omit:
                continue
            if filter_max_b is not None and max_b not in filter_max_b:
                continue
            if filter_num_r is not None and num_r not in filter_num_r:
                continue
            if filter_fixed_sign is not None and fixed_sign not in filter_fixed_sign:
                continue
            if filter_eplhb_fixed_sign is not None and eplhb_fixed_sign not in filter_eplhb_fixed_sign:
                continue
            
            # For repeat filtering, we need to check individual sessions
            if filter_repeat is not None:
                filtered_sessions = {repeat: data for repeat, data in batch_data.items() 
                                  if repeat in filter_repeat}
                if filtered_sessions:  # Only keep batch if it has matching repeats
                    filtered_batches[(kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign)] = filtered_sessions
            else:
                filtered_batches[(kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign)] = batch_data
        
        batches = filtered_batches
        print(f"Applied filters: kd={filter_kd}, omit={filter_omit}, max_b={filter_max_b}, "
              f"num_r={filter_num_r}, fixed_sign={filter_fixed_sign}, eplhb_fixed_sign={filter_eplhb_fixed_sign}, repeat={filter_repeat}")
        print(f"Filtered from {len(raw_results)} to {len(batches)} batches")

    # extract data
    print("Extracting data from batches...")
    all_rewards, all_td, success_trials, cue_errors, t_axis, ant_licks, stuck_counts, all_batch_extras = extract_batch_data(batches)

    return all_rewards, all_td, success_trials, cue_errors, t_axis, ant_licks, stuck_counts, all_batch_extras



def plot_pid_results(root_dir="PID-results", 
                    filter_kd=None, 
                    filter_omit=None, 
                    filter_max_b=None, 
                    filter_num_r=None, 
                    filter_repeat=None,
                    filter_fixed_sign=None,
                    filter_eplhb_fixed_sign=None,
                    title="PID Results"):
    latest = get_latest_run_folder(root_dir)
    all_rewards, all_td, success_trials, cue_errors, t_axis, ant_licks, stuck_counts, all_batch_extras = load_batches(
        root_dir, filter_kd, filter_omit, filter_max_b, filter_num_r, filter_fixed_sign, filter_eplhb_fixed_sign, filter_repeat,)
    
    # ------- plot results -------
    fig = plt.figure(figsize=(26, 8))  # make figure a bit wider
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[0, 2])

    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1:3])


    # Set up colors and x-axis
    # --- 1) figure out your ordering (eplhb_fixed_sign, fixed_sign, num_r, max_b, omit, kd) ---
    sorted_items = sorted(
        all_td.items(), 
        key=lambda kv: (kv[0][5], kv[0][4], kv[0][3], kv[0][2], kv[0][1], kv[0][0])   # (eplhb_fixed_sign, fixed_sign, num_r, max_b, omit, kd)
    )

    sorted_licks = sorted(
        ant_licks.items(),
        key=lambda kv: (kv[0][5], kv[0][4], kv[0][3], kv[0][2], kv[0][1], kv[0][0])   # (eplhb_fixed_sign, fixed_sign, num_r, max_b, omit, kd)
    )

    # --- 2) extract the unique omits in order and pick a base color for each ---
    unique_omits = sorted({omit for (kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign) in all_td.keys()})
    base_rgbs    = plt.cm.tab10(np.linspace(0, 1, len(unique_omits)))
    omit_to_rgb  = dict(zip(unique_omits, base_rgbs))  # { omit → (r,g,b,…) }

    # --- 3) group the kd's under each omit so you can space their alphas ---
    kd_groups = {
        omit: sorted(k for (k, o, m, n, f, e) in all_td.keys() if o == omit)
        for omit in unique_omits
    }

    # --- 4) build a parallel list of RGBA tuples in your plotting order ---
    plot_colors = []
    for (kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign), _ in sorted_items:
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
    labels = [short_label(k, o, m, n, f, e) for (k, o, m, n, f, e) in all_rewards]
    plotScatterBar(all_rewards.values(),labels=labels, colors=colors, style='box', ax=ax1)
    ax1.set_ylabel("Reward")
    ax1.set_title("Combined Reward Distribution")


    # TD Amplitudes: last 25% of trials per session
    print("Plotting TD amplitudes (last 25%)...")
    td_flat_data = []
    for (kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign), td_sessions in sorted_items:
        trimmed = [arr[int(len(arr) * 0.75):] for arr in td_sessions]  # last 25% of each session
        td_flat_data.append(np.concatenate(trimmed))
    plotScatterBar(td_flat_data, labels=labels, colors=colors, style='box', ax=ax2)
    ax2.set_ylabel("TD Amplitude")
    ax2.set_title("TD Amplitude (Last 25% of Trials)")


    # Success trials
    print("Plotting success trials...")
    plotScatterBar(success_trials.values(),labels=labels, colors=colors, style='bar', ax=ax3)
    ax3.set_ylabel("Success trials")
    ax3.set_title("Success trials (reward > 2) after change start")

    # Anticipatory Licks
    for (kd, omit, max_b, num_r, fixed_sign, eplhb_fixed_sign), ant_licks_sessions in sorted_licks:
        # Determine standard_length from the first session's length
        if len(ant_licks_sessions) == 0:
            continue  # skip if no sessions
        standard_length = len(ant_licks_sessions[0])
        fixed_sessions = [
            np.pad(s, (0, standard_length - len(s)), mode='constant') if len(s) < standard_length else s[:standard_length]
            for s in ant_licks_sessions
        ]
        ant_licks_sessions_array = np.stack(fixed_sessions)
        avg = np.mean(ant_licks_sessions_array, axis=0)
        sem = np.std(ant_licks_sessions_array, axis=0) / np.sqrt(ant_licks_sessions_array.shape[0])

        x = np.arange(1, avg.shape[0] + 1)
        ax4.plot(x, avg, label=f"kd={kd}, omit={omit}, max_b={max_b}, num_r={num_r}, model={model_name(fixed_sign, eplhb_fixed_sign)}")
        ax4.fill_between(x, avg - sem, avg + sem, alpha=0.3)  # standard error band

    ax4.set_ylabel("Anticipatory Licks (avg)")
    ax4.set_xlabel("Trial Num")
    ax4.set_title("Anticipatory Licks")
    ax4.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)


    # Stuck Counts
    combos = list(stuck_counts.keys())
    means = [np.mean(stuck_counts[c]) for c in combos]
    sems  = [np.std(stuck_counts[c], ddof=1) / np.sqrt(len(stuck_counts[c])) for c in combos]
    ax5.bar(labels, means, yerr=sems, capsize=5, color='skyblue')
    ax5.set_ylabel("Stuck Counts (mean ± SEM)")
    ax5.set_title("Average Stuck Counts Per Repeat")
    ax5.set_xticklabels(labels, rotation=30, ha='right', fontsize=4.5)
    
    
    # # Success/performance by omission level
    # # currently defining performance as success_trials
    # print("Plotting performance delta by omisssion level...")
    # plotLine(unique_omits=unique_omits, performance=success_trials, ax=ax4)
    # # ax4.hlines(0, color='gray', linestyle='--', linewidth=0.5)
    # ax4.set_ylabel("Δ Avg Success Trials\n(from kd=0)")
    # ax4.set_title("Δ in Performance from Kd=0")
    # ax4.legend(title="Kd levels", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,frameon=False)

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
    fig_path = os.path.join(latest, f"{title}.png")
    fig.savefig(fig_path, dpi=300)
    print(f"Figure saved to: {fig_path}")
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(script_dir, "PID-results-ext_buffer")
    plot_pid_results(results_root)  # No filters applied