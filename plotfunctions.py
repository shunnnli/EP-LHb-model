import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
from matplotlib import cm
import matplotlib.colors as mcolors

def plotSEM(x, y, omissions=None, label=None, color=None, ax=None, alpha=0.2, fill=False):
    """Plot with shaded error margin and red for omissions."""
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()

    norm = mcolors.Normalize(vmin=0, vmax=len(y))

    if not fill:
        for i, trace in enumerate(y):
            if omissions is not None and 1 in omissions[i]:
                alpha_i = 0.5
                color_i = cm.get_cmap('Reds')(norm(i))
            else:
                alpha_i = alpha
                color_i = color
            ax.plot(x, trace, linewidth=0.5, color=color_i, alpha=alpha_i, label="_nolegend_")
    else:
        mean = np.mean(y, axis=0)
        std  = np.std(y, axis=0)
        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color, edgecolor='None', label="_nolegend_")
        

    
def get_traces(data, event, pre_steps, post_steps):
    data = np.asarray(data)
    T    = data.shape[0]

    if len(data) == len(event):
        event_idx = np.where(np.diff(event) == 1)[0] + 1
    else:
        event_idx = np.asarray(event, dtype=int)

    n_trials   = len(event_idx)
    window_len = pre_steps + post_steps + 1

    aligned_data = np.zeros((n_trials, window_len), dtype=data.dtype)

    for i, idx in enumerate(event_idx):
        start = idx - pre_steps
        end   = idx + post_steps
        lo = max(start, 0)
        hi = min(end, T - 1)
        w_lo = lo - start
        w_hi = w_lo + (hi - lo) + 1
        aligned_data[i, w_lo : w_hi] = data[lo : hi + 1]

    return aligned_data


def plotScatterBar(data,
                   labels=None,
                   style='box',
                   ax=None,
                   colors=None,
                   width=0.6,
                   scatter_alpha=0.8,
                   error_bar_width=2,
                   error_bar_darker_factor=0.7):
    """
    Plot either:
      – a boxplot + scatter of every point  (style='box')
      – a barplot (mean±SEM) + scatter of every point (style='bar')

    Parameters
    ----------
    data : sequence of sequences
        A list of N groups, each group being an iterable of numbers.
    labels : sequence of str, optional
        Length-N list of tick labels.
    style : {'box', 'bar'}
        'box' for boxplot+points; 'bar' for barplot+points (SEM error bars).
    ax : matplotlib.axes.Axes, optional
        If None, a new figure+axes is created.
    colors : list of RGBA tuples, optional
        Length-N list of fill colors for each group.
    width : float
        Total width allocated per group.
    scatter_alpha : float
        Alpha for the overlaid scatter points (default 0.8).
    error_bar_width : float
        Line width for whiskers/caps (box) or error bars (bar) (default 2).
    error_bar_darker_factor : float
        How much darker the whiskers/caps or error bars are relative to face color (0 < f <= 1).
    """
    # Ensure we have an Axes
    if ax is None:
        fig, ax = plt.subplots()

    n = len(data)
    if n == 0:
        return ax

    # Default colors
    if colors is None:
        colors = [(0, 0, 0, 1.0)] * n
    if len(colors) != n:
        raise ValueError(f"colors must have length {n}, got {len(colors)}")

    x = np.arange(n)
    jitter = width * 0.4

    if style == 'box':
        flierprops = dict(
            marker='o',
            markerfacecolor='none',    
            markeredgecolor='gray',  
            markersize=4,
            linestyle='none',
            alpha=0.5              
        )

        bp = ax.boxplot(
            data,
            positions=x,
            widths=width,
            patch_artist=True,
            boxprops=dict(linewidth=1),
            whiskerprops=dict(linewidth=error_bar_width),
            capprops=dict(linewidth=error_bar_width),
            medianprops=dict(linewidth=1),
            flierprops=flierprops
        )     
        
        # color boxes
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_edgecolor(col)
        # whiskers and caps darker
        darker_colors = []
        for col in colors:
            r, g, b, a = col
            darker = (r * error_bar_darker_factor,
                      g * error_bar_darker_factor,
                      b * error_bar_darker_factor,
                      a)
            darker_colors.extend([darker, darker])
        for whisker, dc in zip(bp['whiskers'], darker_colors):
            whisker.set_color(dc)
            whisker.set_linewidth(error_bar_width)
        for cap, dc in zip(bp['caps'], darker_colors):
            cap.set_color(dc)
            cap.set_linewidth(error_bar_width)
        # medians same color as box edge
        for median, col in zip(bp['medians'], colors):
            median.set_color(col)
            median.set_linewidth(1)

    elif style == 'bar':
        # compute means & SEM
        means = [np.mean(g) for g in data]
        sems  = [np.std(g, ddof=1)/np.sqrt(len(g)) for g in data]
        # draw bars
        ax.bar(
            x,
            means,
            width=width,
            color=colors,
            edgecolor=colors,
            linewidth=1
        )
        # darker SEM error bars
        for xi, mean, sem, col in zip(x, means, sems, colors):
            r, g, b, a = col
            dc = (r * error_bar_darker_factor,
                  g * error_bar_darker_factor,
                  b * error_bar_darker_factor,
                  a)
            ax.errorbar(
                xi,
                mean,
                yerr=sem,
                fmt='none',
                capsize=error_bar_width,
                capthick=error_bar_width, 
                elinewidth=error_bar_width,
                ecolor=dc
            )

        # overlay scatter
        for xi, group, col in zip(x, data, colors):
            r, g, b, _ = col
            scat_col = (r, g, b, scatter_alpha)
            jit = (np.random.rand(len(group)) - 0.5) * jitter
            ax.scatter(xi + jit, group, color=scat_col, s=10)
    else:
        raise ValueError("style must be 'box' or 'bar'")

    # set tick labels
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=4.5)

    return ax

def plotLine(unique_omits, performance, ax=None):
    """
    Plot, for each kd>0, the difference in avg success trials relative to kd=0,
    across omission levels.
    """
    if ax is None:
        fig, ax = plt.subplots()

    baseline = {
        omit: np.mean(performance.get((0, omit), [0]))
        for omit in unique_omits
    }

    # 2) kd levels > 0
    all_kds = sorted({kd for (kd, _) in performance.keys()})
    kd_levels = [kd for kd in all_kds if kd != 0]
    kd_colors = [plt.get_cmap('tab10')(i) for i in range(len(kd_levels))]

    # 3) For each kd>0, compute diffs and plot a line
    for j, kd in enumerate(kd_levels):
        diffs = [
            np.mean(performance.get((kd, omit), [0])) - baseline[omit]
            for omit in unique_omits
        ]
        ax.plot(
            np.arange(len(unique_omits)),
            diffs,
            marker='o',
            linewidth=2,
            label=f"kd={kd}",
            color=kd_colors[j]
        )

    # 4) Formatting
    ax.set_xticks(np.arange(len(unique_omits)))
    ax.set_xticklabels([f"omit={o}" for o in unique_omits], fontsize=10)

    return ax

    


def fft(td_error, sample_rate):
    all_magnitudes = []

    for i in range(td_error.shape[0]):
        signal = td_error[i]
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)

        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]

        magnitudes = np.abs(fft_vals)
        magnitudes[1:] *= 2
        all_magnitudes.append(magnitudes)

    all_magnitudes = np.stack(all_magnitudes)

    return freqs, all_magnitudes


def get_amplitude(signal, window=None):
    """Calculate the amplitude of a signal."""
    if window is not None:
        # print("Signal before windowing:", signal)
        signal = signal[:,window[0]:window[1]]
        # print("Signal after windowing:", signal)

    maxPerTrial = np.max(signal, axis=1)
    minPerTrial = np.min(signal, axis=1)
    # Use the maximum absolute value for amplitude
    # If max is greater than min, use max, otherwise use min
    amp = np.where(np.abs(maxPerTrial) >= np.abs(minPerTrial),maxPerTrial,minPerTrial)
    return amp


def load_recorder_data(recorder,
                tds_PD=None, tds_TD=None,
                dt=0.1,
                pre_steps=20, post_steps=30, cue_duration=0.5,
                tau_on: float = 0.01, tau_off: float = 0.1, 
                save: bool = False, save_path: str = "session-summary.png"):
    
    # Load data from the recorder
    td_errors = np.array(recorder.td_errors)[1:]
    td_pid_errors = np.array(recorder.td_pid_errors)[1:]
    licks     = np.array(recorder.licks)[1:]
    tones     = np.array(recorder.tones)[1:]
    omissions = np.array(recorder.omissions)[1:]

    rewards = np.array(recorder.rewards)[1:]
    losses  = np.array(recorder.losses)[1:]
    dones   = np.array(recorder.dones)[1:]

    p_step = np.array(recorder.p)[1:]
    d_step = np.array(recorder.d)[1:]
    i_step = np.array(recorder.i)[1:]
    kp_step = np.array(recorder.kp)[1:]
    ki_step = np.array(recorder.ki)[1:]
    kd_step = np.array(recorder.kd)[1:]
    p_update_step = kp_step * p_step
    i_update_step = ki_step * i_step
    d_update_step = kd_step * d_step
    update_step = kp_step * p_step + ki_step * i_step + kd_step * d_step
    

    # Get trial based history
    trial_idx    = np.array(recorder.trial_idx)[1:]
    num_trials   = trial_idx.max() + 1
    reward_history = [rewards[trial_idx == t].sum() for t in range(num_trials)]

    # Get PID parameters history
    p_history = [p_step[trial_idx == t].mean() for t in range(num_trials)]
    d_history = [d_step[trial_idx == t].mean() for t in range(num_trials)]
    i_history = [i_step[trial_idx == t].mean() for t in range(num_trials)]
    kp_history = [kp_step[trial_idx == t].mean() for t in range(num_trials)]
    ki_history = [ki_step[trial_idx == t].mean() for t in range(num_trials)]
    kd_history = [kd_step[trial_idx == t].mean() for t in range(num_trials)]
    p_kp_history = [p_update_step[trial_idx == t].mean() for t in range(num_trials)]
    i_ki_history = [i_update_step[trial_idx == t].mean() for t in range(num_trials)]
    d_kd_history = [d_update_step[trial_idx == t].mean() for t in range(num_trials)]

    # Align cue_licks and TD errors to the cue
    error = td_errors
    pid_error = td_pid_errors
    cue_licks = get_traces(licks, tones, pre_steps, post_steps)
    cue_error   = get_traces(error, tones, pre_steps, post_steps)
    cue_pid_error = get_traces(pid_error, tones, pre_steps, post_steps)
    cue_omissions = get_traces(omissions, tones, pre_steps, post_steps)

    # Count anticipatory licks (lick number during cue window)
    cue_steps = int(cue_duration / dt)
    trial_anticipatory_licks = np.sum(cue_licks[:, pre_steps:pre_steps + cue_steps], axis=1)

    # Get TD signal during the cue window
    trial_TD_amplitude = get_amplitude(cue_error, window=(pre_steps, pre_steps + cue_steps))
    trial_pid_TD_amplitude = get_amplitude(cue_pid_error, window=(pre_steps, pre_steps + cue_steps))

    # Get EPLHb output
    if hasattr(recorder, 'eplhb_out'):
        animal_sign_index = [np.mean(sign_idx) for sign_idx in recorder.eplhb_sign_index[1:]]
        cue_animal_sign_index = get_traces(animal_sign_index, tones, pre_steps, post_steps)
        trial_animal_sign_index = get_amplitude(cue_animal_sign_index, window=(pre_steps, pre_steps + cue_steps))

        eplhb_out = np.array(recorder.eplhb_out)[1:]
        eplhb_coeff = np.array(recorder.eplhb_coeff)[1:]
        cue_EPLHb_output = get_traces(eplhb_out, tones, pre_steps, post_steps)
        cue_EPLHb_coeff = get_traces(eplhb_coeff, tones, pre_steps, post_steps)

        # (unfinished) record eplhb_weight if record_eplhb_weight == True
        # 1. take mean of the weight
        # print(recorder.eplhb_weights)
        # print(mean weights)
        # print(animal_sign_index)
    else:
        animal_sign_index = None
        trial_animal_sign_index = None
        eplhb_out = None
        eplhb_coeff = None
        cue_EPLHb_output = None
        cue_EPLHb_coeff = None 

    # ----------------------------------------------------------------------
    # Fill tds_PD and tds_TD with zeros if not provided
    if tds_PD is None:
        tds_PD = np.zeros_like(cue_error)
    if tds_TD is None:
        tds_TD = np.zeros_like(cue_error)
    
    # convert to numpy arrays if not already
    if not isinstance(cue_licks, np.ndarray):
        cue_licks = np.array(cue_licks)
    if not isinstance(cue_error, np.ndarray):
        cue_error = np.array(cue_error)
    if not isinstance(cue_pid_error, np.ndarray):
        cue_pid_error = np.array(cue_pid_error)
    if not isinstance(tds_PD, np.ndarray):
        tds_PD = np.array(tds_PD)
    if not isinstance(tds_TD, np.ndarray):
        tds_TD = np.array(tds_TD)
    if not isinstance(cue_omissions, np.ndarray):
        cue_omissions = np.array(cue_omissions)

    # time axis from -1s to +2s at 0.1s steps
    dt = 0.1
    max_trial_steps = pre_steps + post_steps
    t_axis = np.linspace(-pre_steps * dt, (max_trial_steps - pre_steps) * dt, max_trial_steps + 1)
    trial_axis = np.arange(num_trials)+1

    # build DA kernel
    t_kernel = np.arange(0, 1.0, dt)  # 0–1 s
    kernel = np.exp(-t_kernel/tau_off) - np.exp(-t_kernel/tau_on)
    kernel /= np.sum(kernel)  # normalize area to 1
    # convolve with DA kernel
    DAs = np.stack([
        np.convolve(trial_tds, kernel, mode='full')[:cue_error.shape[1]]
        for trial_tds in cue_error
    ], axis=0)

    # Build output dictionary
    output_dict = {
        'num_trials': num_trials,
        'reward_history': reward_history,
        'kp_history': kp_history,
        'ki_history': ki_history,
        'kd_history': kd_history,
        'update_step': update_step,
        'cue_licks': cue_licks,
        'cue_error': cue_error,
        'cue_omissions': cue_omissions,
        'trial_anticipatory_licks': trial_anticipatory_licks,
        'trial_TD_amplitude': trial_TD_amplitude,
        'trial_pid_TD_amplitude': trial_pid_TD_amplitude,
        'td_pid_errors': td_pid_errors,
        't_axis': t_axis,
        'trial_axis': trial_axis,
        'DAs': DAs,
        'animal_sign_index': animal_sign_index,
        'trial_animal_sign_index': trial_animal_sign_index,
        'eplhb_out': eplhb_out,
        'eplhb_coeff': eplhb_coeff,
        'cue_EPLHb_output': cue_EPLHb_output,
        'cue_EPLHb_coeff': cue_EPLHb_coeff,
    }
    return output_dict


def plot_figure(recorder,
                tds_PD=None, tds_TD=None,
                dt=0.1,
                pre_steps=20, post_steps=30, cue_duration=0.5,
                tau_on: float = 0.01, tau_off: float = 0.1, 
                save: bool = False, save_path: str = "session-summary.png",
                show: bool = False):
    

    # load recorder data

    output_dict = load_recorder_data(
        recorder,
        tds_PD=tds_PD,
        tds_TD=tds_TD,
        dt=dt,
        pre_steps=pre_steps,
        post_steps=post_steps,
        cue_duration=cue_duration,
        tau_on=tau_on,
        tau_off=tau_off
    )

    # Plotting
    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 1.2], height_ratios=[1, 1, 0.7], wspace=0.2, hspace=0.4)

    # top‐left: Reward per trial
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(output_dict['trial_axis'],output_dict['reward_history'])
    ax0.set_title(f"Reward per trial")
    ax0.set_xlabel("Trial")
    ax0.set_ylabel("Total Reward")

    # top-middle: Kp, Kd, Ki trace or EPLHb
    if hasattr(recorder, 'eplhb_out'):
        # if recorder have eplhb_out, plot it
        ax1 = fig.add_subplot(gs[0, 1])
        plotSEM(output_dict['t_axis'], output_dict['cue_EPLHb_output'], omissions=output_dict['cue_omissions'], label="EPLHb output", color='tab:green', ax=ax1, alpha=0.1)
        ax1.set_title(f"EPLHb output")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("EPLHb output")
        ax1.legend(loc='upper left', fontsize=10, frameon=False)
    else: 
        # top-middle: Kp, Kd, Ki trace
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(output_dict['trial_axis'],output_dict['kp_history'], label='Kp', color='tab:blue', alpha=0.7)
        ax1.plot(output_dict['trial_axis'],output_dict['ki_history'], label='Ki', color='tab:orange', alpha=0.7)
        ax1.plot(output_dict['trial_axis'],output_dict['kd_history'], label='Kd', color='tab:red', alpha=0.7)
        ax1.set_title(f"PID Parameters")
        ax1.set_xlabel("Trial")
        ax1.set_ylabel("Parameter Value")
        ax1.legend(loc='upper left', fontsize=10, frameon=False)

    # middle‐left: Lick raster (scatter)
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(output_dict['num_trials']):
        lick_times = output_dict['t_axis'][output_dict['cue_licks'][i] == 1]
        cur_omissions = output_dict['t_axis'][output_dict['cue_omissions'][i] == 1]
        has_omission = len(cur_omissions) > 0
        if has_omission:
            ax2.scatter(lick_times, np.ones_like(lick_times)*(i+1),
                        color='tab:pink', s=10, marker='o', alpha=0.8, edgecolor='none')
        else:
            ax2.scatter(lick_times, np.ones_like(lick_times)*(i+1),
                        color='tab:blue', s=10, marker='o', alpha=0.8, edgecolor='none')
    # mark cue window
    ax2.fill_betweenx([0, output_dict['num_trials']+1], 0, 0.5, color='tab:orange', alpha=0.2, edgecolor='None')
    ax2.set_title(f"Lick raster")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Trial")
    ax2.set_xlim(output_dict['t_axis'][0], output_dict['t_axis'][-1])
    ax2.set_ylim(0.5, output_dict['num_trials']+1)

    # middle‐middle: average TD error + simulated DA signal
    ax3 = fig.add_subplot(gs[1, 1])
    plotSEM(output_dict['t_axis'], output_dict['cue_error'], omissions=None, label="TD Error", color='tab:blue', ax=ax3, alpha=0.1)
    # Plot cue_pid_error if it exists in output_dict
    if 'cue_pid_error' in output_dict:
        plotSEM(output_dict['t_axis'], output_dict['cue_pid_error'], omissions=None, label="PID TD Error", color='tab:orange', ax=ax3, alpha=0.1)

    # plotSEM(t_axis, DAs, label="DA Signal", color='tab:green', ax=ax3, alpha=0.2)
    # shade cue
    ymin, ymax = ax3.get_ylim()
    ax3.fill_betweenx([ymin, ymax], 0, 0.5, color='tab:orange', alpha=0.2, edgecolor='None')
    ax3.set_ylim(ymin, ymax)
    ax3.set_title(f"TD error vs time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("TD Error")
    ax3.plot([], [], color='tab:blue', label="TD Error", linewidth=1) # labeling
    ax3.plot([], [], color='tab:orange', label="PID TD Error", linewidth=1) # labeling
    ax3.legend(loc='upper left', fontsize=10, frameon=False)

    # bottom left: lick number during cue per trial (i.e. anticipatory licking)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(output_dict['trial_axis'], output_dict['trial_anticipatory_licks'], color='tab:blue')
    ax4.set_title("Anticipatory licks")
    ax4.set_xlabel("Trial")
    ax4.set_ylabel("Lick Count")

    # bottom right: amplitude of TD error during cue
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(output_dict['trial_axis'], output_dict['trial_pid_TD_amplitude'], color='tab:orange', label='PID TD Amplitude', linewidth=0.7)
    ax5.plot(output_dict['trial_axis'], output_dict['trial_TD_amplitude'], color='tab:blue', label='TD Amplitude', linewidth=1)
    if hasattr(recorder, 'eplhb_sign_index'):
        ax5b = ax5.twinx()
        ax5b.plot(output_dict['trial_axis'], output_dict['trial_animal_sign_index'], color='tab:green', label='Animal Sign Index')
        ax5b.set_ylabel("Animal Sign Index", color='tab:green')
        ax5b.tick_params(axis='y', labelcolor='tab:green')
        # Use scientific notation for tick values to avoid overlap
        from matplotlib.ticker import ScalarFormatter
        ax5b.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax5b.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax5.set_title("TD error during cue & animal sign index")
    else:
        ax5.set_title("TD error during cue")
    ax5.legend(loc='upper left', fontsize=10, frameon=False)
    ax5.set_xlabel("Trial")
    ax5.set_ylabel("Amplitude")
    ax5.legend(loc='upper left', fontsize=10, frameon=False)

    # bottom middle: FFT
    # ax5 = fig.add_subplot(gs[2, 1])
    # td_freqs, td_magnitudes = fft(td_error = tds_TD, sample_rate = 50)
    # pd_freqs, pd_magnitudes = fft(td_error = tds_PD, sample_rate = 50)
    # plotSEM(pd_freqs, pd_magnitudes,label='PD', color='tab:blue', ax=ax5, alpha=0.3)
    # plotSEM(td_freqs, td_magnitudes, omissions = None, label='TD', color='tab:orange', ax=ax5, alpha=0.3)
    # ax5.set_xlabel("Frequency")
    # ax5.set_ylabel("Magnitude")
    # ax5.set_title("FFT of TD Errors")
    # ax5.legend(loc='upper right', fontsize=10, frameon=False)

    # the new heatmap, spanning both rows in column 3
    ax_heat = fig.add_subplot(gs[:, 2])
    data = output_dict['cue_error'] 
    im = ax_heat.imshow(
        data,
        aspect='auto',
        interpolation='nearest',
        extent=[output_dict['t_axis'][0], output_dict['t_axis'][-1], output_dict['num_trials'], 1],
        cmap='RdBu_r',
        vmin=-np.max(np.abs(data)),
        vmax=np.max(np.abs(data)),
    )
    ax_heat.set_title("Heatmap")
    ax_heat.set_xlabel("Time (s)")
    ax_heat.set_ylabel("Trial")
    ax_heat.invert_yaxis()   # so Trial 1 is at top
    fig.colorbar(im, ax=ax_heat, orientation='vertical', label='TD Error')

    # Tighten layout and show
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.07)

    # Show the figure if requested
    if show: plt.show()

    # Save the figure if requested
    if save: fig.savefig(save_path, dpi=300, bbox_inches='tight')