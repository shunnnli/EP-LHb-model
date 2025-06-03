import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle

def plotSEM(x, y, label=None, color=None, ax=None, alpha=0.2):
    """Plot with shaded error margin."""
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    if label is None:
        label = ax._get_lines.get_next_label()
    
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color,
                     edgecolor='None', label='_nolegend_')
    

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
    
def plot_figure(licks, tds, reward_history, loss_history, 
                tds_PD=None, tds_TD=None,
                timestamp = None, time_range=(-1,4),
                pre_steps: int = 10, post_steps: int = 20, dt: float = 0.1,
                signal_fs: float = 10.0,
                tau_on: float = 0.01, tau_off: float = 0.1, controller="TD"):
    
    # Fill tds_PD and tds_TD with zeros if not provided
    if tds_PD is None:
        tds_PD = np.zeros_like(tds)
    if tds_TD is None:
        tds_TD = np.zeros_like(tds)
    
    # convert to numpy arrays if not already
    if not isinstance(licks, np.ndarray):
        licks = np.array(licks)
    if not isinstance(tds, np.ndarray):
        tds = np.array(tds)
    if not isinstance(tds_PD, np.ndarray):
        tds_PD = np.array(tds_PD)
    if not isinstance(tds_TD, np.ndarray):
        tds_TD = np.array(tds_TD)

    # time axis from -1s to +2s at 0.1s steps
    if timestamp is None:
        # dt = 0.1
        max_trial_steps = pre_steps + post_steps + 1
        timestamp = np.linspace(-pre_steps*dt, (max_trial_steps-pre_steps)*dt, max_trial_steps)
        # t0, t1    = time_range
        # n_pts     = int(round((t1 - t0) * signal_fs)) + 1
        # timestamp = t0 + np.arange(n_pts) / signal_fs

    # number of trials
    num_trials = licks.shape[0]

    # build DA kernel
    t_kernel = np.arange(0, 1.0, 1/signal_fs)  # 0–1 s
    kernel = np.exp(-t_kernel/tau_off) - np.exp(-t_kernel/tau_on)
    kernel /= np.sum(kernel)  # normalize area to 1
    # convolve with DA kernel
    DAs = np.stack([
        np.convolve(trial_tds, kernel, mode='full')[:tds.shape[1]]
        for trial_tds in tds
    ], axis=0)

    # Plotting
    fig = plt.figure(figsize=(14, 8))
    gs  = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 1.2], height_ratios=[1, 1, 0.7], wspace=0.2, hspace=0.4)

    # top‐left: Reward per trial
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(reward_history)
    ax0.set_title(f"Reward per trial ({controller})")
    ax0.set_xlabel("Trial")
    ax0.set_ylabel("Total Reward")

    # top‐middle: Loss per trial
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(loss_history)
    ax1.set_title(f"Loss per trial ({controller})")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("MSE Loss")

    # middle‐left: Lick raster (scatter)
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(num_trials):
        lick_times = timestamp[licks[i] == 1]
        ax2.scatter(lick_times, np.ones_like(lick_times)*(i+1),
                    color='tab:pink', s=20, marker='o', alpha=0.8, edgecolor='none')
    # mark cue window
    ax2.fill_betweenx([0, num_trials+1], 0, 0.5, color='tab:orange', alpha=0.2, edgecolor='None')
    ax2.set_title(f"Lick raster ({controller})")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Trial")
    ax2.set_xlim(timestamp[0], timestamp[-1])
    ax2.set_ylim(0.5, num_trials+1)

    # middle‐middle: average TD error + simulated DA signal
    ax3 = fig.add_subplot(gs[1, 1])
    plotSEM(timestamp, tds, label="TD Error", color='tab:blue', ax=ax3, alpha=0.2)
    plotSEM(timestamp, DAs, label="DA Signal", color='tab:green', ax=ax3, alpha=0.2)
    # shade cue
    ymin, ymax = ax3.get_ylim()
    ax3.fill_betweenx([ymin, ymax], 0, 0.5, color='tab:orange', alpha=0.2, edgecolor='None')
    ax3.set_ylim(ymin, ymax)
    ax3.set_title(f"TD error vs time ({controller})")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("TD Error")
    ax3.legend(loc='upper left', fontsize=10, frameon=False)

    # bottom left: raw tds
    ax4 = fig.add_subplot(gs[2, 0])
    plotSEM(timestamp, tds_TD, label="TD Only", color='tab:blue', ax=ax4, alpha=0.3)
    plotSEM(timestamp, tds_PD, label="With PD", color='tab:red', ax=ax4, alpha=0.3)
    ymin, ymax = ax4.get_ylim()
    ax4.fill_betweenx([ymin, ymax], 0, 0.5, color='tab:orange', alpha=0.2, edgecolor='None')
    ax4.set_ylim(ymin, ymax)
    ax4.set_title("TD vs PD TD Errors")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("TD Error")
    ax4.legend(loc='upper left', fontsize=10, frameon=False)

    # bottom middle: FFT
    ax5 = fig.add_subplot(gs[2, 1])
    td_freqs, td_magnitudes = fft(td_error = tds_TD, sample_rate = 50)
    pd_freqs, pd_magnitudes = fft(td_error = tds_PD, sample_rate = 50)
    plotSEM(pd_freqs, pd_magnitudes, label='PD', color='tab:blue', ax=ax5, alpha=0.3)
    plotSEM(td_freqs, td_magnitudes, label='TD', color='tab:orange', ax=ax5, alpha=0.3)
    ax5.set_xlabel("Frequency")
    ax5.set_ylabel("Magnitude")
    ax5.set_title("FFT of TD Errors")
    ax5.legend(loc='upper right', fontsize=10, frameon=False)

    # the new heatmap, spanning both rows in column 3
    ax_heat = fig.add_subplot(gs[:, 2])
    data = tds
    im = ax_heat.imshow(
        data,
        aspect='auto',
        interpolation='nearest',
        extent=[timestamp[0], timestamp[-1], num_trials, 1],
        cmap='RdBu_r',
        vmin=-np.max(np.abs(data)),
        vmax=np.max(np.abs(data)),
    )
    ax_heat.set_title("Heatmap")
    ax_heat.set_xlabel("Time (s)")
    ax_heat.set_ylabel("Trial")
    ax_heat.invert_yaxis()   # so Trial 1 is at top
    cbar = fig.colorbar(im, ax=ax_heat, orientation='vertical', label='TD Error')

    # Tighten layout and show
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.07)
    plt.show()