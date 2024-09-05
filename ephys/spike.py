# spike.align(time_spike, time_event, window=[-5, 5])
# spike.count_spike(time_spike, time_event, window=[-0.5, 0.5])
# spike.plot(time_spike, time_event, type_event, window=[-5, 5], reorder=1, binsize=0.01, resolution=10)

import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt

from .utils import tprint, finder


class Spike:
    def __init__(self, path):
        self.path = path

        tprint(f"Loading {self.path}")
        temp = loadmat(path, simplify_cells=True)
        key_pandas = {'vr', 'sync', 'trial', 'nidq'}
        data = {
            key.lower(): pd.DataFrame(value) if key in key_pandas else value
            for key, value in temp.items()
            if not key.startswith('__')
        }
        self.__dict__.update(data)
    
    def __repr__(self):
        """
        Return a string representation of the Spike object structure
        """
        result = []
        for key, value in self.__dict__.items():
            if key.startswith('__'):
                continue
            result.append(key)
            if isinstance(value, pd.DataFrame):
                for col in value.columns:
                    result.append(f"    {col}: {value[col].shape}")
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        result.append(f"    {k}: {v.shape}")
                    elif isinstance(v, list):
                        result.append(f"    {k}: {len(v)}")
                    elif isinstance(v, dict):
                        result.append(f"    {k}:")
                    elif isinstance(v, (int, float)):
                        result.append(f"    {k}: {v}")
            elif isinstance(value, str):
                result.append(f"    {value}")
        return "\n".join(result)
    
    @property
    def spike_df(self):
        """
        Return a dataframe of spike time and unit id.
        """
        spk = np.concatenate(self.spike['time'])
        idx = np.concatenate([np.full_like(x, i) for i, x in enumerate(self.spike['time'])])
        spkidx = np.argsort(spk)
        return pd.DataFrame({'time': spk[spkidx], 'unit': idx[spkidx]})

    def spike_bin(self, binsize=0.10, kernel_size=None):
        """
        Return an array of binned spikes by time and unit id for decoding analysis.

        * If kernel_size is None, return the binned spikes without convolution.
        * kernel_size should be odd number.
        """
        spk = self.spike_df
        xedges = np.arange(spk.time.min(), spk.time.max(), binsize)
        yedges = np.arange(spk.unit.max() + 2)
        xcenter = xedges[:-1] + binsize/2
        
        spks, _, _ = np.histogram2d(spk.time.values, spk.unit.values, bins=(xedges, yedges))

        if kernel_size is None:
            return xcenter, yedges[:-1], spks
        
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        kernel = np.ones(kernel_size)
        spks_conv = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=0, arr=spks)

        return xcenter[kernel_size//2:-kernel_size//2 + 1], yedges[:-1], spks_conv
    
    @property
    def time_nidq(self):
        chans = np.unique(self.nidq.chan.values)
        types = np.unique(self.nidq.type.values)

        time_nidq = {}
        for chan in chans:
            for type in types:
                temp = self.nidq.query(f'chan == {chan} and type == {type}')
                if temp.empty:
                    continue
                time_nidq[f'{chan}_{type}'] = temp['time_imec'].values
        
        return time_nidq

    def plot(self):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        trial_type = list(self.time_nidq.keys())
        unit_ids = list(range(self.spike['n_unit']))

        def update_plot():
            selected_type = type_var.get()
            selected_unit = int(unit_var.get())
            window = [float(window_start.get()), float(window_end.get())]
            reorder = int(reorder_var.get())
            binsize = float(binsize_var.get())
            sigma = int(sigma_var.get())

            time_spike = self.spike['time'][selected_unit]
            time_trial = self.time_nidq[selected_type]

            plot_raster_psth(time_spike, time_trial, window=window, reorder=reorder, binsize=binsize, sigma=sigma, fig=fig)
            canvas.draw()

        root = tk.Tk()
        root.title("Spike Plot GUI")

        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        params = [
            ("Trial Type:", "type_var", trial_type[0], trial_type),
            ("Unit ID:", "unit_var", "0", unit_ids),
            ("Window Start:", "window_start", "-5"),
            ("Window End:", "window_end", "5"),
            ("Reorder:", "reorder_var", "1"),
            ("Bin Size:", "binsize_var", "0.01"),
            ("Sigma:", "sigma_var", "10")
        ]

        for label, var_name, default, values in params:
            ttk.Label(frame, text=label).grid(column=0, row=frame.grid_size()[1], sticky=tk.W)
            var = tk.StringVar(value=default)
            if values:
                widget = ttk.Combobox(frame, textvariable=var, values=values)
            else:
                widget = ttk.Entry(frame, textvariable=var)
            widget.grid(column=1, row=frame.grid_size()[1]-1, sticky=(tk.W, tk.E))
            setattr(frame, var_name, var)

        ttk.Button(frame, text="Update Plot", command=update_plot).grid(column=1, row=frame.grid_size()[1], sticky=tk.E)

        fig = plt.Figure(figsize=(10, 8))
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().grid(row=1, column=0)

        update_plot()
        root.mainloop()

def align(time_spike, time_event, window=[-5, 5]):
    time_aligned = [time_spike[(time_spike >= te + window[0]) & (time_spike <= te + window[1])] - te if not np.isnan(te) else np.array([]) for te in time_event]
    return np.array(time_aligned, dtype=object)


def count_spike(time_spike, time_event, window=[-0.5, 0.5]):
    def count_spk(te):
        if np.isnan(te):
            return np.nan        
        return np.sum((time_spike >= te + window[0]) & (time_spike <= te + window[1])).astype(float)
    return np.vectorize(count_spk)(time_event)


def get_raster(time_aligned, type_event, reorder=1, line=True):
    if type_event.dtype == float:
        type_event = type_event.astype(int)
    n_type = np.max(type_event) + 1
    n_trial_type = np.bincount(type_event)
    cum_trial = np.concatenate([0, np.cumsum(n_trial_type)], axis=None)
    
    x = np.empty(n_type, dtype=object)
    y = np.empty(n_type, dtype=object)
    
    if time_aligned.size == 0:
        return x, y
    
    n_spike = np.vectorize(len)(time_aligned)
    for i_type in range(n_type):
        in_trial = type_event == i_type
        n_spike_type = n_spike[in_trial]
        x_temp = np.concatenate(time_aligned[in_trial])
        y_temp = np.repeat(np.arange(1, n_trial_type[i_type] + 1, dtype=float) if reorder else np.arange(1, len(type_event) + 1, dtype=float)[in_trial], n_spike_type) + cum_trial[i_type]

        if line:
            x[i_type] = np.column_stack((x_temp, x_temp, np.full_like(x_temp, np.nan))).ravel()
            y[i_type] = np.column_stack((y_temp, y_temp + 1, np.full_like(y_temp, np.nan))).ravel()
        else:
            x[i_type] = x_temp
            y[i_type] = y_temp
    
    return x, y


def get_psth(time_aligned, type_event, binsize=0.01, sigma=10, window=[-5, 5]):
    bins = np.arange(window[0], window[1], binsize)
    t = bins[:-1] + binsize/2
    types = np.unique(type_event)
    n_type = len(types)
    
    # make gaussian window
    # acutal sigma will be binsize * sigma
    # if binsize is 0.01 and sigma=10, sigma_actual will be 0.1 s
    gauss_window = signal.gaussian(5*sigma, sigma)
    gauss_window /= np.sum(gauss_window)
    
    bar = np.zeros((n_type, len(t)))
    conv = np.zeros_like(bar)
    for i, i_type in enumerate(types):
        in_trial = type_event == i_type
        time_spike_type = np.concatenate(time_aligned[in_trial])
        y, _ = np.histogram(time_spike_type, bins)
        bar[i, :] = y / binsize / np.sum(in_trial)
        conv[i, :] = np.convolve(bar[i, :], gauss_window, 'same')
    
    return t, bar, conv


def get_raster_psth(time_spike, time_event, type_event=None, 
         window=[-5, 5], reorder=1, binsize=0.01, sigma=10, line=True):
        
    if type_event is None:
        type_event = np.zeros(len(time_event))

    in_event = ~np.isnan(type_event)
    time_event = time_event[in_event]
    type_event = type_event[in_event].astype('int64')
    type_unique, type_index = np.unique(type_event, return_inverse=True)

    
    time_aligned = align(time_spike, time_event, window)
    x, y = get_raster(time_aligned, type_index, reorder, line)
    t, bar, conv = get_psth(time_aligned, type_index, binsize, sigma, window)
    
    return {'x': x, 'y': y, 'type': type_unique}, {'t': t, 'bar': bar, 'conv': conv, 'type': type_unique}


def plot_raster_psth(time_spike, time_event, type_event=None, window=[-5, 5], reorder=1, binsize=0.01, sigma=10, fig=None):
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    else:
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
    
    window_raw = np.array(window) + np.array([-binsize*sigma*3, binsize*sigma*3])
    raster, psth = get_raster_psth(time_spike, time_event, type_event, window=window_raw, reorder=reorder, binsize=binsize, sigma=sigma)

    cmap = [(0, 0, 0)] + list(plt.get_cmap('tab10').colors)
    for i, (x, y, conv) in enumerate(zip(raster['x'], raster['y'], psth['conv'])):
        color = cmap[i % len(cmap)]
        ax1.plot(x, y, color=color, linewidth=0.5)
        ax2.plot(psth['t'], conv, color=color)
    
    ylim_raster = [0, max(np.nanmax(y) for y in raster['y'])]
    ylim_psth = [0, np.nanmax(psth['conv']) * 1.1]
    for ax, ylim in [(ax1, ylim_raster), (ax2, ylim_psth)]:
        ax.vlines(0, 0, ylim[1], color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlim(window)
        ax.set_ylim(ylim)
        ax.spines[['top', 'right']].set_visible(False)

    ax1.set_ylabel('Trial')
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_xlabel('Time (s)')

    fig.tight_layout()
    return fig


def get_latency(spike, event_onset, event_offset, duration=0.08, offset=0.3, min_latency=0.02, prob=0.05):
    assert(len(event_onset) == len(event_offset))
    n_event = len(event_onset)

    spike_event = [spike[(spike >= event_onset[0]) & (spike < event_onset[0] + duration)] - event_onset[0]]
    spike_base = []
    for i_event in range(n_event - 1):
        base = np.arange(event_offset[i_event] + offset, event_onset[i_event + 1], duration)
        n_base_temp = len(base) - 1
        if n_base_temp == 0:
            continue

        spike_event.append(spike[(spike >= event_onset[i_event+1]) & (spike < event_onset[i_event+1] + duration)] - event_onset[i_event+1])

        base_index = np.random.permutation(np.arange(n_base_temp))[:min(n_base_temp, 50)]
        for i_base in base_index:
            spike_base.append(spike[(spike >= base[i_base]) & (spike < base[i_base + 1])] - base[i_base])

    spike_event_all = np.sort(np.concatenate(spike_event))
    count_event = np.arange(len(spike_event_all)) / len(spike_event_all)
    spike_base_all = np.sort(np.concatenate(spike_base))
    count_base = np.arange(len(spike_base_all)) / len(spike_base_all)

    return {'time_event': spike_event_all, 'count_event': count_event, 
            'time_base': spike_base_all, 'count_base': count_base}


if __name__ == '__main__':
    path = finder(path="C:\SGL_DATA", msg='Select a session file', pattern=r'.mat$')
    spike = Spike(path)

    # interactive plot
    # spike.plot()

    # plot raster and psth
    time_spike = spike.spike['time'][0]
    time_event = spike.nidq.query('chan == 2 and type == 1')['time_imec'].values
    plot_raster_psth(time_spike, time_event)
    plt.show()
