import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio

from .spikeglx import read_meta, read_analog, read_digital, get_uV_per_bit, get_channel_idx
from .utils import finder, confirm, savemat_safe, tprint, sync


def run_ks4(path=None, settings=None):
    try:
        from kilosort import run_kilosort
    except ImportError:
        raise ImportError("The module 'kilosort' is not installed. Please install it before running this function.")

    if settings is None:
        settings = {}

    fns = finder(path, 'ap.bin$', multiple=True)
    for fn in fns:
        # check if the kilosort folder already exists
        if os.path.exists(os.path.join(os.path.dirname(fn), 'kilosort4', 'params.py')):
            if not confirm("Kilosort folder already exists. Do you want to run it again?"):
                continue

        meta = read_meta(fn)
        if meta:
            probe = get_probe(meta)
            settings['n_chan_bin'] = meta['nSavedChans']
            settings['data_dir'] = os.path.dirname(fn)

            run_kilosort(settings=settings, probe=probe)


def get_probe(meta: dict) -> dict:
    """
    Create a dictionary to track probe information for Kilosort4.

    Parameters
    ----------
    meta : dict
        Dictionary containing metadata information.

    Returns
    -------
    dict
        Dictionary with the following keys, all corresponding to NumPy ndarrays:
        'chanMap': the channel indices that are included in the data.
        'xc':      the x-coordinates (in micrometers) of the probe contact centers.
        'yc':      the y-coordinates (in micrometers) of the probe contact centers.
        'kcoords': shank or channel group of each contact (not used yet, set all to 0).
        'n_chan':  the number of channels.
    """
    probe_info = {
        'chanMap': np.array([i['channel'] for i in meta['snsChanMap']['channel_map']], dtype=np.int32)[get_channel_idx(meta)],
        'xc': np.array([i['x'] for i in meta['snsGeomMap']['electrodes']], dtype=np.float32),
        'yc': np.array([i['z'] for i in meta['snsGeomMap']['electrodes']], dtype=np.float32),
        'kcoords': np.array([i['shank'] for i in meta['snsGeomMap']['electrodes']], dtype=np.float32),
        'n_chan': np.array(meta['nSavedChans'], dtype=np.float32)
    }
    return probe_info

def nearest_channel(channel_position, channel_index, count=14):
    """
    Get the indices of the nearest channels for a given channel index.

    Parameters
    ----------
    channel_position : ndarray
        Positions of channels on the probe (n_channel, 2).
    channel_index : ndarray
        Indices of channels.
    count : int, optional
        Number of nearest channels to get. Default is 14.

    Returns
    -------
    ndarray
        Indices of the nearest channels for each input channel, sorted by distance.
    """
    all_distances = np.linalg.norm(channel_position[:, np.newaxis] - channel_position, axis=2)
    return np.argsort(all_distances, axis=1)[channel_index, :count]


class Kilosort():
    def __init__(self, path=None):
        if path is None:
            fn = finder(None, 'params.py$')
            path = os.path.dirname(fn)

        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        self.path = path
        self.session = path.split(os.path.sep)[-2]
        self.sync = None
        self.nidq = None

        self.load_meta()
        self.load_kilosort()
        self.load_waveforms()
        self.load_sync()
        self.load_nidq()

    def load_meta(self):
        ops = np.load(os.path.join(self.path, 'ops.npy'), allow_pickle=True).item()
        self.data_file_path_orig = str(ops.get('filename'))
        parent_folder = os.path.dirname(self.path)
        self.data_file_path = finder(parent_folder, 'ap.bin$', ask=False)[0]

        # check if the data filename matches the original one
        if os.path.basename(self.data_file_path) != os.path.basename(self.data_file_path_orig):
            if not confirm(f"Data file name does not match original name. {self.data_file_path} != {self.data_file_path_orig}. Do you want to continue?"):
                raise ValueError("Data file name does not match original name.")

        self.meta = read_meta(self.data_file_path)
        self.n_channel = self.meta['snsApLfSy']['AP']
        self.uV_per_bit = get_uV_per_bit(self.meta)
        self.sample_rate = self.meta.get('imSampRate') or self.meta.get('niSampRate') or 1
        self.file_create_time = self.meta.get('fileCreateTime')

    def load_kilosort(self, load_all=False):
        """
        Load Kilosort data from the specified path.

        Parameters:
        -----------
        load_all : bool, optional
            If True, load all clusters. If False, load only 'good' clusters. Default is False.

        Attributes:
        -----------
        time : ndarray of object
            Spike times in seconds.
        frame : ndarray of object
            Spike times in samples.
        firing_rate : list
            Firing rates.
        position : ndarray
            Median spike positions on the probe.

        waveform : ndarray
            Mean template waveforms across the nearest 14 channels (n_unit, 61, 14).
        waveform_idx : ndarray
            Channel indices for the 14 nearest channels (not actual channel numbers).
        waveform_channel : ndarray
            Actual channel numbers corresponding to the waveform sites (n_unit, 14).
        waveform_position: ndarray
            Channel positions on the probe for the 14 nearest channels (n_unit, 14, 2).
        Vpp : ndarray
            Peak-to-peak amplitude in arbitrary units.

        n_unit : int
            Number of good units (or all units if load_all is True).

        cluster_group : ndarray
            Group labels for all clusters (good, mua, noise, or nan).

        channel_map : ndarray
            Mapping of channel indices to physical channels.
        channel_position : ndarray
            Positions of all channels on the probe.
        """
        tprint(f"Loading Kilosort data from {self.path}")

        # load spike_times, and unit_id
        frame = np.load(os.path.join(self.path, "spike_times.npy"))
        cluster = np.load(os.path.join(self.path, 'spike_clusters.npy'))
        spike_templates = np.load(os.path.join(self.path, 'spike_templates.npy'))
        
        # manual clustering information
        cluster_fn = os.path.join(self.path, 'cluster_info.tsv')
        if os.path.exists(cluster_fn):
            cluster_info = pd.read_csv(cluster_fn, sep='\t')
            self.cluster_group = cluster_info['group'].values
        else:
            tprint("No cluster_info.tsv found. Loading all clusters.")
            load_all = True
        
        if not load_all:
            good_id = cluster_info[cluster_info['group'] == 'good'].cluster_id.values
        else:
            good_id = np.unique(cluster)
        
        main_template_id = [np.bincount(spike_templates[cluster == c]).argmax() 
                            for c in good_id]

        self.n_unit = len(good_id)
        
        # Convert spike times to seconds and group by good clusters
        self.time = np.array([frame[cluster == c] / self.sample_rate for c in good_id], dtype=object)
        self.frame = np.array([frame[cluster == c] for c in good_id], dtype=object)
        
        # Calculate firing rates for good clusters
        self.firing_rate = [len(i) / (frame.max() / self.sample_rate) for i in self.time]

        # Load and calculate mean spike positions for good clusters
        spike_positions = np.load(os.path.join(self.path, 'spike_positions.npy'))
        self.position = np.array([np.median(spike_positions[cluster == c], axis=0) 
                                  for c in good_id])
    
        # Load templates and amplitudes
        templates = np.load(os.path.join(self.path, 'templates.npy'))
        amplitudes = np.load(os.path.join(self.path, 'amplitudes.npy'))
        winv = np.load(os.path.join(self.path, 'whitening_mat_inv.npy'))
        temp_unwhitened = templates @ winv
        
        # Find the channel with maximum amplitude for each template
        max_channel = (temp_unwhitened.max(axis=1) - temp_unwhitened.min(axis=1)).argmax(axis=-1)

        # Load channel map
        self.channel_map = np.load(os.path.join(self.path, 'channel_map.npy'))
        self.channel_position = np.load(os.path.join(self.path, 'channel_positions.npy'))

        # Find the best channel (waveform site) for each good cluster
        waveform_idx = max_channel[main_template_id]
        self.waveform_idx = nearest_channel(self.channel_position, waveform_idx) # (n_unit, 14)
        self.waveform_position = self.channel_position[self.waveform_idx] # (n_unit, 14, 2)

        # Map waveform sites to actual channel numbers (n_unit, 14). This is needed for extracting waveforms from raw data.
        self.waveform_channel = self.channel_map[self.waveform_idx]
        
        # Calculate mean waveforms for good clusters
        waveform = np.zeros((self.n_unit, temp_unwhitened.shape[1], 14))
        for i, c in enumerate(good_id):
            spikes = spike_templates[cluster == c]
            amplitudes_c = amplitudes[cluster == c].reshape(-1, 1, 1)
            mean_waveform = (temp_unwhitened[spikes] * amplitudes_c).mean(axis=0)

            # We don't know the exact scaling factor that was used by Kilosort, but it was approximately 10.
            waveform[i] = mean_waveform[:, self.waveform_idx[i]] / 10
        self.waveform = waveform
    
        self.Vpp = np.ptp(self.waveform, axis=(1, 2))  # peak-to-peak amplitude in arbitrary unit

    def load_waveforms(self, spk_range=(-20, 41), sample_range=(0, 30000*300)):
        """
        Load waveforms from the raw data file
        
        Parameters
        ----------
        spk_range : tuple, optional
            The range of spike times to load, in samples. Default is (-20, 41).
        sample_range : tuple, optional
            The range of samples to load. Default is (0, 30000*60).
        """
        if not os.path.exists(self.data_file_path):
            print(f"Data file {self.data_file_path} does not exist")
            return

        MAX_MEMORY = int(4e9)

        n_sample_file = self.meta['fileSizeBytes'] // (self.meta['nSavedChans'] * np.dtype(np.int16).itemsize)
        if sample_range[0] > n_sample_file:
            raise ValueError(f"Sample range {sample_range} is out of bounds for the data file {self.data_file_path}")
        if sample_range[1] > n_sample_file:
            sample_range[1] = n_sample_file

        n_sample = sample_range[1] - sample_range[0]
        n_sample_per_batch = min(int(MAX_MEMORY / self.n_channel / np.dtype(np.int16).itemsize), n_sample)
        n_batch = int(np.ceil(n_sample / n_sample_per_batch))

        spks = np.concatenate(self.frame)
        idx = np.concatenate([np.full_like(c, i) for i, c in enumerate(self.frame)])
        n_spk = len(spks)

        sort_idx = np.argsort(spks)
        spks, idx = spks[sort_idx], idx[sort_idx]
        spk_used = np.zeros(n_spk, dtype=bool)

        spk_width = spk_range[1] - spk_range[0]

        spkwav = np.full((n_spk, spk_width, 14), np.nan)
        batch_indices = np.arange(n_batch)
        batch_starts = batch_indices * n_sample_per_batch + sample_range[0]
        batch_ends = np.minimum((batch_indices + 1) * n_sample_per_batch + sample_range[0], sample_range[1])

        for i_batch, (batch_start, batch_end) in enumerate(zip(batch_starts, batch_ends)):
            tprint(f"Loading waveforms from {self.data_file_path} (batch {i_batch+1}/{n_batch})")
            data = read_analog(self.data_file_path, sample_range=(batch_start, batch_end))

            in_range = (spks >= batch_start + spk_width) & (spks < batch_end - spk_width)
            spk = spks[in_range]
            spk_idx = idx[in_range]
            spk_used[in_range] = True
            spk_no = np.where(in_range)[0]
            for i, i_spk, i_idx in zip(spk_no, spk, spk_idx):
                spkwav_temp = data[i_spk + spk_range[0] - batch_start:i_spk + spk_range[1] - batch_start, self.waveform_channel[i_idx]]
                spkwav[i] = spkwav_temp - spkwav_temp[0, :]
        
        self.waveform_raw = np.array([np.median(spkwav[(idx == i_unit) & spk_used], axis=0) 
                                      for i_unit in range(self.n_unit)])
        self.Vpp_raw = np.ptp(self.waveform_raw, axis=(1, 2))

    def load_sync(self):
        if not os.path.exists(self.data_file_path):
            print(f"Data file {self.data_file_path} does not exist")
            return

        if self.meta.get('typeThis') != 'imec':
            print(f"Unsupported data type: {self.meta.get('typeThis')}")
            return

        tprint(f"Loading sync from {self.data_file_path}")
        data_sync = read_digital(self.data_file_path).query('chan == 6 and type == 1')
        sync = {
            'time_imec': data_sync['time'].values,
            'frame_imec': data_sync['frame'].values,
            'type_imec': data_sync['type'].values,
        }

        if self.sync is None:
            self.sync = sync
        else:
            self.sync.update(sync)

    def load_nidq(self, path=None):
        nidq_fn = path if path and os.path.isfile(path) else finder(os.path.dirname(os.path.dirname(self.data_file_path)), 'nidq.bin$') or finder(pattern='nidq.bin$')
        
        if not nidq_fn:
            tprint("Could not find a NIDQ file")
            return

        tprint(f"Loading nidq data from {nidq_fn}")
        data_nidq = read_digital(nidq_fn)
        
        df_nidq = data_nidq[data_nidq['chan'] > 0]
        df_sync = data_nidq[(data_nidq['chan'] == 0) & (data_nidq['type'] == 1)]

        self.nidq = {key: df_nidq[key].values for key in ['time', 'frame', 'chan', 'type']}
        data_sync = {f'{key}_nidq': df_sync[key].values for key in ['time', 'frame', 'type']}

        if self.sync is None:
            self.sync = data_sync
        else:
            self.sync.update(data_sync)
        
        self.nidq['time_imec'] = sync(self.sync['time_nidq'], self.sync['time_imec'])(self.nidq['time'])

    def save(self, path=None):
        path = path or self.path

        spike = {
            'time': self.time,
            'frame': self.frame,
            'firing_rate': self.firing_rate,
            'position': self.position,
            'waveform': self.waveform,
            'waveform_idx': self.waveform_idx,
            'waveform_channel': self.waveform_channel,
            'waveform_position': self.waveform_position,
            'Vpp': self.Vpp,
            'n_unit': self.n_unit,
            'channel_map': self.channel_map,
            'channel_position': self.channel_position,
            'cluster_group': self.cluster_group,
            'meta': self.meta,
            'n_channel': self.n_channel,
            'file_create_time': self.file_create_time,
            'data_file_path': self.data_file_path,
            'data_file_path_orig': self.data_file_path_orig,
            'sample_rate': self.sample_rate,
        }

        if self.waveform_raw is not None:
            spike.update({
                'waveform_raw': self.waveform_raw,
                'Vpp_raw': self.Vpp_raw
            })
        
        data = {'spike': spike}
        if self.sync:
            data['sync'] = self.sync
        if self.nidq:
            data['nidq'] = self.nidq

        fn = os.path.join(path, f'{self.session}_data.mat')
        tprint(f"Saving Kilosort data to {fn}")
        savemat_safe(fn, data)

    def plot(self, idx=0, xscale=1, yscale=1):
        """
        Plot the template waveform and raw waveform for a given unit.
        """
        import matplotlib.pyplot as plt

        waveform = self.waveform[idx]
        waveform_raw = self.waveform_raw[idx]
        Vpp = self.Vpp[idx]
        Vpp_raw = self.Vpp_raw[idx]
        position = self.waveform_position[idx]

        x = np.arange(-20.0, 41.0) / 3 * xscale
        y = waveform / Vpp * 40 * yscale
        y_raw = waveform_raw / Vpp_raw * 40 * yscale

        fig, ax = plt.subplots(figsize=(6, 8))
        
        for i in range(y.shape[1]):  # Iterate over the second dimension of y
            xi = x + position[i, 0]
            yi = y[:, i] + position[i, 1]
            yi_raw = y_raw[:, i] + position[i, 1]
            ax.plot(xi, yi, 'k', linewidth=0.5)
            ax.plot(xi, yi_raw, 'r', linewidth=0.5)
            ax.text(xi[0], yi[0], f'{self.waveform_channel[idx, i]}', ha='right', va='center', fontsize=8)

        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_title(f'Unit {idx}')
        ax.legend(['template', 'raw'])
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    fn = finder("C:\\SGL_DATA", "params.py$", folder=True)
    ks = Kilosort(fn)
    ks.save()
    # ks.plot(idx=0)

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(4, 4)
    # axs = axs.flatten()
    # for i in range(spike.n_unit):
    #     axs[i].imshow(spike.waveform_raw[i, :, :].T)

    # run_ks4("C:\\SGL_DATA")
    # fn = finder("C:\\SGL_DATA")
    # meta = read_meta(fn)
    # info = get_probe(meta)