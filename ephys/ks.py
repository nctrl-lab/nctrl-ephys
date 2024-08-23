import os
import numpy as np
import pandas as pd
import scipy.io as sio

from .utils import finder, confirm
from .spikeglx import read_meta, read_analog, get_uV_per_bit


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

        settings['data_dir'] = os.path.dirname(fn)

        meta = read_meta(fn)
        if meta:
            probe = get_probe(meta)
            settings['n_chan_bin'] = meta['nSavedChans']

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


class Spike():
    def __init__(self, path=None):
        if path is None:
            fn = finder(None, 'params.py$')
            path = os.path.dirname(fn)

        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        self.path = path
        self.session = path.split(os.path.sep)[-2]
        self.load_meta()
        self.load_kilosort()
        self.load_waveforms()

    def load_meta(self):
        ops = np.load(os.path.join(self.path, 'ops.npy'), allow_pickle=True).item()
        self.data_file_path_orig = str(ops['data_file_path'])
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

    def load_kilosort(self):
        print(f"ks.Spike.load_kilosort: Loading Kilosort data from {self.path}")

        # Load spike times (in samples)
        frame = np.load(os.path.join(self.path, "spike_times.npy"))
        
        # Load spike-cluster assignments
        cluster = np.load(os.path.join(self.path, 'spike_clusters.npy'))
        
        # Load spike-template assignments
        spike_templates = np.load(os.path.join(self.path, 'spike_templates.npy'))
        
        # Load cluster information
        cluster_info = pd.read_csv(os.path.join(self.path, 'cluster_info.tsv'), sep='\t')
        
        # Get IDs of good clusters
        good_id = cluster_info[cluster_info['group'] == 'good'].cluster_id.values
        
        # Find the main template ID for each good cluster
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
        self.position = np.array([spike_positions[cluster == c].mean(axis=0) 
                                  for c in good_id])
    
        # Load templates and amplitudes
        templates = np.load(os.path.join(self.path, 'templates.npy'))
        amplitudes = np.load(os.path.join(self.path, 'amplitudes.npy'))
        
        # Unwhiten templates
        winv = np.load(os.path.join(self.path, 'whitening_mat_inv.npy'))
        temp_unwhitened = templates @ winv
        
        # Find the channel with maximum amplitude for each template
        max_channel = (temp_unwhitened.max(axis=1) - temp_unwhitened.min(axis=1)).argmax(axis=-1)

        # Load channel map
        self.channel_map = np.load(os.path.join(self.path, 'channel_map.npy'))
        
        # Find the best channel (waveform site) for each good cluster
        self.waveform_site = max_channel[main_template_id]
        
        # Map waveform sites to actual channel numbers
        self.max_channel = self.channel_map[self.waveform_site]
        
        # Calculate mean waveforms for good clusters
        waveform = np.zeros((self.n_unit, temp_unwhitened.shape[1], self.n_channel))
        for i, c in enumerate(good_id):
            spikes = spike_templates[cluster == c]
            amplitudes_c = amplitudes[cluster == c].reshape(-1, 1, 1)
            mean_waveform = (temp_unwhitened[spikes] * amplitudes_c).mean(axis=0)
            waveform[i, :, self.channel_map] = mean_waveform.T
        self.waveform = waveform
    
        self.Vpp = np.ptp(self.waveform, axis=(1, 2))  # peak-to-peak amplitude in uV

    def load_waveforms(self, spk_range=(-20, 41), sample_range=(0, 30000*60)):
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

        spkwav = np.full((n_spk, spk_width, self.n_channel), np.nan)
        batch_indices = np.arange(n_batch)
        batch_starts = batch_indices * n_sample_per_batch + sample_range[0]
        batch_ends = np.minimum((batch_indices + 1) * n_sample_per_batch + sample_range[0], sample_range[1])

        for i_batch, (batch_start, batch_end) in enumerate(zip(batch_starts, batch_ends)):
            print(f"ks.Spike.load_waveforms: Loading waveforms from {self.data_file_path} (batch {i_batch+1}/{n_batch})")
            data = read_analog(self.data_file_path, sample_range=(batch_start, batch_end))

            in_range = (spks >= batch_start + spk_width) & (spks < batch_end - spk_width)
            spk = spks[in_range]
            spk_used[in_range] = True
            spk_no = np.where(in_range)[0]
            for i, i_spk in zip(spk_no, spk):
                spkwav_temp = data[i_spk + spk_range[0] - batch_start:i_spk + spk_range[1] - batch_start, :]
                spkwav[i] = spkwav_temp - spkwav_temp[0, :]
        
        self.waveform_raw = np.array([np.median(spkwav[(idx == i_unit) & spk_used], axis=0) 
                                      for i_unit in range(self.n_unit)])
        self.Vpp_raw = np.ptp(self.waveform_raw, axis=(1, 2))

    def save(self):
        spike = {
            'time': self.time,
            'firing_rate': self.firing_rate,
            'position': self.position,
            'waveform': self.waveform,
            'waveform_site': self.waveform_site,
            'max_channel': self.max_channel,
            'channel_map': self.channel_map,
            'Vpp': self.Vpp,
            'n_unit': self.n_unit,
        }
        if self.waveform_raw is not None:
            spike['waveform_raw'] = self.waveform_raw
            spike['Vpp_raw'] = self.Vpp_raw

        sio.savemat(os.path.join(self.path, f'{self.session}_data.mat'), {'Spike': spike})


if __name__ == "__main__":
    fn = finder("C:\\SGL_DATA", "params.py$")
    fd = os.path.dirname(fn)
    spike = Spike(fd)
    spike.save()

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(4, 4)
    # axs = axs.flatten()
    # for i in range(spike.n_unit):
    #     axs[i].imshow(spike.waveform_raw[i, :, :].T)

    # run_ks4("C:\\SGL_DATA")