import os
import numpy as np
import pandas as pd
import scipy.io as sio

from ephys.util import finder
from ephys.spikeglx import read_meta, read_analog, get_uV_per_bit


def run_ks(path):
    from kilosort import run_kilosort
    from kilosort.parameters import DEFAULT_SETTINGS

    fns = finder(path, 'ap.meta$')
    for fn in fns:
        run_kilosort(DEFAULT_SETTINGS, data_dir=fn)


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

    def load_meta(self):
        ops = np.load(os.path.join(self.path, 'ops.npy'), allow_pickle=True).item()
        self.data_file_path = str(ops['data_file_path'])
        self.meta = read_meta(self.data_file_path)
        self.uV_per_bit = get_uV_per_bit(self.meta)
        self.sample_rate = self.meta.get('imSampRate') or self.meta.get('niSampRate') or 1

    def load_kilosort(self):
        # Load spike times (in samples)
        sample = np.load(os.path.join(self.path, "spike_times.npy"))
        
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
        self.time = np.array([sample[cluster == c] / self.sample_rate for c in good_id], dtype=object)
        self.sample = np.array([sample[cluster == c] for c in good_id], dtype=object)
        
        # Calculate firing rates for good clusters
        self.firing_rate = [len(i) / (sample.max() / self.sample_rate) for i in self.time]

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
        self.waveform = np.stack([(temp_unwhitened[spike_templates[cluster == c]] * 
                                   amplitudes[cluster == c].reshape(-1, 1, 1)).mean(axis=0) 
                                  for i, c in enumerate(good_id)], axis=0)
        
        self.Vpp = np.ptp(self.waveform, axis=(1, 2))  # peak-to-peak amplitude in uV
    
    def load_waveforms(self):
        """
        Load waveforms from the raw data file
        """
        MAX_MEMORY = int(4e9)
        valid_channels = np.unique(self.max_channel)
        n_sample = self.meta['fileSizeBytes'] // (self.meta['nSavedChans'] * 2)
        n_memory = n_sample * len(valid_channels) * 2
        n_batch = int(np.ceil(n_memory / MAX_MEMORY))
        waveform_raw = [np.zeros((len(i), 61)) for i in self.sample]

        spk_idx = []
        for sample in self.sample:
            out = (np.diff(sample) < 61) | (sample < 61) | (sample > n_sample - 61)
            sample = sample[~out]
            spk_idx.append((sample[:, np.newaxis] + np.arange(-20, 41)[np.newaxis, :]).flatten())

        for i_batch in range(n_batch):
            sample_range = (i_batch * MAX_MEMORY, min((i_batch + 1) * MAX_MEMORY, n_sample))
            data = read_analog(self.data_file_path, channel_idx=valid_channels, sample_range=sample_range)
            
            for i_unit, i_spk in enumerate(spk_idx):
                mask = (i_sample >= sample_range[0]) & (i_sample < sample_range[1])
                samples = i_sample[mask] - sample_range[0]
                
                if len(samples) > 0:
                    # Calculate start and end indices for each sample
                    start_indices = np.maximum(0, samples - 19)
                    end_indices = np.minimum(data.shape[0], samples + 42)
                    
                    # Create a list of ranges for each sample
                    waveform_ranges = [np.arange(start, end) for start, end in zip(start_indices, end_indices)]
                    
                    waveforms = data[np.clip(waveform_ranges, 0, data.shape[0] - 1), self.max_channel[i_unit]]
                    
                    mid_point = min(19, waveforms.shape[1] // 2)
                    waveform_raw[i_unit][mask] = waveforms - waveforms[:, mid_point:mid_point+1]

        self.waveform_raw = waveform_raw
        breakpoint()

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
        sio.savemat(os.path.join(self.path, f'{self.session}_data.mat'), {'Spike': spike})


if __name__ == "__main__":
    fn = finder("C:/Users/lapis/Dropbox (HHMI)/data/", "params.py$")
    fd = os.path.dirname(fn)
    spike = Spike(fd)
    spike.load_waveforms()
    # spike.save()
