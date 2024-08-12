import os
import numpy as np
import pandas as pd
import scipy.io as sio

from nctrle.util import finder
from nctrle.spikeglx import read_meta, get_uV_per_bit

class Spike():
    def __init__(self, path=None):
        if path is None:
            fn = finder(os.path.expanduser('~'), 'params.py$')
            path = os.path.dirname(fn)

        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        self.path = path
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
        sio.savemat(os.path.join(self.path, 'data.mat'), spike)


if __name__ == "__main__":
    fn = finder("C:/Users/lapis/Dropbox (HHMI)/data/", "params.py$")
    fd = os.path.dirname(fn)
    spike = Spike(fd)
    spike.save()
