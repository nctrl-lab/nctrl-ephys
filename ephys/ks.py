import os
import time
import re
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA

from .spikeglx import read_meta, read_analog, read_digital, get_uV_per_bit, get_channel_idx
from .utils import finder, confirm, savemat_safe, tprint, sync

from .metrics import calculate_metrics, DEFAULT_PARAMS


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

def nearest_channel(channel_position, channel_index=None, count=14):
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
    if channel_index is None:
        return np.argsort(all_distances, axis=1)[:, :count]
    else:
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

    def __repr__(self):
        result = []
        for key, value in self.__dict__.items():
            if key.startswith('__'):
                continue
            if isinstance(value, pd.DataFrame):
                result.append(key)
                for col in value.columns:
                    result.append(f"    {col}: {value[col].shape}")
            elif isinstance(value, dict):
                result.append(key)
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
                result.append(f"{key}: {value}")
            elif isinstance(value, np.ndarray):
                result.append(f"{key}: {value.shape}")
        return "\n".join(result)

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
    
    def _load_kilosort(self):
        tprint(f"Loading Kilosort data from {self.path}")

        self.spike_times = np.load(os.path.join(self.path, "spike_times.npy"))
        self.spike_clusters = np.load(os.path.join(self.path, 'spike_clusters.npy'))
        self.spike_templates = np.load(os.path.join(self.path, 'spike_templates.npy'))
        self.spike_positions = np.load(os.path.join(self.path, 'spike_positions.npy'))
        self.pc_features = np.load(os.path.join(self.path, 'pc_features.npy'))
        self.pc_feature_ind = np.load(os.path.join(self.path, 'pc_feature_ind.npy'))
        self.templates = np.load(os.path.join(self.path, 'templates.npy'))
        self.amplitudes = np.load(os.path.join(self.path, 'amplitudes.npy'))
        self.winv = np.load(os.path.join(self.path, 'whitening_mat_inv.npy'))
        self.channel_map = np.load(os.path.join(self.path, 'channel_map.npy'))
        self.channel_position = np.load(os.path.join(self.path, 'channel_positions.npy'))

        if os.path.exists(os.path.join(self.path, 'energy.npy')):
            self.energy = np.load(os.path.join(self.path, 'energy.npy'))
        if os.path.exists(os.path.join(self.path, 'pc1.npy')):
            self.pc1 = np.load(os.path.join(self.path, 'pc1.npy'))

    def load_kilosort(self, load_all=False):
        """
        Load Kilosort data from the specified path.

        Parameters:
        -----------
        load_all : bool, optional
            If True, load all clusters. If False, load only 'good' clusters. Default is False.

        Attributes:
        -----------
        cluster_id : ndarray
            Array of cluster IDs.
        cluster_group : ndarray
            Group labels for all clusters (good, mua, noise, or nan).
        cluster_good : ndarray
            Array of 'good' cluster IDs.
        cluster_template_id : ndarray
            Array of template IDs for each cluster.
        n_unit : int
            Number of clusters.
        n_unit_good : int
            Number of 'good' clusters.
        time : ndarray of object
            Spike times in seconds for 'good' clusters.
        frame : ndarray of object
            Spike times in samples for 'good' clusters.
        firing_rate : list
            Firing rates for 'good' clusters.
        position : ndarray
            Median spike positions on the probe for 'good' clusters.
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

        channel_map : ndarray
            Mapping of channel indices to physical channels.
        channel_position : ndarray
            Positions of all channels on the probe.
        """
        self._load_kilosort()
        
        tprint("Processing Kilosort data...")
        
        # Manual clustering information
        cluster_fn = os.path.join(self.path, 'cluster_info.tsv')
        if os.path.exists(cluster_fn):
            cluster_info = pd.read_csv(cluster_fn, sep='\t')
            self.cluster_id = cluster_info['cluster_id'].values
            self.cluster_group = cluster_info['group'].values
        else:
            tprint("No cluster_info.tsv found. Loading all clusters.")
            self.cluster_id = np.unique(self.spike_clusters)
            self.cluster_group = np.full_like(self.cluster_id, np.nan)
            load_all = True
        self.cluster_id_inv = {c: i for i, c in enumerate(self.cluster_id)}
        
        tprint("Finished cluster information")
        
        self.cluster_good = self.cluster_id if load_all else self.cluster_id[self.cluster_group == 'good']
        
        self.cluster_template_id = np.array([
            np.bincount(self.spike_templates[self.spike_clusters == c]).argmax()
            for c in self.cluster_good
        ]) # main template id for each cluster

        self.n_unit = len(self.cluster_good)
        
        # Spike times 
        self.time = np.array([self.spike_times[self.spike_clusters == c] / self.sample_rate for c in self.cluster_good], dtype=object)
        self.frame = np.array([self.spike_times[self.spike_clusters == c] for c in self.cluster_good], dtype=object)
        self.firing_rate = [len(i) / (self.spike_times.max() / self.sample_rate) for i in self.time]
        self.position = np.array([np.median(self.spike_positions[self.spike_clusters == c], axis=0) 
                                  for c in self.cluster_good])
        
        tprint("Finished spike times")
    
        # Template waveforms
        temp_unwhitened = self.templates @ self.winv
        template_idx = np.ptp(temp_unwhitened, axis=1).argmax(axis=-1) # main index for each template
        cluster_idx = template_idx[self.cluster_template_id] # main index for each cluster

        self.waveform_idx = nearest_channel(self.channel_position, cluster_idx) # (n_unit, 14)

        waveform = np.zeros((self.n_unit, temp_unwhitened.shape[1], 14))
        for i, c in enumerate(self.cluster_good):
            cluster_mask = self.spike_clusters == c
            spike_templates = self.spike_templates[cluster_mask]
            template_ids, counts = np.unique(spike_templates, return_counts=True)
            
            mean_amplitudes = np.array([self.amplitudes[cluster_mask & (self.spike_templates == tid)].mean() for tid in template_ids])
            weighted_waveforms = temp_unwhitened[template_ids] * mean_amplitudes[:, np.newaxis, np.newaxis] * counts[:, np.newaxis, np.newaxis]
            
            mean_waveform = weighted_waveforms.sum(axis=0) / cluster_mask.sum()
            # We don't know the exact scaling factor that was used by Kilosort, but it was approximately 10.
            waveform[i] = mean_waveform[:, self.waveform_idx[i]] / 10
        self.waveform = waveform
        self.waveform_position = self.channel_position[self.waveform_idx] # channel positions on the probe(n_unit, 14, 2)
        self.waveform_channel = self.channel_map[self.waveform_idx] # actual channel numbers (n_unit, 14)
        self.Vpp = np.ptp(self.waveform, axis=(1, 2))  # peak-to-peak amplitude

        tprint("Finished waveform")


    def load_waveforms(self, spk_range=(-20, 41), sample_range=(0, 30000*300)):
        """
        Load waveforms and related metrics from the raw data file
        
        Parameters
        ----------
        spk_range : tuple, optional
            The range of spike times to load, in samples. Default is (-20, 41).
        sample_range : tuple, optional
            The range of samples to load. Default is (0, 30000*300).
        
        Notes
        -----
        This will calculate the energy and the first PC for each waveform.
        """
        if not os.path.exists(self.data_file_path):
            print(f"Data file {self.data_file_path} does not exist")
            return

        MAX_MEMORY = int(4e9)

        n_sample_file = self.meta['fileSizeBytes'] // (self.meta['nSavedChans'] * np.dtype(np.int16).itemsize)
        sample_range = (max(0, sample_range[0]), min(sample_range[1], n_sample_file))
        
        n_sample = sample_range[1] - sample_range[0]
        n_sample_per_batch = min(int(MAX_MEMORY / self.n_channel / np.dtype(np.int16).itemsize), n_sample)
        n_batch = int(np.ceil(n_sample / n_sample_per_batch))

        spks = np.concatenate(self.frame)
        idx = np.concatenate([i * np.ones_like(f) for i, f in enumerate(self.frame)])

        in_range = (spks >= sample_range[0] - spk_range[0]) & (spks < sample_range[1] - spk_range[1])
        spks, idx = spks[in_range], idx[in_range]

        n_spk = len(spks)
        spk_width = spk_range[1] - spk_range[0]

        spkwav = np.full((n_spk, spk_width, 14), np.nan)
        batch_starts = np.arange(n_batch) * n_sample_per_batch
        batch_ends = np.minimum(batch_starts + n_sample_per_batch, sample_range[1])

        for i_batch, (batch_start, batch_end) in enumerate(zip(batch_starts, batch_ends)):
            tprint(f"Loading waveforms from {self.data_file_path} (batch {i_batch+1}/{n_batch})")
            data = read_analog(self.data_file_path, sample_range=(batch_start+spk_range[0], batch_end+spk_range[1]))

            mask = (spks >= batch_start) & (spks < batch_end)
            spk, spk_idx, spk_no = spks[mask], idx[mask], np.where(mask)[0]
            
            starts = spk + spk_range[0] - batch_start
            starts[i_batch == 0] += spk_range[0]  # Adjust for first batch
            channels = self.waveform_channel[spk_idx]
            time_indices = np.arange(spk_width)
            indices = (starts[:, None] + time_indices[None, :])[:, :, None]
            indices = np.broadcast_to(indices, (len(spk_no), spk_width, 14))
            waveforms = data[indices, channels[:, None]]
            spkwav[spk_no] = waveforms - waveforms[:, 0:1, :]

        self.waveform_raw = np.array([np.nanmedian(spkwav[idx == i_unit], axis=0) 
                                      for i_unit in np.unique(idx)])
        self.Vpp_raw = np.ptp(self.waveform_raw, axis=(1, 2))

    def load_waveforms_full(self, spk_range=(-20, 41), sample_range=(0, 30000*180)):
        """
        Load waveforms and related metrics from the raw data file
        
        Parameters
        ----------
        spk_range : tuple, optional
            The range of spike times to load, in samples. Default is (-20, 41).
        sample_range : tuple, optional
            The range of samples to load. Default is (0, 30000*180).
        
        Notes
        -----
        This will calculate the energy and the first PC for each waveform.
        """
        if not os.path.exists(self.data_file_path):
            print(f"Data file {self.data_file_path} does not exist")
            return

        MAX_MEMORY = int(4e9)

        n_sample_file = self.meta['fileSizeBytes'] // (self.meta['nSavedChans'] * np.dtype(np.int16).itemsize)
        sample_range = (max(0, sample_range[0]), min(sample_range[1], n_sample_file))
        
        n_sample = sample_range[1] - sample_range[0]
        n_sample_per_batch = min(int(MAX_MEMORY / self.n_channel / np.dtype(np.int16).itemsize), n_sample)
        n_batch = int(np.ceil(n_sample / n_sample_per_batch))

        spks = self.spike_times
        idx = np.digitize(self.spike_clusters, self.cluster_id, right=True)

        in_range = (spks >= sample_range[0] - spk_range[0]) & (spks < sample_range[1] - spk_range[1])
        spks, idx = spks[in_range], idx[in_range]

        n_spk = len(spks)
        spk_width = spk_range[1] - spk_range[0]

        spkwav = np.full((n_spk, spk_width, 14), np.nan)
        batch_starts = np.arange(n_batch) * n_sample_per_batch
        batch_ends = np.minimum(batch_starts + n_sample_per_batch, sample_range[1])

        for i_batch, (batch_start, batch_end) in enumerate(zip(batch_starts, batch_ends)):
            tprint(f"Loading waveforms from {self.data_file_path} (batch {i_batch+1}/{n_batch})")
            data = read_analog(self.data_file_path, sample_range=(batch_start+spk_range[0], batch_end+spk_range[1]))

            mask = (spks >= batch_start) & (spks < batch_end)
            spk, spk_idx, spk_no = spks[mask], idx[mask], np.where(mask)[0]
            
            starts = spk + spk_range[0] - batch_start
            starts[i_batch == 0] += spk_range[0]  # Adjust for first batch
            channels = self.waveform_channel[spk_idx]
            time_indices = np.arange(spk_width)
            indices = (starts[:, None] + time_indices[None, :])[:, :, None]
            indices = np.broadcast_to(indices, (len(spk_no), spk_width, 14))
            waveforms = data[indices, channels[:, None]]
            spkwav[spk_no] = waveforms - waveforms[:, 0:1, :]

        self.energy = np.full((self.spike_times.shape[0], spkwav.shape[2]), np.nan)
        self.energy[in_range] = np.linalg.norm(spkwav, axis=1)
        spkwav_norm = spkwav / self.energy[in_range, np.newaxis, :]

        self.pc1 = np.full_like(self.energy, np.nan)
        channel_idx = self.waveform_channel[idx]
        unique_channels = np.unique(channel_idx)
        
        pc1 = np.full((spkwav_norm.shape[0], spkwav_norm.shape[2]), np.nan)
        for channel in unique_channels:
            in_channel = np.where(channel_idx == channel)
            waves = spkwav_norm[in_channel[0], :, in_channel[1]]
            pc1_temp = PCA(n_components=1).fit_transform(waves)
            pc1[in_channel[0], in_channel[1]] = pc1_temp.flatten()
        
        self.pc1[in_range] = pc1

        np.save(os.path.join(self.path, 'energy.npy'), self.energy)
        np.save(os.path.join(self.path, 'pc1.npy'), self.pc1)

        self.waveform_raw = np.array([np.nanmedian(spkwav[idx == i_unit], axis=0) 
                                      for i_unit in np.unique(idx)])
        self.Vpp_raw = np.ptp(self.waveform_raw, axis=(1, 2))

    def load_metrics(self):
        self.metrics = calculate_metrics(
            self.spike_times / self.sample_rate,
            self.spike_clusters,
            self.spike_templates,
            self.amplitudes,
            self.channel_position,
            self.pc_features,
            self.pc_feature_ind,
            self.energy,
            self.pc1,
            self.waveform_idx,
            self.cluster_id_inv,
            DEFAULT_PARAMS
        )

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
            'n_unit_good': self.n_unit_good,
            'cluster_id': self.cluster_id,
            'cluster_good': self.cluster_good,
            'cluster_template_id': self.cluster_template_id,
            'cluster_group': self.cluster_group,
            'channel_map': self.channel_map,
            'channel_position': self.channel_position,
            'meta': self.meta,
            'n_channel': self.n_channel,
            'file_create_time': self.file_create_time,
            'data_file_path': self.data_file_path,
            'data_file_path_orig': self.data_file_path_orig,
            'sample_rate': self.sample_rate,
        }

        if hasattr(self, 'waveform_raw'):
            spike.update({
                'waveform_raw': self.waveform_raw,
                'Vpp_raw': self.Vpp_raw
            })
        
        if hasattr(self, 'metrics'):
            spike.update({
                'metrics': self.metrics
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
    ks = Kilosort("C:\\SGL_DATA\\Y02_20240731_M1_g0\\Y02_20240731_M1_g0_imec0\\kilosort4")
    ks.load_waveforms()
    breakpoint()
    ks.load_metrics()
    # ks.save()
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