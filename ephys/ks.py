import os
import numpy as np
import pandas as pd
import subprocess
from sklearn.decomposition import PCA
from yaml import load_all

from .spikeglx import read_meta, read_analog, read_digital_chunked, get_uV_per_bit, get_channel_idx
from .utils import finder, confirm, savemat_safe, tprint, sync, get_file

from .metrics import calculate_metrics, DEFAULT_PARAMS, DEFAULT_WAVEFORMS


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
    
    return fns

def run_ks2(path=None, settings=None):
    """
    Run Kilosort2.

    - Kilosort2 path and subfolders should be in the path.
    - kilosort-runner path should be in the path.
    - npy-matlab should be in the path.
    """
    matlab = get_file("matlab", "matlab.exe", "MATLAB")
    cmd = f'"{matlab}" -nodesktop -nosplash -r "runKs2; exit;"'
    subprocess.run(cmd)

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
        Dictionary with probe information for Kilosort4.
    """
    geom_data = meta['snsGeomMap']
    electrodes = pd.DataFrame(geom_data['electrodes'])
    shank_spacing = geom_data['header']['shank_spacing']
    
    x = electrodes['x'].values.astype(np.float32) + electrodes['shank'].values.astype(np.float32) * shank_spacing
    y = electrodes['z'].values.astype(np.float32)
    connected = electrodes['used'].values.astype(np.bool_)
    
    channel_map = np.array([i['channel'] for i in meta['snsChanMap']['channel_map']], dtype=np.int32)
    channel_idx = get_channel_idx(meta)
    
    return {
        'chanMap': channel_map[channel_idx][connected],
        'xc': x[connected],
        'yc': y[connected],
        'kcoords': electrodes['shank'].values.astype(np.int32)[connected],
        'n_chan': np.int32(meta['nSavedChans'] - 1),
    }

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
    """
    A class for loading and processing Kilosort output data.

    This class provides methods to load Kilosort output files, process spike data,
    calculate various metrics, and save the processed data.

    Attributes:
        path (str): Path to the Kilosort output directory.
        session (str): Name of the recording session.
        sync (dict): Synchronization data.
        nidq (dict): NIDQ data.
        meta (dict): Metadata from the recording.
        n_channel (int): Number of channels.
        uV_per_bit (float): Microvolts per bit conversion factor.
        sample_rate (float): Sampling rate of the recording.
        file_create_time (str): Time when the file was created.
        data_file_path (str): Path to the raw data file.
        spike_times (ndarray): Array of spike times.
        spike_clusters (ndarray): Array of cluster IDs for each spike.
        spike_templates (ndarray): Array of template IDs for each spike.
        spike_positions (ndarray): Array of spike positions.
        pc_features (ndarray): Array of PC features for each spike.
        pc_feature_ind (ndarray): Array of PC feature indices.
        templates (ndarray): Array of spike templates.
        amplitudes (ndarray): Array of spike amplitudes.
        winv (ndarray): Inverse of the whitening matrix.
        channel_map (ndarray): Map of channel indices to physical channels.
        channel_position (ndarray): Positions of channels on the probe.
        cluster_id (ndarray): Array of cluster IDs.
        cluster_group (ndarray): Array of cluster group labels.
        cluster_good (ndarray): Array of 'good' cluster IDs.
        cluster_template_id (ndarray): Array of template IDs for each cluster.
        n_unit (int): Number of units (clusters).
        time (ndarray): Array of spike times in seconds.
        frame (ndarray): Array of spike times in samples.
        firing_rate (list): List of firing rates for each unit.
        position (ndarray): Array of median spike positions for each unit.
        waveform (ndarray): Array of mean template waveforms for each unit.
        waveform_idx (ndarray): Array of channel indices for waveforms.
        waveform_channel (ndarray): Array of channel numbers for waveforms.
        waveform_position (ndarray): Array of channel positions for waveforms.
        Vpp (ndarray): Array of peak-to-peak amplitudes for each unit.
        metrics (dict): Dictionary of calculated metrics for each unit.

    Methods:
        __init__(self, path=None): Initialize the Kilosort object.
        __repr__(self): Return a string representation of the object.
        load_meta(self): Load metadata from the recording.
        _load_kilosort(self): Load Kilosort output files.
        load_kilosort(self, load_all=False): Process Kilosort data.
        load_waveforms(self, spk_range=(-20, 41), sample_range=(0, 30000*300)): Load waveforms from raw data.
        load_energy_pc1(self, spk_range=(-20, 41), sample_range=(0, 30000*300), max_spike=1e6): Calculate energy and PC1 for spikes.
        load_metrics(self): Calculate various metrics for each unit.
        save_metrics(self): Save calculated metrics to a file.
        load_sync(self): Load synchronization data.
        load_nidq(self, path=None): Load NIDQ data.
        save(self, path=None): Save processed data to a file.
        plot(self, idx=0, xscale=1, yscale=1): Plot waveforms for a given unit.
    """

    def __init__(self, path=None, group='good'):
        """
        Initialize the Kilosort object.

        Args:
            path (str, optional): Path to the Kilosort output directory. If None, attempts to find it automatically.

        Raises:
            ValueError: If the specified path does not exist or is not a valid directory.
        """
        # find params.py if no path provided
        if path is None:
            path = finder(None, 'params.py$')

        if os.path.isfile(path):
            path = os.path.dirname(path)

        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        self.path = path
        
        # look for ap.bin in the current directory or its parent directory
        self.data_file_path = None
        for _ in range(2):  # Check current and parent directory
            if any('ap.bin' in f for f in os.listdir(path)):
                self.data_file_path = finder(path, 'ap.bin$', ask=False)[0]
                break
            path = os.path.dirname(path)
        else:
            raise ValueError(f"No ap.bin file found in {path} or its parent directory")

        self.session = path.split(os.path.sep)[-2]
        self.sync = None
        self.nidq = None

        self.load_meta()
        self.load_kilosort(group=group)

    def __repr__(self):
        """
        Return a string representation of the Kilosort object.

        Returns:
            str: A string containing information about the object's attributes.
        """
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
        """
        Load metadata from the recording.

        This method reads the metadata from the ops.npy file and the associated binary data file.
        It sets various attributes of the object based on the metadata.

        Raises:
            ValueError: If the data file name does not match the original name and the user chooses not to continue.
        """
        self.meta = read_meta(self.data_file_path)
        self.n_channel = self.meta['snsApLfSy']['AP']
        self.uV_per_bit = get_uV_per_bit(self.meta)
        self.sample_rate = self.meta.get('imSampRate') or self.meta.get('niSampRate') or 1
        self.file_create_time = self.meta.get('fileCreateTime')
    
    def _load_kilosort(self):
        """
        Load Kilosort output files.

        This method loads various Kilosort output files, including spike times, clusters, templates, and more.
        It sets several attributes of the object based on these files.
        """
        tprint(f"Loading Kilosort data from {self.path}")

        # Kilosort2 saves unsqueezed arrays. 1D arrays are saved as 2D with shape (n_spike, 1)
        self.spike_times = np.load(os.path.join(self.path, "spike_times.npy")).squeeze().astype(np.int64)
        self.spike_clusters = np.load(os.path.join(self.path, 'spike_clusters.npy')).squeeze().astype(np.int64)
        self.spike_templates = np.load(os.path.join(self.path, 'spike_templates.npy')).squeeze().astype(np.int64)
        if os.path.exists(os.path.join(self.path, 'spike_positions.npy')):
            self.spike_positions = np.load(os.path.join(self.path, 'spike_positions.npy'))
        self.pc_features = np.load(os.path.join(self.path, 'pc_features.npy'))
        self.pc_feature_ind = np.load(os.path.join(self.path, 'pc_feature_ind.npy'))
        self.templates = np.load(os.path.join(self.path, 'templates.npy'))
        self.amplitudes = np.load(os.path.join(self.path, 'amplitudes.npy')).squeeze()
        self.winv = np.load(os.path.join(self.path, 'whitening_mat_inv.npy'))
        self.channel_map = np.load(os.path.join(self.path, 'channel_map.npy')).squeeze().astype(np.int64)
        self.channel_position = np.load(os.path.join(self.path, 'channel_positions.npy'))

        if os.path.exists(os.path.join(self.path, 'energy.npy')):
            self.energy = np.load(os.path.join(self.path, 'energy.npy'))
            tprint(f"Loaded energy: it has {np.isnan(self.energy[:, 0]).sum()}/{self.energy.shape[0]} NaN values")

        if os.path.exists(os.path.join(self.path, 'pc1.npy')):
            self.pc1 = np.load(os.path.join(self.path, 'pc1.npy'))
            tprint(f"Loaded PC1: it has {np.isnan(self.pc1[:, 0]).sum()}/{self.pc1.shape[0]} NaN values")
        
        if os.path.exists(os.path.join(self.path, 'waveform_raw.npy')):
            self.waveform_raw_all = np.load(os.path.join(self.path, 'waveform_raw.npy'))
            tprint(f"Loaded waveform_raw")

        if os.path.exists(os.path.join(self.path, 'peak_raw.npy')):
            self.peak_raw_all = np.load(os.path.join(self.path, 'peak_raw.npy'))
            tprint(f"Loaded peak_raw")

    def load_kilosort(self, group='good'):
        """
        Process Kilosort data.

        This method processes the loaded Kilosort data, calculating various attributes such as
        cluster information, spike times, waveforms, and more.

        Args:
            group (str, optional): The group of clusters to load. Default is 'good'.
            can be 'good', 'mua', or 'all'

        Attributes modified:
            cluster_id, cluster_group, cluster_good, cluster_template_id, n_unit, time, frame, firing_rate,
            position, waveform, waveform_idx, waveform_position, waveform_channel, Vpp, peak
        """
        self._load_kilosort()
        
        tprint("Processing Kilosort data...")
        
        # Manual clustering information
        cluster_fn = os.path.join(self.path, 'cluster_info.tsv')
        if os.path.exists(cluster_fn):
            cluster_info = pd.read_csv(cluster_fn, sep='\t')
            self.cluster_id = cluster_info['cluster_id'].values
            self.cluster_group = cluster_info['group'].values

            # When cluster_info.tsv is generated by Phy, but didn't sort the clusters
            if not (self.cluster_group.astype(object) == 'good').any():
                tprint("cluster_info.tsv was found, but couldn't find 'good' clusters. Loading all clusters.")
                group = 'all'

        else:
            tprint("No cluster_info.tsv found. Loading all clusters.")
            self.cluster_id = np.unique(self.spike_clusters)
            self.cluster_group = np.full_like(self.cluster_id, np.nan, dtype=object)
            group = 'all'
        self.cluster_id_inv = {c: i for i, c in enumerate(self.cluster_id)}
        
        tprint("Finished cluster information")
        
        if group == 'all':
            self.cluster_good = self.cluster_id
        elif group == 'good':
            self.cluster_good = self.cluster_id[self.cluster_group == 'good']
        elif group == 'mua':
            self.cluster_good = self.cluster_id[(self.cluster_group == 'good') | (self.cluster_group == 'mua') ]
        else:
            raise ValueError(f"Invalid group: {group}")

        self.cluster_template_id = np.array([
            np.bincount(self.spike_templates[self.spike_clusters == c]).argmax()
            for c in self.cluster_good
        ]) # main template id for good clusters
        cluster_template_id_all = np.array([
            np.bincount(self.spike_templates[self.spike_clusters == c]).argmax()
            for c in self.cluster_id
        ]) # main template id for all clusters

        self.n_unit = len(self.cluster_good)
        self.n_unit_all = len(self.cluster_id)
        
        # Spike times 
        self.time = np.array([self.spike_times[self.spike_clusters == c] / self.sample_rate for c in self.cluster_good], dtype=object)
        self.frame = np.array([self.spike_times[self.spike_clusters == c] for c in self.cluster_good], dtype=object)
        self.firing_rate = [len(i) / (self.spike_times.max() / self.sample_rate) for i in self.time]
        if hasattr(self, 'spike_positions'):
            self.position = np.array([np.median(self.spike_positions[self.spike_clusters == c], axis=0) 
                                      for c in self.cluster_good])
        
        tprint("Finished spike times")
    
        # Template waveforms
        temp_unwhitened = self.templates @ self.winv

        if temp_unwhitened.shape[1] > 61: # Kilosort2 saves 82 samples instead of 61 (the first 21 points are zeros)
            temp_unwhitened = temp_unwhitened[:, -61:, :]
        
        template_idx = np.ptp(temp_unwhitened, axis=1).argmax(axis=-1) # main index for each template
        cluster_idx = template_idx[self.cluster_template_id] # main index for each cluster
        cluster_idx_all = template_idx[cluster_template_id_all] # main index for all clusters
        self.cluster_ind = nearest_channel(self.channel_position, cluster_idx_all)

        waveform = np.zeros((self.n_unit_all, temp_unwhitened.shape[1], 14))
        for i, c in enumerate(self.cluster_id):
            cluster_mask = self.spike_clusters == c
            spike_templates = self.spike_templates[cluster_mask]
            template_ids, counts = np.unique(spike_templates, return_counts=True)
            
            mean_amplitudes = np.array([self.amplitudes[cluster_mask & (self.spike_templates == tid)].mean() for tid in template_ids])
            weighted_waveforms = temp_unwhitened[template_ids] * mean_amplitudes[:, np.newaxis, np.newaxis] * counts[:, np.newaxis, np.newaxis]
            
            mean_waveform = weighted_waveforms.sum(axis=0) / cluster_mask.sum()
            # We don't know the exact scaling factor that was used by Kilosort, but it was approximately 10.
            waveform[i] = mean_waveform[:, self.cluster_ind[i]] / 10

        self.waveform_all = waveform[:, :, 0] # just use the main channel
        self.peak_all = -self.waveform_all.min(axis=1)

        self.waveform = waveform[np.isin(self.cluster_id, self.cluster_good)]
        self.waveform_idx = nearest_channel(self.channel_position, cluster_idx) # (n_unit, 14)
        self.waveform_position = self.channel_position[self.waveform_idx] # channel positions on the probe(n_unit, 14, 2)
        self.waveform_channel = self.channel_map[self.waveform_idx] # actual channel numbers (n_unit, 14)
        with np.errstate(invalid='ignore'):
            self.Vpp = np.ptp(self.waveform, axis=(1, 2))  # peak-to-peak amplitude
            self.peak = -self.waveform.min(axis=(1, 2))

        tprint("Finished waveform (template)")

    def load_waveforms(self, spk_range=(-20, 41), max_spike=1000000):
        """
        Load waveforms from the raw data file.

        This method extracts waveforms for each spike from the raw data file and calculates
        various waveform-related metrics.

        Args:
            spk_range (tuple, optional): The range of spike times to load, in samples. Default is (-20, 41).
            max_spike (int, optional): The maximum number of spikes to process. If the actual number of spikes exceeds
                this value, the sample range will be adjusted to include only the first max_spike spikes.
                Default is 1000000.

        Attributes modified:
            waveform_raw, Vpp_raw, peak_raw
        """
        if not os.path.exists(self.data_file_path):
            print(f"Data file {self.data_file_path} does not exist")
            return

        MAX_MEMORY = int(4e9)

        if len(self.spike_times) > max_spike:
            sample_range = (0, self.spike_times[max_spike])
        else:
            sample_range = (0, self.spike_times[-1])

        # n_sample_file = self.meta['fileSizeBytes'] // (self.meta['nSavedChans'] * np.dtype(np.int16).itemsize)
        # sample_range = (max(0, sample_range[0]), min(sample_range[1], n_sample_file))
        
        n_sample = sample_range[1] - sample_range[0]
        n_sample_per_batch = min(int(MAX_MEMORY / self.n_channel / np.dtype(np.int16).itemsize), n_sample)
        n_batch = int(np.ceil(n_sample / n_sample_per_batch))

        spks = np.concatenate(self.frame)
        idx = np.concatenate([i * np.ones_like(f) for i, f in enumerate(self.frame)])

        in_range = (spks >= sample_range[0] - spk_range[0]) & (spks < sample_range[1] - spk_range[1])
        spks, idx = spks[in_range], idx[in_range]

        n_spk = len(spks)
        spk_width = spk_range[1] - spk_range[0]
        time_indices = np.arange(spk_width)

        spkwav = np.full((n_spk, spk_width, 14), np.nan)
        batch_starts = np.arange(n_batch) * n_sample_per_batch
        batch_ends = np.minimum(batch_starts + n_sample_per_batch, sample_range[1])

        for i_batch, (batch_start, batch_end) in enumerate(zip(batch_starts, batch_ends)):
            tprint(f"Loading waveforms from {self.data_file_path} (batch {i_batch+1}/{n_batch})")
            data = read_analog(self.data_file_path, sample_range=(batch_start+spk_range[0], batch_end+spk_range[1]))

            mask = (spks >= batch_start) & (spks < batch_end)
            spk, spk_idx, spk_no = spks[mask], idx[mask], np.where(mask)[0]
            
            starts = spk - batch_start
            starts[i_batch == 0] += spk_range[0]  # Adjust for first batch
            channels = self.waveform_channel[spk_idx]
            indices = (starts[:, None] + time_indices[None, :])[:, :, None]
            indices = np.broadcast_to(indices, (len(spk_no), spk_width, 14))
            waveforms = data[indices, channels[:, None]]
            spkwav[spk_no] = waveforms - waveforms[:, 0:1, :]
            del data, waveforms

        self.waveform_raw = np.array([np.nanmedian(spkwav[idx == i_unit], axis=0) 
                                      for i_unit in np.unique(idx)])
        self.Vpp_raw = np.ptp(self.waveform_raw, axis=(1, 2))
        self.peak_raw = -self.waveform_raw.min(axis=(1, 2))
        
        tprint("Finished waveform (raw)")

    def load_energy_pc1(self, spk_range=(-20, 41), max_spike=1000000):
        """
        Calculate energy and first principal component (PC1) for each spike.

        This method processes the raw data file in batches to manage memory usage efficiently.
        It calculates the energy of each spike waveform and performs PCA to extract the first
        principal component, which are useful features for spike sorting and cluster quality assessment.

        Args:
            spk_range (tuple of int, optional): The range of samples around each spike to extract, relative to the spike time.
                Default is (-20, 41), which extracts 61 samples centered on each spike.
            max_spike (int, optional): The maximum number of spikes to process. If the actual number of spikes exceeds
                this value, the sample range will be adjusted to include only the first max_spike spikes.
                Default is 1e6 (1 million spikes).

        Attributes modified:
            energy, pc1, waveform_raw_all, peak_raw_all

        Notes:
            - The method saves the calculated energy and PC1 values as .npy files in the same directory as the raw data.
            - The energy is calculated as the L2 norm of each waveform divided by the square root of the number of samples.
            - The PC1 is calculated using PCA on the waveforms normalized by the L2 norm.
        """
        if not os.path.exists(self.data_file_path):
            print(f"Data file {self.data_file_path} does not exist")
            return

        MAX_MEMORY = int(4e9)

        if len(self.spike_times) > max_spike:
            sample_range = (0, self.spike_times[max_spike])
        else:
            sample_range = (0, self.spike_times[-1])

        # n_sample_file = self.meta['fileSizeBytes'] // (self.meta['nSavedChans'] * np.dtype(np.int16).itemsize)
        # sample_range = (max(0, sample_range[0]), min(sample_range[1], n_sample_file))
        
        n_sample = sample_range[1] - sample_range[0]
        n_sample_per_batch = min(int(MAX_MEMORY / self.n_channel / np.dtype(np.int16).itemsize), n_sample)
        n_batch = int(np.ceil(n_sample / n_sample_per_batch))

        spks = self.spike_times
        idx = np.array([self.cluster_id_inv[int(cluster)] for cluster in self.spike_clusters])

        in_range = (spks >= sample_range[0] - spk_range[0]) & (spks < sample_range[1] - spk_range[1])
        spks, idx = spks[in_range], idx[in_range]

        n_spk = len(spks)
        spk_width = spk_range[1] - spk_range[0]
        time_indices = np.arange(spk_width)

        # get the main 14 channels for each cluster
        cluster_channels = self.channel_map[self.cluster_ind]

        spkwav = np.zeros((n_spk, spk_width, 14))
        batch_starts = np.arange(n_batch) * n_sample_per_batch
        batch_ends = np.minimum(batch_starts + n_sample_per_batch, sample_range[1])

        for i_batch, (batch_start, batch_end) in enumerate(zip(batch_starts, batch_ends)):
            tprint(f"Loading waveforms from {self.data_file_path} (batch {i_batch+1}/{n_batch})")
            data = read_analog(self.data_file_path, sample_range=(batch_start+spk_range[0], batch_end+spk_range[1]))

            mask = (spks >= batch_start) & (spks < batch_end)
            spk, spk_idx, spk_no = spks[mask], idx[mask], np.where(mask)[0]
            
            starts = spk - batch_start
            starts[i_batch == 0] += spk_range[0]  # Adjust for first batch
            channels = cluster_channels[spk_idx]
            indices = (starts[:, None] + time_indices[None, :])[:, :, None]
            indices = np.broadcast_to(indices, (len(spk_no), spk_width, 14))
            waveforms = data[indices, channels[:, None]]
            spkwav[spk_no] = waveforms - waveforms[:, :5, :].mean(axis=1, keepdims=True)
            del data, waveforms

        spkwav[np.isinf(spkwav) | np.isnan(spkwav)] = 0

        # Waveform features
        self.waveform_raw_all = np.full((len(self.cluster_id), spk_width), np.nan)
        for i in np.unique(idx):
            self.waveform_raw_all[i] = np.nanmedian(spkwav[idx == i, :, 0], axis=0)
        self.peak_raw_all = -np.nanmin(self.waveform_raw_all, axis=1)

        self.energy = np.full((self.spike_times.shape[0], 14), np.nan)
        self.energy[in_range] = np.linalg.norm(spkwav, axis=1) / np.sqrt(spk_width)
        spkwav /= np.linalg.norm(spkwav, axis=1, keepdims=True)
        spkwav[np.isinf(spkwav) | np.isnan(spkwav)] = 0

        channel_ind_idx = cluster_channels[idx]
        pc1 = np.full((spkwav.shape[0], spkwav.shape[2]), np.nan)
        for channel in np.unique(cluster_channels):
            in_channel = np.where(channel_ind_idx == channel)
            waves = spkwav[in_channel[0], 5:, in_channel[1]]
            if waves.shape[0] > 0:
                wave_mean = np.nanmean(waves, axis=0, keepdims=True)
                wave_std = np.nanstd(waves, axis=0, keepdims=True)
                wave_std[wave_std == 0] = 1
                waves_z = (waves - wave_mean) / wave_std
                np.nan_to_num(waves_z, copy=False, nan=0, posinf=0, neginf=0)
                pca = PCA(n_components=1)
                pc1[in_channel[0], in_channel[1]] = pca.fit_transform(waves_z).ravel()

        self.pc1 = np.full((self.spike_times.shape[0], 14), np.nan)
        self.pc1[in_range] = pc1


        tprint("Saving energy")
        np.save(os.path.join(self.path, 'energy.npy'), self.energy)
        tprint("Saving PC1")
        np.save(os.path.join(self.path, 'pc1.npy'), self.pc1)
        tprint("Saving waveform_raw")
        np.save(os.path.join(self.path, 'waveform_raw.npy'), self.waveform_raw_all)
        tprint("Saving peak_raw")
        np.save(os.path.join(self.path, 'peak_raw.npy'), self.peak_raw_all)

    def load_metrics(self):
        """
        Calculate various metrics for each unit.

        This method calculates metrics such as L-ratio, isolation distance, and presence ratio
        for each unit. It uses the spike times, clusters, templates, amplitudes, and other features
        to compute these metrics.

        Attributes modified:
            metrics (dict): A dictionary containing calculated metrics for each unit.
        """
        if not hasattr(self, 'energy') or not hasattr(self, 'pc1'):
            self.load_energy_pc1()
        tprint("Calculating metrics")
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
            self.cluster_ind,
            self.cluster_id_inv,
            DEFAULT_PARAMS
        )
    
    def save_metrics(self):
        """
        Save cluster metrics to a TSV file.

        This method calculates and saves various metrics for each cluster, including:
        - A quality score based on multiple criteria
        - L-ratio
        - Isolation distance
        - Waveform correlations with default pyramidal and interneuron waveforms

        The metrics are saved to a file named 'cluster_metrics.tsv' in the Kilosort output directory.

        Note: The scoring system used to calculate the quality score can be easily modified
        by adjusting the criteria and thresholds in the score calculation section of this method.
        """
        if not hasattr(self, 'metrics'):
            self.load_metrics()

        # Extract relevant metrics
        ids = self.metrics['cluster_id']
        fr = self.metrics['firing_rate']
        presence_ratio = self.metrics['presence_ratio']
        l_ratio = self.metrics['l_ratio']
        isolation_distance = self.metrics['isolation_distance']
        isi_viol_corrected = self.metrics['isi_viol_corrected']
        peak = self.peak_raw_all

        # Calculate waveform correlations
        waveform_reshaped = self.waveform_all.reshape(self.waveform_all.shape[0], -1)
        default_waveforms_reshaped = DEFAULT_WAVEFORMS.reshape(DEFAULT_WAVEFORMS.shape[0], -1)
        corr_coeffs = np.corrcoef(waveform_reshaped, default_waveforms_reshaped)
        waveform_corr = corr_coeffs[:self.waveform_all.shape[0], self.waveform_all.shape[0]:]

        # Calculate quality score based on multiple criteria
        # Note: This scoring system can be easily modified by adjusting the criteria and thresholds below
        score = np.zeros_like(l_ratio, dtype=int)
        score += (l_ratio < 1.0) + (l_ratio < 0.1) + (l_ratio < 0.05)
        score += (isolation_distance > 10)
        score += (fr > 0.5)
        score += (presence_ratio > 0.5)
        score += (isi_viol_corrected < 0.5)
        score += (waveform_corr[:, 0] > 0.9) | (waveform_corr[:, 1] > 0.9) | (waveform_corr[:, 2] > 0.9)
        score += (peak > 40)

        # Create DataFrame with metrics
        df = pd.DataFrame({
            'cluster_id': ids,
            'score': score,
            'l_ratio': l_ratio,
            'iso_dist': isolation_distance,
            'wav_pyr': waveform_corr[:, 0],
            'wav_int': waveform_corr[:, 2],
            'peak_uV': peak
        })

        # Save metrics to TSV file
        tprint("Saving metrics")
        cluster_metrics_fn = os.path.join(self.path, 'cluster_metrics.tsv')
        df.to_csv(cluster_metrics_fn, sep='\t', index=False)

    def load_sync(self):
        """
        Load synchronization data from the IMEC data file.

        This method reads digital data from the IMEC file, specifically looking for
        synchronization pulses on channel 6. It extracts time, frame, and type information
        for these pulses and stores them in the 'sync' attribute.

        If the data file doesn't exist or is not of the IMEC type, appropriate error
        messages are printed and the method returns without loading any data.

        Returns:
            None
        """
        if not os.path.exists(self.data_file_path):
            print(f"Data file {self.data_file_path} does not exist")
            return

        if self.meta.get('typeThis') != 'imec':
            print(f"Unsupported data type: {self.meta.get('typeThis')}")
            return

        tprint(f"Loading sync from {self.data_file_path}")
        data_sync = read_digital_chunked(self.data_file_path).query('chan == 6 and type == 1')
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
        """
        Load National Instruments Data Acquisition (NIDQ) data.

        This method reads digital data from the NIDQ file, processes it, and stores
        the results in the 'nidq' and 'sync' attributes.

        Args:
            path (str, optional): Path to the NIDQ file. If not provided, the method
                                  attempts to find the file automatically.

        Returns:
            None

        Side effects:
            - Updates self.nidq with NIDQ data
            - Updates self.sync with synchronization data from NIDQ
            - Calculates and stores IMEC time for NIDQ data
        """
        nidq_fn = path if path and os.path.isfile(path) else finder(os.path.dirname(os.path.dirname(self.data_file_path)), 'nidq.bin$') or finder(pattern='nidq.bin$')
        
        if not nidq_fn:
            tprint("Could not find a NIDQ file")
            return

        tprint(f"Loading nidq data from {nidq_fn}")
        data_nidq = read_digital_chunked(nidq_fn)
        
        df_nidq = data_nidq[data_nidq['chan'] > 0]
        df_sync = data_nidq[(data_nidq['chan'] == 0) & (data_nidq['type'] == 1)]

        self.nidq = {key: df_nidq[key].values for key in ['time', 'frame', 'chan', 'type']}
        data_sync = {f'{key}_nidq': df_sync[key].values for key in ['time', 'frame', 'type']}

        if self.sync is None:
            self.sync = data_sync
        else:
            self.sync.update(data_sync)
        
        self.nidq['time_imec'] = sync(self.sync['time_nidq'], self.sync['time_imec'])(self.nidq['time'])

    def load_obx(self, path=None, use_digital=True, analog_threshold=2.8):
        """
        Load OneBox (OBX) data.

        This method reads digital data from the OBX file, processes it, and stores
        the results in the 'obx' and 'sync' attributes.

        Args:
            path (str, optional): Path to the OBX file. If not provided, the method
                                  attempts to find the file automatically.
            use_digital (bool, optional): Whether to use digital data. If False, the method
                                          will use analog data.
            analog_threshold (float, optional): Threshold for analog data.

        Returns:
            None
        """
        obx_fn = path if path and os.path.isfile(path) else finder(os.path.dirname(os.path.dirname(self.data_file_path)), 'obx.bin$') or finder(pattern='obx.bin$')
        if not obx_fn:
            tprint("Could not find a OBX file")
            return

        tprint(f"Loading obx data from {obx_fn}")

        # load digital data
        digital_obx = read_digital_chunked(obx_fn)
        meta_obx = read_meta(obx_fn)
        sync_channel = 6 if meta_obx['acqXaDwSy']['DW'] == 0 else 22
        df_sync = digital_obx[(digital_obx['chan'] == sync_channel) & (digital_obx['type'] == 1)]
        data_sync = {f'{key}_obx': df_sync[key].values for key in ['time', 'frame', 'type']}
        if self.sync is None:
            self.sync = data_sync
        else:
            self.sync.update(data_sync)

        # load analog data
        if use_digital:
            self.obx = digital_obx
        else:
            data_obx = read_analog(obx_fn) > analog_threshold * 1e6
            changes = np.where(np.diff(data_obx, axis=0) != 0)
            timestamps = changes[0] + 1
            event_id = changes[1]
            event_type = data_obx[changes[0] + 1, changes[1]]

            self.obx = pd.DataFrame({
                'time': timestamps / meta_obx['obSampRate'],
                'frame': timestamps,
                'chan': event_id,
                'type': event_type
            })

        # calculate IMEC time
        self.obx['time_imec'] = sync(self.sync['time_obx'], self.sync['time_imec'])(self.obx['time'])

    def save(self, path=None):
        """
        Save Kilosort data to a MATLAB file.

        Args:
            path (str, optional): Directory to save the file. Uses default if None.

        Saves spike data, sync data, and NIDQ data (if available) to '{self.session}_data.mat'.
        Includes all unit data, raw waveforms, and metrics if present.
        """
        path = path or self.path

        spike = {
            'time': self.time,
            'frame': self.frame,
            'firing_rate': self.firing_rate,
            'waveform': self.waveform,
            'waveform_idx': self.waveform_idx,
            'waveform_channel': self.waveform_channel,
            'waveform_position': self.waveform_position,
            'Vpp': self.Vpp,
            'peak': self.peak,
            'n_unit': self.n_unit,
            'n_unit_all': self.n_unit_all,
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
            'sample_rate': self.sample_rate,
        }

        if hasattr(self, 'position'):
            spike.update({
                'position': self.position,
            })

        if hasattr(self, 'waveform_raw'):
            spike.update({
                'waveform_raw': self.waveform_raw,
                'Vpp_raw': self.Vpp_raw,
                'peak_raw': self.peak_raw,
            })
        
        if hasattr(self, 'metrics'):
            spike.update({
                'metrics': self.metrics,
                'peak_raw_all': self.peak_raw_all,
            })
        
        data = {'spike': spike}
        if self.sync:
            data['sync'] = self.sync
        if self.nidq:
            data['nidq'] = self.nidq
        if hasattr(self, 'obx'):
            data['obx'] = self.obx

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
    """
    This section is for testing the Kilosort class.

    To run this code block, you can use the following command:
    > python -m ephys.ks
    """
    # run_ks2()
    ks = Kilosort()
    # ks.load_waveforms()
    # ks.load_energy_pc1()
    # ks.load_metrics()
    ks.save_metrics()
    # breakpoint()
    # breakpoint()
    # ks.load_metrics()
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
