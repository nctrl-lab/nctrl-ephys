"""
Original code from:
https://github.com/AllenInstitute/ecephys_spike_sorting/tree/main/ecephys_spike_sorting/modules/quality_metrics
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Tuple

import warnings

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import cdist
from scipy.stats import chi2
from scipy.ndimage.filters import gaussian_filter1d


DEFAULT_PARAMS = {
    'isi_threshold': 0.0015,
    'min_isi': 0.0,
    'num_channels_to_compare': 4,
    'max_spikes_for_unit': 1e9,
    'max_spikes_for_nn': 10000,
    'n_neighbors': 4,
    'n_silhouette': 10000,
    'drift_metrics_interval_s': 10,
    'drift_metrics_min_spikes_per_interval': 100.0,
    'do_parallel': True
}


def calculate_metrics(spike_times,
                      spike_clusters,
                      spike_templates,
                      amplitudes,
                      channel_pos,
                      pc_features,
                      pc_feature_ind,
                      energy,
                      pc1,
                      waveform_idx,
                      cluster_id_inv,
                      params):
    """
    Calculate quality metrics for all units on one probe.

    Parameters:
    -----------
    spike_times : array-like
        Timestamps of all spikes.
    spike_clusters : array-like
        Cluster IDs for all spikes.
    spike_templates : array-like
        Template IDs for all spikes.
    amplitudes : array-like
        Amplitudes of all spikes.
    channel_pos : array-like
        Positions of channels.
    pc_features : array-like or None
        PC features for all spikes.
    pc_feature_ind : array-like or None
        Indices of PC features.
    energy : array-like
        Energy of all spikes.
    pc1 : array-like
        PC1 of all spikes.
    waveform_idx : array-like
        Channel indices of all templates.
    cluster_id_inv : array-like
        Inverse mapping of cluster IDs.
    params : dict
        Dictionary of parameters for metric calculations.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing calculated metrics for all units.

    Notes:
    ------
    This function calculates various quality metrics for spike sorting results, including:
    - ISI violations
    - Presence ratio
    - Firing rate
    - Amplitude cutoff
    If PC features are provided, it also calculates:
    - Isolation distance
    - L-ratio
    - d-prime
    - Nearest-neighbors hit rate and miss rate
    - Silhouette score
    """

    cluster_ids = np.unique(spike_clusters)
    total_units = len(cluster_ids)
    metrics = OrderedDict()

    np.random.seed(9999)

    print("Calculating isi violations")
    isi_viol, num_viol = calculate_isi_violations(spike_times, spike_clusters, total_units, params['isi_threshold'], params['min_isi'])

    print("Calculating corrected isi violations")
    isi_viol_corrected = calculate_isi_violations_corrected(spike_times, spike_clusters, total_units, params['isi_threshold'], params['min_isi'])

    print("Calculating presence ratio")
    presence_ratio = calculate_presence_ratio(spike_times, spike_clusters, total_units)

    print("Calculating firing rate")
    firing_rate = calculate_firing_rate(spike_times, spike_clusters, total_units)

    print("Calculating amplitude cutoff")
    amplitude_cutoff = calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units)

    metrics.update({
        'cluster_id': cluster_ids,
        'firing_rate': firing_rate,
        'presence_ratio': presence_ratio,
        'isi_viol': isi_viol,
        'isi_viol_corrected': isi_viol_corrected,
        'num_viol': num_viol,
        'amplitude_cutoff': amplitude_cutoff
    })

    if pc_features is not None:
        print("Calculating PC-based metrics")
        isolation_distance, l_ratio, d_prime, nn_hit_rate, nn_miss_rate = calculate_pc_metrics(
            spike_clusters, spike_templates, pc_features, pc_feature_ind, 
            energy, pc1, waveform_idx, cluster_id_inv,
            channel_pos, params['num_channels_to_compare'], 
            params['max_spikes_for_unit'], params['max_spikes_for_nn'], params['n_neighbors'],
            params['do_parallel']
        )

        print("Calculating silhouette score")
        silhouette_score = calculate_silhouette_score(
            spike_clusters, spike_templates, total_units, pc_features,
            pc_feature_ind, params['n_silhouette']
        )

        print("Calculating drift metrics")
        max_drift, cumulative_drift = calculate_drift_metrics(
            spike_times, spike_clusters, spike_templates, total_units,
            pc_features, pc_feature_ind, params['drift_metrics_interval_s'],
            params['drift_metrics_min_spikes_per_interval']
        )
                                                       
        metrics.update({
            'isolation_distance': isolation_distance,
            'l_ratio': l_ratio,
            'd_prime': d_prime,
            'nn_hit_rate': nn_hit_rate,
            'nn_miss_rate': nn_miss_rate,
            'silhouette_score': silhouette_score,
            'max_drift': max_drift,
            'cumulative_drift': cumulative_drift
        })

    return pd.DataFrame(metrics)


# ===============================================================

# HELPER FUNCTIONS TO LOOP THROUGH CLUSTERS:

# ===============================================================
def calculate_isi_violations(spike_times, spike_clusters, total_units, isi_threshold, min_isi):
    cluster_ids = np.unique(spike_clusters)
    viol_rates = np.zeros(total_units)
    num_viol = np.zeros(total_units)

    min_time, max_time = np.min(spike_times), np.max(spike_times)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_mask = spike_clusters == cluster_id
        viol_rates[idx], num_viol[idx] = isi_violations(
            spike_times[cluster_mask],
            min_time=min_time,
            max_time=max_time,
            isi_threshold=isi_threshold,
            min_isi=min_isi
        )

    return viol_rates, num_viol


def calculate_isi_violations_corrected(spike_times, spike_clusters, total_units, isi_threshold, min_isi):
    cluster_ids = np.unique(spike_clusters)
    viol_rates = np.zeros(total_units)

    min_time, max_time = np.min(spike_times), np.max(spike_times)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_mask = spike_clusters == cluster_id
        viol_rates[idx] = isi_violations_corrected(
            spike_times[cluster_mask],
            min_time=min_time,
            max_time=max_time,
            isi_threshold=isi_threshold,
            min_isi=min_isi
        )

    return viol_rates


def calculate_presence_ratio(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    total_units: int
) -> np.ndarray:
    cluster_ids = np.unique(spike_clusters)
    ratios = np.zeros(total_units)
    min_time, max_time = np.min(spike_times), np.max(spike_times)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_mask = spike_clusters == cluster_id
        ratios[idx] = presence_ratio(
            spike_times[cluster_mask],
            min_time=min_time,
            max_time=max_time
        )

    return ratios


def calculate_firing_rate(spike_times, spike_clusters, total_units):
    cluster_ids = np.unique(spike_clusters)
    firing_rates = np.zeros(total_units)
    min_time, max_time = np.min(spike_times), np.max(spike_times)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_mask = spike_clusters == cluster_id
        firing_rates[idx] = firing_rate(spike_times[cluster_mask],
                                        min_time=min_time,
                                        max_time=max_time)

    return firing_rates


def calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units):
    cluster_ids = np.unique(spike_clusters)
    amplitude_cutoffs = np.zeros(total_units)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_mask = spike_clusters == cluster_id
        amplitude_cutoffs[idx] = amplitude_cutoff(amplitudes[cluster_mask])

    return amplitude_cutoffs


def calculate_pc_metrics_one_cluster(cluster_peak_channels, idx, cluster_id, cluster_ids,
                                     nearest_channels, pc_features, pc_feature_ind,
                                     energy, pc1, cluster_channel_indices, cluster_id_inv,
                                     spike_clusters, spike_templates,
                                     max_spikes_for_cluster, max_spikes_for_nn, n_neighbors):
    peak_channel = cluster_peak_channels[idx]
    num_spikes_in_cluster = np.sum(spike_clusters == cluster_id)

    channels_to_use = nearest_channels[peak_channel]
    units_valid = cluster_ids[np.isin(cluster_channel_indices, channels_to_use).sum(axis=1) == nearest_channels.shape[1]]

    spike_counts = np.array([np.sum(spike_clusters == cluster_id2) for cluster_id2 in units_valid])
    relative_counts = spike_counts if num_spikes_in_cluster <= max_spikes_for_cluster else spike_counts / num_spikes_in_cluster * max_spikes_for_cluster

    all_pcs = []
    all_labels = []

    for idx2, cluster_id2 in enumerate(units_valid):
        subsample = int(relative_counts[idx2])

        pcs = get_unit_pcs(cluster_id2, spike_clusters, energy, pc1, cluster_channel_indices, cluster_id_inv, channels_to_use,
                           subsample=subsample)

        if pcs is not None and pcs.ndim == 2:
            labels = np.full(pcs.shape[0], cluster_id2)
            all_pcs.append(pcs)
            all_labels.append(labels)

    if all_pcs:
        all_pcs = np.concatenate(all_pcs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_pcs = np.zeros((0, pc_features.shape[1], len(channels_to_use)))
        all_labels = np.zeros(0)

    if (all_pcs.shape[0] > 10 and not np.all(all_labels == cluster_id) and
            (all_labels == cluster_id).sum() > 20 and channels_to_use.size > 0):
        all_pcs = all_pcs.reshape(all_pcs.shape[0], -1)
        isolation_distance, l_ratio = mahalanobis_metrics(all_pcs, all_labels, cluster_id)
        d_prime = lda_metrics(all_pcs, all_labels, cluster_id)
        nn_hit_rate, nn_miss_rate = nearest_neighbors_metrics(all_pcs, all_labels, cluster_id,
                                                              max_spikes_for_nn, n_neighbors)
    else:
        isolation_distance = d_prime = nn_miss_rate = nn_hit_rate = l_ratio = np.nan

    return isolation_distance, l_ratio, d_prime, nn_hit_rate, nn_miss_rate


def calculate_pc_metrics(spike_clusters,
                         spike_templates,
                         pc_features,
                         pc_feature_ind,
                         energy,
                         pc1,
                         waveform_idx,
                         cluster_id_inv,
                         channel_pos,
                         num_channels_to_compare,
                         max_spikes_for_cluster,
                         max_spikes_for_nn,
                         n_neighbors,
                         do_parallel=True):
    """
    Calculate PC-based metrics for all clusters.

    Parameters:
    -----------
    spike_clusters : np.ndarray
        Cluster IDs for all spikes.
    spike_templates : np.ndarray
        Template IDs for all spikes.
    pc_features : np.ndarray
        PC features for all spikes.
    pc_feature_ind : np.ndarray
        Indices of PC features.
    energy : np.ndarray
        Energy of all spikes.
    pc1 : np.ndarray
        PC1 of all spikes.
    waveform_idx : np.ndarray
        Channel indices of all templates.
    cluster_id_inv : np.ndarray
        Inverse mapping of cluster IDs.
    channel_pos : np.ndarray
        Positions of channels.
    num_channels_to_compare : int
        Number of channels to compare.
    max_spikes_for_cluster : int
        Maximum number of spikes for each cluster.
    max_spikes_for_nn : int
        Maximum number of spikes for nearest neighbors.
    n_neighbors : int
        Number of neighbors.
    do_parallel : bool, optional
        Whether to use parallel processing. Default is True.

    Returns:
    --------
    tuple of np.ndarray
        Isolation distances, L-ratios, d-primes, nearest-neighbor hit rates, and miss rates.
    """
    all_distances = np.linalg.norm(channel_pos[:, np.newaxis] - channel_pos, axis=2)
    nearest_channels = np.argsort(all_distances, axis=1)[:, :num_channels_to_compare]

    cluster_ids = np.unique(spike_clusters)
    template_ids = np.unique(spike_templates)

    template_peak_channels = np.zeros(len(template_ids), dtype='uint16')
    cluster_peak_channels = np.zeros(len(cluster_ids), dtype='uint16')


    for idx, template_id in enumerate(template_ids):
        for_template = np.squeeze(spike_templates == template_id)
        pc_max = np.argmax(np.mean(pc_features[for_template, 0, :], 0))
        template_peak_channels[idx] = pc_feature_ind[template_id, pc_max]

    for idx, cluster_id in enumerate(cluster_ids):
        for_unit = np.squeeze(spike_clusters == cluster_id)
        templates_for_unit = np.unique(spike_templates[for_unit])
        template_positions = np.where(np.isin(template_ids, templates_for_unit))[0]
        cluster_peak_channels[idx] = np.median(template_peak_channels[template_positions])
    
    def process_cluster(idx, cluster_id):
        return calculate_pc_metrics_one_cluster(
            cluster_peak_channels, idx, cluster_id, cluster_ids,
            nearest_channels, pc_features, pc_feature_ind,
            energy, pc1, waveform_idx, cluster_id_inv,
            spike_clusters, spike_templates,
            max_spikes_for_cluster, max_spikes_for_nn, n_neighbors
        )

    if do_parallel:
        from joblib import Parallel, delayed
        meas = Parallel(n_jobs=-1, verbose=3)(
            delayed(process_cluster)(idx, cluster_id)
            for idx, cluster_id in enumerate(cluster_ids)
        )
    else:
        from tqdm import tqdm
        meas = [
            process_cluster(idx, cluster_id)
            for idx, cluster_id in tqdm(enumerate(cluster_ids), total=len(cluster_ids), desc='PC metrics')
        ]

    isolation_distances, l_ratios, d_primes, nn_hit_rates, nn_miss_rates = zip(*meas)

    return (np.array(isolation_distances), np.array(l_ratios), np.array(d_primes),
            np.array(nn_hit_rates), np.array(nn_miss_rates))


def calculate_silhouette_score(spike_clusters,
                               spike_templates,
                               total_units,
                               pc_features,
                               pc_feature_ind,
                               total_spikes,
                               do_parallel=True):

    def score_inner_loop(i, cluster_ids):
        """
        Helper to loop over cluster_ids in one dimension. We dont want to loop over both dimensions in parallel-
        that will create too much worker overhead
        Args:
            i: index of first dimension
            cluster_ids: iterable of cluster ids

        Returns: scores for dimension j

        """
        scores_1d = []
        for j in cluster_ids:
            if j > i:
                inds = np.in1d(cluster_labels, np.array([i, j]))
                X = all_pcs[inds, :]
                labels = cluster_labels[inds]

                # len(np.unique(labels))=1 Can happen if total_spikes is low:
                if (len(labels) > 2) and (len(np.unique(labels)) > 1):
                    scores_1d.append(silhouette_score(X, labels))
                else:
                    scores_1d.append(np.nan)
            else:
                scores_1d.append(np.nan)
        return scores_1d

    cluster_ids = np.unique(spike_clusters)

    random_spike_inds = np.random.permutation(spike_clusters.size)
    random_spike_inds = random_spike_inds[:total_spikes]
    num_pc_features = pc_features.shape[1]
    num_channels = np.max(pc_feature_ind) + 1

    all_pcs = np.zeros((total_spikes, num_channels * num_pc_features))

    for idx, i in enumerate(random_spike_inds):

        unit_id = spike_templates[i]
        channels = pc_feature_ind[unit_id, :]

        for j in range(0, num_pc_features):
            all_pcs[idx, channels + num_channels * j] = pc_features[i, j, :]

    cluster_labels = np.squeeze(spike_clusters[random_spike_inds])

    SS = np.empty((total_units, total_units))
    SS[:] = np.nan

    # Build lists
    if do_parallel:
        from joblib import Parallel, delayed
        scores = Parallel(n_jobs=-1, verbose=2)(delayed(score_inner_loop)(i, cluster_ids) for i in cluster_ids)
    else:
        scores = [score_inner_loop(i, cluster_ids) for i in cluster_ids]

    # Fill the 2d array
    for i, col_score in enumerate(scores):
        for j, one_score in enumerate(col_score):
            SS[i, j] = one_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = np.nanmin(SS, 0)
        b = np.nanmin(SS, 1)

    return np.array([np.nanmin([a, b]) for a, b in zip(a, b)])


def calculate_drift_metrics(spike_times,
                            spike_clusters,
                            spike_templates,
                            total_units,
                            pc_features,
                            pc_feature_ind,
                            interval_length,
                            min_spikes_per_interval,
                            do_parallel=True):
    def calc_one_cluster(cluster_id):
        """
        Helper to calculate drift for one cluster
        Args:
            cluster_id:

        Returns:
            max_drift, cumulative_drift
        """
        in_cluster = spike_clusters == cluster_id
        times_for_cluster = spike_times[in_cluster]
        depths_for_cluster = depths[in_cluster]

        median_depths = []

        for t1, t2 in zip(interval_starts, interval_ends):

            in_range = (times_for_cluster > t1) * (times_for_cluster < t2)

            if np.sum(in_range) >= min_spikes_per_interval:
                median_depths.append(np.median(depths_for_cluster[in_range]))
            else:
                median_depths.append(np.nan)

        median_depths = np.array(median_depths)
        max_drift = np.around(np.nanmax(median_depths) - np.nanmin(median_depths), 2)
        cumulative_drift = np.around(np.nansum(np.abs(np.diff(median_depths))), 2)
        return max_drift, cumulative_drift

    max_drifts = []
    cumulative_drifts = []

    depths = get_spike_depths(spike_templates, pc_features, pc_feature_ind)

    interval_starts = np.arange(np.min(spike_times), np.max(spike_times), interval_length)
    interval_ends = interval_starts + interval_length

    cluster_ids = np.unique(spike_clusters)

    if do_parallel:
        from joblib import Parallel, delayed
        meas = Parallel(n_jobs=-1, verbose=2)(delayed(calc_one_cluster)(cluster_id)
                                              for cluster_id in cluster_ids)
    else:
        meas = [calc_one_cluster(cluster_id) for cluster_id in cluster_ids]

    for max_drift, cumulative_drift in meas:
        max_drifts.append(max_drift)
        cumulative_drifts.append(cumulative_drift)
    return np.array(max_drifts), np.array(cumulative_drifts)


# ==========================================================

# IMPLEMENTATION OF ACTUAL METRICS:

# ==========================================================

def isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=0):
    """Calculate ISI violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    Modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz

    Args:
        spike_train (array): Spike times
        min_time (float): Minimum time for potential spikes
        max_time (float): Maximum time for potential spikes
        isi_threshold (float): Threshold for ISI violation
        min_isi (float, optional): Threshold for duplicate spikes. Defaults to 0.

    Returns:
        tuple: (fpRate, num_violations)
            fpRate (float): Rate of contaminating spikes as fraction of overall rate
                            0 = perfect, <0.5 = some contamination, >1.0 = lots of contamination
            num_violations (int): Total number of violations
    """
    spike_train = np.delete(spike_train, np.where(np.diff(spike_train) <= min_isi)[0] + 1)
    isis = np.diff(spike_train)

    num_spikes = len(spike_train)
    num_violations = np.sum(isis < isi_threshold)
    violation_time = 2 * num_spikes * (isi_threshold - min_isi)
    total_rate = firing_rate(spike_train, min_time, max_time)

    fpRate = (num_violations / violation_time) / total_rate

    return fpRate, num_violations


def isi_violations_corrected(spike_train, min_time, max_time, isi_threshold, min_isi=0):
    """
    Calculate ISI violations for a spike train with bias correction.

    This function was updated in September 2023 by Nick Steinmetz to correct
    two problems with the original implementation (text copied from
    https://github.com/cortex-lab/sortingQuality repo):

    1) The approximation previously used, which was chosen to avoid getting 
    imaginary results, wasn't accurate to the Hill et al paper on which this
    method was based, nor was it accurate to the correct solution to the problem.
    
    2) Hill et al also did not have the correct solution to the problem. The Hill
    paper used an expression derived from an earlier work (Meunier et al
    2003) which had assumed a special case: the "contamination" was itself
    only generated by a single neuron and therefore the contaminating spikes
    themselves had a refractory period. If instead the contaminating spikes
    are generated from a real Poisson process (as in the case of electrical
    noise or many nearby neurons generating the contamination), then the
    correct expression is different, as now calculated here. This expression
    is given in Llobet et al. bioRxiv 2022:
    
    https://www.biorxiv.org/content/10.1101/2022.02.08.479192v1.full.pdf

    In practice, the three methods (the real Hill equation, the original
    isi_violations calculation, and the correct equation implemented below)
    return almost identical values for contamination less than ~20%. They
    diverge strongly for 30% or more. For contamination levels above 50%
    (based on the original calculation), the corrected version is undefined
    due to a negative square root.

    Args:
        spike_train (array): Spike times
        min_time (float): Minimum time for potential spikes
        max_time (float): Maximum time for potential spikes
        isi_threshold (float): Threshold for ISI violation
        min_isi (float, optional): Threshold for duplicate spikes. Defaults to 0.

    Returns:
        fpRate (float): Contaminating spike rate as fraction of overall rate
                        0 = perfect, <0.5 = some contamination, >1.0 = high contamination
    """
    # Remove duplicate spikes
    unique_spikes = np.unique(spike_train)
    isis = np.diff(unique_spikes)
    duration = max_time - min_time

    num_spikes = len(unique_spikes)
    num_violations = np.sum(isis < isi_threshold)

    # Calculate corrected false positive rate
    fpRate = 1 - np.sqrt(1 - (num_violations * duration) /
                         (num_spikes**2 * (isi_threshold - min_isi)))

    return fpRate


def presence_ratio(spike_train: np.ndarray, min_time: float, max_time: float, num_bins: int = 100) -> float:
    """
    Calculate fraction of time the unit is present within an epoch.

    Args:
        spike_train (np.ndarray): Array of spike times.
        min_time (float): Minimum time for potential spikes.
        max_time (float): Maximum time for potential spikes.
        num_bins (int, optional): Number of bins to use for histogram. Defaults to 100.

    Returns:
        float: Fraction of time bins in which this unit is spiking.
    """
    h, _ = np.histogram(spike_train, np.linspace(min_time, max_time, num_bins))
    return np.sum(h > 0) / (num_bins - 1)


def firing_rate(spike_train: np.ndarray, min_time: float = None, max_time: float = None) -> float:
    """
    Calculate firing rate for a spike train.

    If no temporal bounds are specified, the first and last spike time are used.

    Args:
        spike_train (np.ndarray): Array of spike times in seconds
        min_time (float, optional): Time of first possible spike
        max_time (float, optional): Time of last possible spike

    Returns:
        float: Firing rate in Hz
    """
    duration = (max_time - min_time) if min_time is not None and max_time is not None else (np.max(spike_train) - np.min(spike_train))
    return spike_train.size / duration


def amplitude_cutoff(amplitudes: np.ndarray, num_histogram_bins: int = 500, histogram_smoothing_value: int = 3) -> float:
    """
    Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    Assumes the amplitude histogram is symmetric (not valid in the presence of drift).
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705.

    Args:
        amplitudes (np.ndarray): Array of amplitudes (don't need to be in physical units).
        num_histogram_bins (int, optional): Number of bins for histogram. Defaults to 500.
        histogram_smoothing_value (int, optional): Smoothing value for histogram. Defaults to 3.

    Returns:
        float: Fraction of missing spikes (0-0.5).
               If more than 50% of spikes are missing, an accurate estimate isn't possible.
    """
    h, b = np.histogram(amplitudes, bins=num_histogram_bins, density=True)
    pdf = gaussian_filter1d(h, sigma=histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:]) * bin_size

    return min(fraction_missing, 0.5)


def mahalanobis_metrics(all_pcs: np.ndarray, all_labels: np.ndarray, this_unit_id: int) -> Tuple[float, float]:
    """
    Calculate isolation distance and L-ratio using Mahalanobis distance.

    Based on metrics described in Schmitzer-Torbert et al. (2005) Neurosci 131: 1-11

    Args:
        all_pcs (np.ndarray): 2D array of PCs for all spikes (num_spikes x PCs)
        all_labels (np.ndarray): 1D array of cluster labels for all spikes
        this_unit_id (int): ID of the unit for which metrics will be calculated

    Returns:
        Tuple[float, float]: Isolation distance and L-ratio for this unit
    """
    pcs_for_this_unit = all_pcs[all_labels == this_unit_id]
    pcs_for_other_units = all_pcs[all_labels != this_unit_id]

    mean_value = np.mean(pcs_for_this_unit, axis=0, keepdims=True)

    try:
        VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
    except np.linalg.LinAlgError:  # Case of singular matrix
        return np.nan, np.nan

    mahalanobis_other = np.sort(cdist(mean_value, pcs_for_other_units, 'mahalanobis', VI=VI)[0])
    mahalanobis_self = np.sort(cdist(mean_value, pcs_for_this_unit, 'mahalanobis', VI=VI)[0])

    n = min(len(pcs_for_this_unit), len(pcs_for_other_units))

    if n < 2:
        return np.nan, np.nan

    dof = pcs_for_this_unit.shape[1]  # Number of features
    l_ratio = np.sum(1 - chi2.cdf(mahalanobis_other**2, dof)) / len(mahalanobis_self)
    isolation_distance = mahalanobis_other[n-1]**2

    return isolation_distance, l_ratio


def lda_metrics(all_pcs, all_labels, this_unit_id):
    """
    Calculates d-prime based on Linear Discriminant Analysis

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated

    Outputs:
    --------
    d_prime : float
        Isolation distance of this unit
    """

    y = all_labels == this_unit_id

    lda = LDA(n_components=1)
    X_flda = lda.fit_transform(all_pcs, y)

    flda_this_cluster = X_flda[y]
    flda_other_cluster = X_flda[~y]

    d_prime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster)) / np.sqrt(
        0.5 * (np.var(flda_this_cluster) + np.var(flda_other_cluster))
    )

    return d_prime


def nearest_neighbors_metrics(all_pcs, all_labels, this_unit_id, max_spikes_for_nn, n_neighbors):
    """
    Calculates unit contamination based on NearestNeighbors search in PCA space.

    Based on metrics described in Chung, Magland et al. (2017) Neuron 95: 1381-1394

    Parameters:
    -----------
    all_pcs : numpy.ndarray
        2D array of PCs for all spikes (num_spikes x PCs)
    all_labels : numpy.ndarray
        1D array of cluster labels for all spikes (num_spikes x 0)
    this_unit_id : int
        Number corresponding to unit for which these metrics will be calculated
    max_spikes_for_nn : int
        Number of spikes to use (calculation can be very slow when this number is >20000)
    n_neighbors : int
        Number of neighbors to use

    Returns:
    --------
    hit_rate : float
        Fraction of neighbors for target cluster that are also in target cluster
    miss_rate : float
        Fraction of neighbors outside target cluster that are in target cluster
    """

    total_spikes = all_pcs.shape[0]
    ratio = min(max_spikes_for_nn / total_spikes, 1)
    this_unit = all_labels == this_unit_id

    X = np.concatenate((all_pcs[this_unit], all_pcs[~this_unit]), axis=0)
    n = int(np.sum(this_unit) * ratio)

    if ratio < 1:
        indices = np.linspace(0, X.shape[0] - 1, int(X.shape[0] * ratio)).astype(int)
        X = X[indices]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    this_cluster_nearest = indices[:n, 1:].flatten()
    other_cluster_nearest = indices[n:, 1:].flatten()

    hit_rate = np.mean(this_cluster_nearest < n)
    miss_rate = np.mean(other_cluster_nearest < n)

    return hit_rate, miss_rate

# ==========================================================

# HELPER FUNCTIONS:

# ==========================================================

def features_intersect(pc_feature_ind, these_channels):
    """
    Take only the channels that have calculated features out of the ones we are interested in.
    This should reduce the occurrence of 'except IndexError' below.

    Args:
        pc_feature_ind (np.ndarray): Array of PC feature indices.
        these_channels (np.ndarray): Channels to use for calculating metrics.

    Returns:
        np.ndarray: Intersect of what's available in PCs and what's needed.
    """
    intersect = set(pc_feature_ind[these_channels[0], :])
    for cluster_id in these_channels:
        intersect &= set(pc_feature_ind[cluster_id, :])
    return np.array(list(intersect))


def get_unit_pcs(unit_id, spike_clusters, energy, pc1, cluster_channel_indices, cluster_id_inv, channels_to_use, subsample):
    """
    Return PC features for one unit

    Parameters:
    -----------
    unit_id : int
        ID for this unit
    spike_clusters : np.ndarray
        Cluster labels for each spike
    energy : np.ndarray
        Energy of all spikes
    pc1 : np.ndarray
        PC1 of all spikes
    cluster_channel_indices : np.ndarray
        Channels used for PC calculation for each unit
    cluster_id_inv : np.ndarray
        Inverse mapping of cluster IDs
    channels_to_use : np.ndarray
        Channels to use for calculating metrics
    subsample : int
        Maximum number of spikes to return

    Returns:
    --------
    np.ndarray or None
        PCs for one unit (num_spikes x num_PCs x num_channels) or None if no PCs are found
    """
    valid_spikes = ~np.isnan(energy[:, 0])
    inds_for_unit = np.where((spike_clusters == unit_id) & valid_spikes)[0]
    spikes_to_use = np.random.permutation(inds_for_unit)[:subsample]
    unique_template_ids = np.unique(spike_clusters[spikes_to_use])
    unit_PCs = []

    for template_id in unique_template_ids:
        index_mask = spikes_to_use[spike_clusters[spikes_to_use] == template_id]
        these_inds = cluster_channel_indices[cluster_id_inv[template_id]]

        pc_array = [
            pc1[index_mask, np.argwhere(these_inds == i)[0][0]]
            for i in channels_to_use if i in these_inds
        ]

        energy_array = [
            energy[index_mask, np.argwhere(these_inds == i)[0][0]]
            for i in channels_to_use if i in these_inds
        ]

        if len(pc_array) < len(channels_to_use):
            continue

        unit_PCs.append(np.stack(pc_array + energy_array, axis=-1))

    return np.concatenate(unit_PCs) if unit_PCs else None


def get_spike_depths(spike_templates, pc_features, pc_feature_ind):
    """
    Calculates the distance (in microns) of individual spikes from the probe tip.

    This implementation is based on Matlab code from github.com/cortex-lab/spikes.

    Parameters:
    -----------
    spike_templates : np.ndarray
        Template IDs for N spikes (N x 0).
    pc_features : np.ndarray
        PC features for each spike (N x channels x num_PCs).
    pc_feature_ind : np.ndarray
        Channels used for PC calculation for each unit (M x channels).

    Returns:
    --------
    np.ndarray
        Distance (in microns) from each spike waveform from the probe tip (N x 0).
    """
    pc_features_copy = np.clip(np.squeeze(pc_features[:, 0, :]), 0, None)
    pc_power = np.square(pc_features_copy)

    spike_feat_ind = pc_feature_ind[spike_templates, :]
    spike_depths = np.sum(spike_feat_ind * pc_power, axis=1) / np.sum(pc_power, axis=1)

    return spike_depths * 10