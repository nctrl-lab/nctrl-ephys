# spike.align(time_spike, time_event, window=[-5, 5])
# spike.count_spike(time_spike, time_event, window=[-0.5, 0.5])
# spike.plot(time_spike, time_event, type_event, window=[-5, 5], reorder=1, binsize=0.01, resolution=10)

import numpy as np
from scipy import signal

def align(time_spike, time_event, window=[-5, 5]):
    time_aligned = [time_spike[(time_spike >= te + window[0]) & (time_spike <= te + window[1])] - te if not np.isnan(te) else np.array([]) for te in time_event]
    return np.array(time_aligned, dtype=object)

def count_spike(time_spike, time_event, window=[-0.5, 0.5]):
    def count_spk(te):
        if np.isnan(te):
            return np.nan        
        return np.sum((time_spike >= te + window[0]) & (time_spike <= te + window[1])).astype(np.float)
    return np.vectorize(count_spk)(time_event)

def raster(time_aligned, type_event, reorder=1):
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
        x[i_type] = np.concatenate(time_aligned[in_trial])
        
        n_spike_type = n_spike[in_trial]
        y[i_type] = np.repeat(np.arange(1, n_trial_type[i_type] + 1) if reorder else np.arange(1, len(type_event) + 1)[in_trial], n_spike_type) + cum_trial[i_type]
    
    return x, y

def psth(time_aligned, type_event, binsize=0.01, resolution=10, window=[-5, 5]):
    bins = np.arange(window[0], window[1], binsize)
    t = bins[:-1] + binsize/2
    n_type = np.max(type_event) + 1
    
    # make gaussian window
    gauss_window = signal.gaussian(5*resolution, resolution)
    gauss_window /= np.sum(gauss_window)
    
    bar = np.zeros((n_type, len(t)))
    conv = np.zeros_like(bar)
    for i_type in range(n_type):
        in_trial = type_event == i_type
        time_spike_type = np.concatenate(time_aligned[in_trial])
        y, _ = np.histogram(time_spike_type, bins)
        bar[i_type, :] = y / binsize / np.sum(in_trial)
        conv[i_type, :] = np.convolve(bar[i_type, :], gauss_window, 'same')
    
    return t, bar, conv

def plot(time_spike, time_event, type_event, 
         window=[-5, 5], reorder=1, binsize=0.01, resolution=10):

    in_event = ~np.isnan(type_event)
    time_event = time_event[in_event]
    type_event = type_event[in_event].astype('int64')
    type_unique, type_index = np.unique(type_event, return_inverse=True)
    
    time_aligned = align(time_spike, time_event, window)
    x, y = raster(time_aligned, type_index, reorder)
    t, bar, conv = psth(time_aligned, type_index, binsize, resolution, window)
    
    return {'x': x, 'y': y, 'type': type_unique}, {'t': t, 'bar': bar, 'conv': conv, 'type': type_unique}

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
