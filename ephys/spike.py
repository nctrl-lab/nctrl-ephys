# spike.align(time_spike, time_event, window=[-5, 5])
# spike.count_spike(time_spike, time_event, window=[-0.5, 0.5])
# spike.plot(time_spike, time_event, type_event, window=[-5, 5], reorder=1, binsize=0.01, resolution=10)

import numpy as np
from scipy import signal

def align(time_spike, time_event, window=[-5, 5]):
    n_event = len(time_event)
    time_aligned = np.empty(n_event, dtype=object)
    for i_event in range(n_event):
        if np.isnan(time_event[i_event]):
            time_aligned[i_event] = np.array([])
            continue
        in_time = (time_spike >= time_event[i_event] + window[0]) & (time_spike <= time_event[i_event] + window[1])
        time_aligned[i_event] = time_spike[in_time] - time_event[i_event]
    return time_aligned


def count_spike(time_spike, time_event, window=[-0.5, 0.5]):
    def count_spk(time_spike, time_event, window):
        if np.isnan(time_event):
            return np.nan        
        in_time = (time_spike >= time_event + window[0]) & (time_spike <= time_event + window[1])
        return np.sum(in_time).astype(np.float)

    return np.vectorize(lambda x: count_spk(time_spike, x, window))(time_event)

def raster(time_aligned, type_event, reorder=1):
    n_trial = len(type_event)
    n_type = np.max(type_event) + 1
    n_trial_type = np.bincount(type_event)
    cum_trial = np.concatenate([0, np.cumsum(n_trial_type)], axis=None)
    
    x = np.empty(n_type, dtype=object)
    y = x.copy()
    
    if time_aligned.size == 0:
        return x, y
    
    n_spike = np.vectorize(len)(time_aligned)
    for i_type in range(n_type):
        in_trial = type_event == i_type
        x[i_type] = np.concatenate(time_aligned[in_trial])
        
        n_spike_type = n_spike[in_trial]
        if reorder:
            y[i_type] = np.repeat(np.arange(1, n_trial_type[i_type] + 1), n_spike_type) + cum_trial[i_type]
        else:
            y[i_type] = np.repeat(np.arange(1, n_trial + 1)[in_trial], n_spike_type)
    
    return x, y

def psth(time_aligned, type_event, binsize=0.01, resolution=10, window=[-5, 5]):
    bins = np.arange(window[0], window[1], binsize)
    t = bins[:-1] + binsize/2
    n_type = np.max(type_event) + 1
    
    # make gaussian window
    window = signal.gaussian(5*resolution, resolution)
    window /= np.sum(window)
    
    bar = np.zeros((n_type, len(t)))
    conv = bar.copy()
    for i_type in range(n_type):
        in_trial = type_event == i_type
        time_spike_type = np.concatenate(time_aligned[in_trial])
        y, _ = np.histogram(time_spike_type, bins)
        bar[i_type, :] = y / binsize / np.sum(in_trial)
        conv[i_type, :] = np.convolve(bar[i_type, :], window, 'same')
    
    return t, bar, conv

def plot(time_spike, time_event, type_event, 
         window=[-5, 5], reorder=1, binsize=0.01, resolution=10):

    in_event = ~np.isnan(type_event)
    time_event = time_event[in_event]
    type_event = type_event[in_event].astype('int64')
    [type_unique, type_index] = np.unique(type_event, return_inverse = True)
    
    time_aligned = align(time_spike, time_event, window)
    x, y = raster(time_aligned, type_index, reorder)
    t, bar, conv = psth(time_aligned, type_index, binsize, resolution, window)
    
    return {'x': x, 'y': y, 'type': type_unique}, {'t': t, 'bar': bar, 'conv': conv, 'type': type_unique}

def get_latency(spike, event_onset, event_offset, duration=0.08, offset=0.3, min_latency=0.02, prob=0.05):
    assert(len(event_onset) == len(event_offset))
    n_event = len(event_onset)

    # spike time for event
    in_event = (spike >= event_onset[0]) & (spike < event_onset[0] + duration)
    spike_event = [spike[in_event] - event_onset[0]]
    spike_base = []
    for i_event in range(n_event):
        
        if i_event < n_event - 1:
            # spike time for base
            base = np.arange(event_offset[i_event] + offset, event_onset[i_event + 1], duration)
            n_base_temp = len(base) - 1
            if n_base_temp == 0:
                continue

            # spike time for event: only include events with enough offset
            in_event = (spike >= event_onset[i_event+1]) & (spike < event_onset[i_event+1] + duration)
            spike_event.append(spike[in_event] - event_onset[i_event+1])

            if n_base_temp > 50:
                base_index = np.random.permutation(np.arange(n_base_temp))[:50]
            else:
                base_index = np.arange(n_base_temp)

            for i_base in base_index:
                in_base = (spike >= base[i_base]) & (spike < base[i_base + 1])
                spike_base.append(spike[in_base] - base[i_base])


    spike_event_all = np.sort(np.concatenate(spike_event))
    count_event = np.arange(len(spike_event_all)) / len(spike_event)
    spike_base_all = np.sort(np.concatenate(spike_base))
    count_base = np.arange(len(spike_base_all)) / len(spike_base)

    out = {'time_event': spike_event_all, 'count_event': count_event, 
           'time_base': spike_base_all, 'count_base': count_base}
    return out
