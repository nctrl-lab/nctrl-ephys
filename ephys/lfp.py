import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample

def filter(data, fs=500, f_low=None, f_high=None, order=3, axis=0):
    nyquist_freq = fs / 2

    if f_low is None and f_high is None:
        return data
    elif f_low is not None:
        passtype = 'lowpass'
        passband = f_low / nyquist_freq
    elif f_high is not None:
        passtype = 'highpass'
        passband = f_high / nyquist_freq
    else:
        passtype = 'bandpass'
        passband = [f_low / nyquist_freq, f_high / nyquist_freq]
    
    b, a = butter(order, passband, passtype)
    data_filt = filtfilt(b, a, data, axis=axis)
    return data_filt

def notch_filter(data, fs=500, f_notch=60, Q=30, axis=0):
    b, a = iirnotch(w0=f_notch, Q=Q, fs=fs)
    data_filt = filtfilt(b, a, data, axis=axis)
    return data_filt

def downsample(t, data, fs=2000, fs_target=500):
    # lowpass filter at 160 Hz (if target fs is 500 Hz)
    f_low_cutoff = fs_target * 0.32
    data_filt = filter(data, fs=fs, f_low=f_low_cutoff)

    # downsample
    data_ds, t_ds = resample(data_filt, int(fs_target / fs * data.shape[0]), t=t, axis=0)

    # notch filter at 60 Hz
    data_ds_notch = notch_filter(data_ds, fs=fs_target)

    return t_ds, data_ds_notch

def get_band(f, psd, f_center=None, is_sum=True):
    # get frequency bands (2-128 Hz)
    if f_center is None:
        f_center = 2**np.arange(1, 7.2, 0.2) # 2-128 Hz with log spacing
        f_log_center = np.log2(f_center)
    else:
        f_log_center = np.log2(f_center)
    f_log_gap = np.diff(f_log_center) / 2
    f_log_gap_low = np.concatenate([f_log_gap[0], f_log_gap])
    f_log_gap_high = np.concatenate([f_log_gap, f_log_gap[-1]])
    f_low = f_log_center - f_log_gap_low
    f_high = f_log_center + f_log_gap_high
    f_bands = np.column_stack([2**(f_low), 2**(f_high)])
    n_band = len(f_center)

    df = f[1] - f[0]
    psd_band = np.zeros(n_band)
    for i_band in range(n_band):
        in_band = (f >= f_bands[i_band, 0]) & (f <= f_bands[i_band, 1])
        if is_sum:
            psd_band[i_band] = np.sum(psd[in_band]) * df
        else:
            psd_band[i_band] = np.mean(psd[in_band])
    return f_center, psd_band