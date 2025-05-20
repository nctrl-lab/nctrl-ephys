import numpy as np

def test_logrank(spikes, lasers, window=0.01, latency=0.000, n_base=None):
    """
    Perform log-rank test between spontaneous and stimulated spike trains.
    
    Args:
        spikes (numpy.ndarray): Array of spontaneous spike times
        lasers (numpy.ndarray): Array of laser times
        window (list): Window to censor spikes
        latency (float): Additional latency of the laser to the spike
    Returns:
        tuple: (p_value, times, H_base, H_test)

    Author: Dohoung Kim (2025. 5. 20)
    """
    spikes_base, spikes_laser = prepare_logrank(spikes, lasers, window, latency, n_base)
    return logrank(spikes_base, spikes_laser)

def test_salt(spikes, lasers, window=0.01, bin_size=0.001, latency=0.000, n_base=None):
    """
    Perform stimulus-associated spike latency test.
    """
    time_spikes = prepare_salt(spikes, lasers, window, latency, n_base)
    spikes_base = time_spikes[:, :-1]
    spikes_test = time_spikes[:, -1].reshape(-1, 1)
    return salt(spikes_test, spikes_base, window, bin_size)

def prepare_logrank(spikes, lasers, window=0.01, latency=0.000, n_base=None):
    """
    Prepare data for tagging test.

    Args:
        spikes (numpy.ndarray): Array of spontaneous spike times
        lasers (numpy.ndarray): Array of laser times
        window (float): Window to censor spikes
        n_base (int): Number of base trials
        latency (float): Additional latency of the laser to the spike
            ┌───────┐ ┌───────┐     ┌───────┐ Laser ┌────┐
            │base_1 │ │base_2 │ ... │base_n │   |   │test│
            └───────┘ └───────┘     └───────┘   |   └────┘ 
            |--win--| |--win--| ... |--win--|       |-win-|
    Returns:
        tuple: (spikes_base, spikes_laser)
            spikes_base: Base spike times with censoring
            spikes_laser: Laser spike times with censoring
    """
    if n_base is None:
        n_base = (np.diff(lasers).min() - 5 * window) // window # do not use the 5 windows after the laser

    test_times = lasers + latency
    base_times = np.sort((lasers[:, np.newaxis] + np.arange(-window*n_base, 0, window)).flatten())

    spikes_laser = np.full_like(test_times, window)
    censor_laser = np.ones_like(test_times)
    spikes_base = np.full_like(base_times, window)
    censor_base = np.ones_like(base_times)

    for i, laser in enumerate(test_times):
        mask_laser = (spikes >= laser) & (spikes <= laser + window)
        if np.any(mask_laser):
            spikes_laser[i] = spikes[mask_laser][0] - laser
            censor_laser[i] = 0

    for i, laser in enumerate(base_times):
        mask_base = (spikes >= laser) & (spikes <= laser + window)
        if np.any(mask_base):
            spikes_base[i] = spikes[mask_base][0] - laser
            censor_base[i] = 0
    
    spikes_base = np.column_stack((spikes_base, censor_base))
    spikes_laser = np.column_stack((spikes_laser, censor_laser))
    
    return spikes_base, spikes_laser

def prepare_salt(spikes, lasers, window=0.01, latency=0.000, n_base=None):
    if n_base is None:
        n_base = int((np.diff(lasers).min() - 5 * window) // window) # do not use the 5 windows after the laser

    n_laser = len(lasers)

    bins = np.arange(-window*n_base, 0, window)
    bins = np.concatenate((bins, [latency]))

    time_spikes = np.full((n_laser, n_base + 1), window)

    for i, laser in enumerate(lasers):
        for j, bin in enumerate(bins):
            time = laser + bin
            mask = (spikes >= time) & (spikes <= time + window)
            if np.any(mask):
                time_spikes[i, j] = spikes[mask][0] - time
    
    return time_spikes

def logrank(X, Y):
    """
    Perform log-rank test between two groups.
    
    Args:
        X (numpy.ndarray): First group data with shape (n,2). Usually, the base group.
            First column: duration of observation
            Second column: censoring (1: if censored observation)
        Y (numpy.ndarray): Second group data with shape (n,2). Usually, the test group.
            Same format as X
            
    Returns:
        tuple: (p_value, time, H1, H2) where:
            p_value: p-value of log-rank test
            time: time points
            H1: hazard function of group X
            H2: hazard function of group Y
    
    Author: Dohoung Kim (2025. 5. 20)
    """
    from scipy import stats
    
    n1, n2 = len(X), len(Y)
    if n1 == 0 or n2 == 0:
        return 1, np.array([]), np.array([]), np.array([])
    
    # Get unique time points and initialize arrays
    times = np.unique(np.concatenate([X[:,0], Y[:,0]]))
    n_times = len(times)
    
    m1 = np.zeros(n_times)  # events in group 1
    m2 = np.zeros(n_times)  # events in group 2
    n1_risk = np.zeros(n_times)  # at risk in group 1
    n2_risk = np.zeros(n_times)  # at risk in group 2
    
    X_times = X[:,0]
    Y_times = Y[:,0]
    X_events = X[:,1] == 0
    Y_events = Y[:,1] == 0
    
    for i, t in enumerate(times):
        m1[i] = np.sum(X_events[X_times == t])
        m2[i] = np.sum(Y_events[Y_times == t])
        n1_risk[i] = np.sum(X_times >= t)
        n2_risk[i] = np.sum(Y_times >= t)
    
    mask = (m1 + m2) > 0
    times = times[mask]
    m1 = m1[mask]
    m2 = m2[mask]
    n1_risk = n1_risk[mask]
    n2_risk = n2_risk[mask]
    
    total_risk = n1_risk + n2_risk
    total_events = m1 + m2
    risk_ratio = n1_risk / total_risk
    
    e1 = risk_ratio * total_events
    oe1 = m1 - e1
    
    v = total_events * risk_ratio * (1 - risk_ratio) * (total_risk - total_events) / (total_risk - 1)
    
    ch = np.sum(oe1)**2 / np.nansum(v)
    p_value = 1 - stats.chi2.cdf(ch, 1)
    
    H1 = 1 - np.cumprod(1 - m1/n1_risk)
    H2 = 1 - np.cumprod(1 - m2/n2_risk)
    
    return p_value, times, H1, H2

def salt(test, base, wn, dt):
    """
    Stimulus-associated spike latency test.
    
    Author: Dohoung Kim (2025. 5. 20)
    
    Parameters:
    -----------
    test : array-like
        Test time points
    base : array-like
        Baseline time points
    wn : float
        Window size
    dt : float
        Time step
        
    Returns:
    --------
    p : float
        P-value
    l : float
        Information difference
    """
    # Create time edges and combine time points
    edges = np.arange(0, wn + dt, dt)
    time = np.column_stack((base, test))
    nB = time.shape[1]
    
    # Calculate normalized histograms
    hst = np.apply_along_axis(lambda x: np.histogram(x, bins=edges)[0], 0, time)
    nhIsi = hst / np.sum(hst, axis=0)

    # Calculate Jensen-Shannon divergence
    jsd = np.full((nB, nB), np.nan)
    for iB in range(nB):
        for jB in range(iB+1, nB):
            D1 = nhIsi[:, iB]
            D2 = nhIsi[:, jB]
            jsd[iB, jB] = np.sqrt(2 * js_divergence(D1, D2))
    
    return make_p(jsd, nB)

def make_p(kld, kn):
    """
    Calculates p value from distance matrix.
    """
    
    pnhk = kld[:kn-1, :kn-1]
    nullhypkld = pnhk[~np.isnan(pnhk)]  # null hypothesis
    testkld = np.median(kld[:kn-1, kn-1])  # value to test
    sno = len(nullhypkld)  # sample size for null hyp. distribution
    p_value = np.sum(nullhypkld >= testkld) / sno
    Idiff = testkld - np.median(nullhypkld)  # information difference
    return p_value, Idiff

def js_divergence(P, Q):
    """
    Jensen-Shannon divergence.
    Calculates the Jensen-Shannon divergence of the two input distributions.
    """
    # Input validation
    if not np.allclose(np.sum(P), 1) or not np.allclose(np.sum(Q), 1):
        raise ValueError('Input arguments must be probability distributions.')
    if P.shape != Q.shape:
        raise ValueError('Input distributions must be of the same size.')
    
    # JS-divergence
    M = (P + Q) / 2
    D1 = kl_distance(P, M)
    D2 = kl_distance(Q, M)
    return (D1 + D2) / 2

def kl_distance(P, Q):
    """
    Kullback-Leibler distance.
    Calculates the Kullback-Leibler distance (information divergence) of the two input distributions.
    """
    # Input validation
    if not np.allclose(np.sum(P), 1) or not np.allclose(np.sum(Q), 1):
        raise ValueError('Input arguments must be probability distributions.')
    if P.shape != Q.shape:
        raise ValueError('Input distributions must be of the same size.')
    
    # KL-distance
    mask = (P > 0) & (Q > 0)
    if not np.any(mask):
        return np.inf
    
    P2 = P[mask]
    Q2 = Q[mask]
    return np.sum(P2 * np.log(P2 / Q2))