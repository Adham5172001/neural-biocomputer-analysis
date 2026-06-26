"""
Neural Network Analysis for MEA Recordings
Author: Adham Aboulkheir | University of Essex | PhD Research
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BurstEvent:
    start_time: float
    end_time: float
    n_electrodes: int
    mean_firing_rate: float
    duration: float


def detect_network_bursts(firing_rates: np.ndarray, fs_window: float = 0.1,
                           threshold_factor: float = 2.0) -> List[BurstEvent]:
    """
    Detect network-wide bursting events in MEA recordings.
    A burst is when the population firing rate exceeds threshold_factor * mean.
    """
    # Population firing rate (mean across all electrodes)
    pop_rate = firing_rates.mean(axis=0)
    threshold = pop_rate.mean() * threshold_factor

    bursts = []
    in_burst = False
    burst_start = 0

    for t, rate in enumerate(pop_rate):
        if rate > threshold and not in_burst:
            in_burst = True
            burst_start = t
        elif rate <= threshold and in_burst:
            in_burst = False
            burst_end = t
            duration = (burst_end - burst_start) * fs_window
            active_electrodes = int((firing_rates[:, burst_start:burst_end].mean(axis=1) > 1).sum())
            bursts.append(BurstEvent(
                start_time=burst_start * fs_window,
                end_time=burst_end * fs_window,
                n_electrodes=active_electrodes,
                mean_firing_rate=float(pop_rate[burst_start:burst_end].mean()),
                duration=duration
            ))

    return bursts


def compute_cross_correlation(spike_times_i: np.ndarray, spike_times_j: np.ndarray,
                               max_lag: float = 0.05, bin_size: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation between two electrode spike trains.
    Reveals functional connectivity between electrode pairs.
    """
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    lags = (bins[:-1] + bins[1:]) / 2
    counts = np.zeros(len(lags))

    for t_i in spike_times_i:
        diffs = spike_times_j - t_i
        diffs = diffs[(diffs >= -max_lag) & (diffs < max_lag)]
        indices = np.digitize(diffs, bins) - 1
        valid = (indices >= 0) & (indices < len(counts))
        np.add.at(counts, indices[valid], 1)

    return lags, counts


def compute_connectivity_matrix(spike_times: List[np.ndarray],
                                  n_electrodes: int = 20,
                                  max_lag: float = 0.02) -> np.ndarray:
    """
    Compute functional connectivity matrix between all electrode pairs.
    Entry (i, j) = peak cross-correlation between electrodes i and j.
    """
    n = min(n_electrodes, len(spike_times))
    connectivity = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if len(spike_times[i]) > 0 and len(spike_times[j]) > 0:
                lags, counts = compute_cross_correlation(
                    spike_times[i], spike_times[j], max_lag=max_lag
                )
                peak = float(counts.max()) / (len(spike_times[i]) + 1e-9)
                connectivity[i, j] = peak
                connectivity[j, i] = peak

    return connectivity


if __name__ == "__main__":
    print("Network Analysis Demo")
    print("=" * 40)

    np.random.seed(42)
    n_electrodes = 20
    duration = 60.0
    fs = 20000

    # Simulate spike times
    spike_times = []
    for e in range(n_electrodes):
        n_spikes = np.random.randint(50, 300)
        times = np.sort(np.random.uniform(0, duration, n_spikes))
        spike_times.append(times)

    # Compute firing rates
    window = 0.1
    n_windows = int(duration / window)
    firing_rates = np.zeros((n_electrodes, n_windows))
    for e, times in enumerate(spike_times):
        for w in range(n_windows):
            firing_rates[e, w] = np.sum((times >= w*window) & (times < (w+1)*window)) / window

    # Detect bursts
    bursts = detect_network_bursts(firing_rates)
    print(f"Network bursts detected: {len(bursts)}")
    if bursts:
        print(f"  Mean duration: {np.mean([b.duration for b in bursts]):.2f}s")
        print(f"  Mean electrodes per burst: {np.mean([b.n_electrodes for b in bursts]):.1f}")

    # Connectivity
    conn_matrix = compute_connectivity_matrix(spike_times, n_electrodes=10)
    print(f"\nConnectivity matrix: {conn_matrix.shape}")
    print(f"  Mean connectivity: {conn_matrix.mean():.4f}")
    print(f"  Max connectivity: {conn_matrix.max():.4f}")
    most_connected = np.unravel_index(conn_matrix.argmax(), conn_matrix.shape)
    print(f"  Most connected pair: Electrode_{most_connected[0]+1} <-> Electrode_{most_connected[1]+1}")
