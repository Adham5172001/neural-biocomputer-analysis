"""Neural Biocomputer Analysis Demo — Author: Adham Aboulkheir | University of Essex"""
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from mea.loader import MEARecording
from scipy import signal as sp_signal

def bandpass_filter(data, fs=20000, low=300, high=3000):
    nyq = fs/2; b, a = sp_signal.butter(4, [low/nyq, high/nyq], btype="band")
    return sp_signal.filtfilt(b, a, data, axis=-1)

def detect_spikes(data, fs=20000, threshold_factor=5.0):
    spike_times = []
    for e in range(data.shape[0]):
        trace = data[e]; threshold = -threshold_factor * np.sqrt(np.mean(trace**2))
        crossings = np.where((trace[:-1] > threshold) & (trace[1:] <= threshold))[0]
        spike_times.append(crossings / fs)
    return spike_times

def main():
    print("Neural Biocomputer Analysis Demo")
    os.makedirs("outputs", exist_ok=True)
    rec = MEARecording.simulate(n_electrodes=64, duration=10.0, n_active=20, seed=42)
    print(f"  {rec}")
    filtered = bandpass_filter(rec.data, fs=rec.fs)
    spike_times = detect_spikes(filtered, fs=rec.fs)
    active = sum(1 for st in spike_times if len(st) > 0)
    total = sum(len(st) for st in spike_times)
    print(f"  Active electrodes: {active}/{rec.n_electrodes} | Total spikes: {total}")
    window = 0.1; n_windows = int(rec.duration / window)
    firing_rates = np.zeros((rec.n_electrodes, n_windows))
    for e, times in enumerate(spike_times):
        for w in range(n_windows):
            firing_rates[e, w] = np.sum((times >= w*window) & (times < (w+1)*window)) / window
    print(f"  Mean firing rate: {firing_rates.mean():.2f} Hz")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")
    for e, times in enumerate(spike_times[:20]):
        if len(times) > 0: axes[0].scatter(times, np.ones_like(times)*e, s=1, c="#00c9b1", alpha=0.6)
    axes[0].set_title("Spike Raster Plot (20 electrodes)", color="white"); axes[0].set_xlabel("Time (s)", color="white"); axes[0].set_ylabel("Electrode", color="white"); axes[0].tick_params(colors="white"); axes[0].grid(alpha=0.3, color="#21262d")
    mean_rates = firing_rates.mean(axis=1)
    if rec.electrode_positions is not None:
        pos = rec.electrode_positions
        sc = axes[1].scatter(pos[:,0], pos[:,1], c=mean_rates, cmap="plasma", s=80, alpha=0.9)
        plt.colorbar(sc, ax=axes[1], label="Firing Rate (Hz)")
    axes[1].set_title("Electrode Activity Heatmap", color="white"); axes[1].set_xlabel("X (um)", color="white"); axes[1].set_ylabel("Y (um)", color="white"); axes[1].tick_params(colors="white")
    axes[2].hist(mean_rates[mean_rates>0], bins=20, color="#00c9b1", alpha=0.85, edgecolor="none")
    axes[2].set_title("Firing Rate Distribution", color="white"); axes[2].set_xlabel("Mean Firing Rate (Hz)", color="white"); axes[2].set_ylabel("Count", color="white"); axes[2].tick_params(colors="white"); axes[2].grid(alpha=0.3, color="#21262d")
    plt.tight_layout()
    plt.savefig("outputs/neural_biocomputer_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved: outputs/neural_biocomputer_results.png")

if __name__ == "__main__":
    main()
