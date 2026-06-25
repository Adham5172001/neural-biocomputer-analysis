# Neural Biocomputer Analysis Toolkit

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green?logo=numpy)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![PhD](https://img.shields.io/badge/University%20of%20Essex-PhD%20Research-purple)](https://essex.ac.uk)

A comprehensive analysis toolkit for processing and visualising electrophysiological data from Multi-Electrode Array (MEA) recordings of living neural biocomputers. Developed as part of my PhD research at the University of Essex.

## What are Neural Biocomputers?

Neural biocomputers are biological computing systems where living neurons — cultured from rodent cortical tissue — are grown on MEA chips. These chips contain hundreds of electrodes that can both record and stimulate neural activity. The neurons form networks that can process information, making them a promising platform for biological computation, drug discovery, and neuroprosthetics research.

## Toolkit Features

### Data Loading & Preprocessing
- Load HD-MEA recordings from HDF5 and CSV formats
- Bandpass filtering (300–3000 Hz for spike detection)
- Common average referencing (CAR) for noise reduction
- Electrode quality assessment and bad channel rejection

### Spike Detection
- Threshold-based detection (adaptive threshold: 5× RMS noise)
- Spike sorting using PCA + K-Means clustering
- Raster plot generation

### Network Analysis
- Firing rate computation (per electrode, per time window)
- Cross-correlation analysis between electrode pairs
- Network burst detection
- Functional connectivity mapping

### Visualisation
- Multi-electrode raster plots
- Electrode activity heatmaps
- Spike waveform overlays
- Network connectivity graphs

## Installation

```bash
git clone https://github.com/Adham5172001/neural-biocomputer-analysis.git
cd neural-biocomputer-analysis
pip install -r requirements.txt
```

## Quick Start

```python
from mea_toolkit import MEARecording, SpikeDetector, NetworkAnalyser

# Load recording
rec = MEARecording.from_hdf5("recording.h5")
print(f"Loaded: {rec.n_electrodes} electrodes, {rec.duration:.1f}s")

# Detect spikes
detector = SpikeDetector(threshold_factor=5.0)
spikes = detector.detect(rec)
print(f"Detected {spikes.total_count} spikes across {spikes.active_electrodes} electrodes")

# Analyse network activity
analyser = NetworkAnalyser()
firing_rates = analyser.compute_firing_rates(spikes, window_size=0.1)
connectivity = analyser.compute_connectivity(spikes, method="cross_correlation")

# Visualise
rec.plot_raster(spikes, title="Neural Activity Raster Plot")
analyser.plot_connectivity_map(connectivity)
```

## Data Sources

This toolkit was developed and validated using data from:
- **ETH Zurich LBB** (Küchler et al., 2025) — HD-MEA cortical cultures
- **Cortical Labs** — CL1 biological neural computing platform

## License

MIT License
