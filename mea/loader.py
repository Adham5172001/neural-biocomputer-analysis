"""MEA Data Loader — Author: Adham Aboulkheir | University of Essex | PhD Research"""
import numpy as np
from dataclasses import dataclass

@dataclass
class MEARecording:
    data: np.ndarray; fs: float; n_electrodes: int; duration: float; electrode_positions: np.ndarray = None

    @classmethod
    def simulate(cls, n_electrodes=64, duration=60.0, fs=20000.0, n_active=20, seed=42):
        np.random.seed(seed)
        n_samples = int(fs * duration)
        data = np.random.normal(0, 10, (n_electrodes, n_samples))
        for e in range(n_active):
            positions = np.random.choice(n_samples-100, np.random.randint(100, 500), replace=False)
            for pos in positions:
                amp = np.random.choice([-80, -60, -70])
                data[e, pos:pos+20] += amp * np.hanning(20)
        rows = cols = int(np.sqrt(n_electrodes))
        positions = np.array([[i*200, j*200] for i in range(rows) for j in range(cols)])[:n_electrodes]
        return cls(data=data, fs=fs, n_electrodes=n_electrodes, duration=duration, electrode_positions=positions)

    def __repr__(self):
        return f"MEARecording(n_electrodes={self.n_electrodes}, duration={self.duration:.1f}s, fs={self.fs:.0f}Hz)"

if __name__ == "__main__":
    rec = MEARecording.simulate(n_electrodes=64, duration=10.0, n_active=20)
    print(f"Recording: {rec}")
    print(f"Data shape: {rec.data.shape}")
    print(f"Signal range: [{rec.data.min():.1f}, {rec.data.max():.1f}]")
