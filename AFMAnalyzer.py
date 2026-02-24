import os
import igor2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter


class AFMAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.metadata = {}
        self.height = None
        self.defl = None
        self.amp = None
        self.phase = None
        self.filtered = None
        self.waviness = None
        self.roughness = None
        self.Sa = None
        self.Sq = None
        self.Sa_r = None
        self.Sq_r = None
        self.PSD = None

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_data(self):
        """Load .ibw file and extract channel data and metadata."""
        ibw = igor2.binarywave.load(self.file_path)
        self.data = ibw['wave']['wData']

        note = ibw['wave']['note'].decode('utf-8', errors='ignore')
        for line in note.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                self.metadata[key.strip()] = value.strip()

        self.height = self.data[:, :, 0] / 1e-9  # convert to nm
        self.defl   = self.data[:, :, 1]
        self.amp    = self.data[:, :, 2]
        self.phase  = self.data[:, :, 3]

        print("Data shape:", self.data.shape)
        return self

    def get_metadata(self):
        """Return metadata as a formatted DataFrame."""
        return pd.DataFrame(self.metadata.items(), columns=["Field", "Value"])

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    def apply_median_filter(self, size=3):
        """Apply a median filter to the height channel.

        Args:
            size: kernel size for the median filter (default: 3).
        """
        self.filtered = median_filter(self.height.astype(float), size=size)
        return self

    def apply_gaussian_filter(self, sigma=1):
        """Apply a Gaussian filter to the height channel.

        Args:
            sigma: standard deviation for Gaussian kernel (default: 1).
        """
        self.filtered = gaussian_filter(self.height.astype(float), sigma=sigma)
        return self

    # -------------------------------------------------------------------------
    # Metric Extraction
    # -------------------------------------------------------------------------

    def compute_roughness(self):
        """Compute Sa (mean absolute roughness) and Sq (RMS roughness) on raw height."""
        Z = self.height.astype(float)
        Z_mean = np.mean(Z)
        self.Sa = np.mean(np.abs(Z - Z_mean))
        self.Sq = np.sqrt(np.mean((Z - Z_mean) ** 2))
        print(f"Sa: {self.Sa.round(4)}")
        print(f"Sq: {self.Sq.round(4)}")
        return self

    def compute_spatially_filtered_roughness(self, sigma=2.0):
        """Decompose height into waviness and roughness using a Gaussian low-pass filter.
        Computes Sa and Sq on the short-wavelength roughness component.

        Args:
            sigma: Gaussian sigma for waviness estimation (default: 2).
        """
        Z = np.nan_to_num(self.height.astype(float))
        self.waviness = gaussian_filter(Z, sigma=sigma)
        self.roughness = Z - self.waviness

        r_mean = np.mean(self.roughness)
        self.Sa_r = np.mean(np.abs(self.roughness - r_mean))
        self.Sq_r = np.sqrt(np.mean((self.roughness - r_mean) ** 2))

        print(f"Spatially Filtered Sa (short-wavelength): {self.Sa_r.round(4)}")
        print(f"Spatially Filtered Sq (short-wavelength): {self.Sq_r.round(4)}")
        return self

    # -------------------------------------------------------------------------
    # PSD
    # -------------------------------------------------------------------------

    def compute_psd_2d(self):
        """Compute the 2D Power Spectral Density of the waviness map."""
        Z = self.waviness.astype(float)
        Z = Z - np.mean(Z)
        F_shift = np.fft.fftshift(np.fft.fft2(Z))
        self.PSD = np.abs(F_shift) ** 2
        return self

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_channels(self):
        """Plot all four raw AFM data channels."""
        channels = [self.height, self.defl, self.amp, self.phase]
        names = ['Height', 'Deflection', 'Amplitude', 'Phase']

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, channel, name in zip(axes, channels, names):
            ax.imshow(channel, cmap='viridis')
            ax.set_title(f"AFM {name} Map")
        plt.tight_layout()
        plt.show()

    def plot_filter_comparison(self, vmin=-2, vmax=5):
        """Plot original height map alongside the filtered result."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im0 = axes[0].imshow(self.height, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title("Original Height Map")
        plt.colorbar(im0, ax=axes[0]).set_label('Height (nm)')

        im1 = axes[1].imshow(self.filtered, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title("Filtered Height Map")
        plt.colorbar(im1, ax=axes[1]).set_label('Height (nm)', size=14, weight='bold')

        plt.tight_layout()
        plt.show()

    def plot_roughness_decomposition(self, vmin=-2, vmax=5):
        """Plot original height, waviness, and roughness components side by side."""
        names = ['Original', 'Waviness (Gaussian Filter)', 'Roughness (Original - Waviness)']
        plots = [self.height, self.waviness, self.roughness]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, data, name in zip(axes, plots, names):
            im = ax.imshow(data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(name)
            fig.colorbar(im, ax=ax, shrink=0.5).set_label('Height (nm)')
        plt.tight_layout()
        plt.show()

    def _build_psd_figure(self, vmin=2, vmax=6):
        """Build and return the 2D PSD figure."""
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(np.log10(self.PSD + 1e-15), cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title("2D Power Spectral Density (log scale)")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    def plot_psd(self, vmin=2, vmax=6):
        """Display the 2D Power Spectral Density plot."""
        fig = self._build_psd_figure(vmin, vmax)
        plt.show()

    def save_psd(self, filename="psd_2d.png", vmin=2, vmax=6, dpi=150):
        """Save the 2D Power Spectral Density plot to a .png in the current directory.

        Args:
            filename: output file name (default: 'psd_2d.png').
            vmin: colorscale min for log10(PSD) (default: 2).
            vmax: colorscale max for log10(PSD) (default: 6).
            dpi: image resolution (default: 150).
        """
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        fig = self._build_psd_figure(vmin, vmax)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 2D PSD plot to: {out_path}")

    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------

    def run(self):
        """Execute the full analysis pipeline and display all plots."""
        self.load_data()
        self.plot_channels()
        self.apply_gaussian_filter(sigma=1)
        self.plot_filter_comparison()
        self.compute_roughness()
        self.compute_spatially_filtered_roughness(sigma=2)
        self.plot_roughness_decomposition()
        self.compute_psd_2d()
        self.plot_psd()
        self.save_psd("psd_2d.png")
        return self


if __name__ == "__main__":
    import os
    file_path = os.path.join(os.path.dirname(__file__), "GV007_STO0000.ibw")
    analyzer = AFMAnalyzer(file_path)
    analyzer.run()
