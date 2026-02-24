# AFM Analysis

Loading, processing, and analysis of Atomic Force Microscopy (AFM) data collected on thin film samples using Asylum Research AFM systems. Data is stored in Igor Binary Wave (.ibw) format.

---

## AFM_Analysis_Notebook.ipynb

An exploratory notebook for processing AFM .ibw files. Steps through data loading, channel visualization, median and Gaussian filtering, roughness metric extraction (Sa, Sq), spatially filtered roughness decomposition, and 2D Power Spectral Density (PSD) analysis with inline plots at each stage.

## AFMAnalyzer.py

A production-ready OOP refactor of the notebook. Encapsulates the full analysis pipeline — .ibw parsing, channel extraction, filtering, roughness metrics, waviness/roughness decomposition, and 2D PSD computation — into a single `AFMAnalyzer` class. Designed for reproducible, scriptable analysis with methods for displaying and saving figures. The full pipeline can be executed in a single `run()` call.

```python
from AFMAnalyzer import AFMAnalyzer

analyzer = AFMAnalyzer("GV007_STO0000.ibw")
analyzer.run()

# Or call steps individually
analyzer.load_data()
analyzer.apply_gaussian_filter(sigma=1)
analyzer.compute_spatially_filtered_roughness(sigma=2)
analyzer.compute_psd_2d()
analyzer.save_psd("psd_2d.png", dpi=150)
```

## Output

`psd_2d.png` — 2D Power Spectral Density of the Gaussian-filtered height map (log scale). Symmetric lobes identify the underlying periodic step structure of the thin film surface.
