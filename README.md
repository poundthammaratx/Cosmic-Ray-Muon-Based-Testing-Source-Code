# Cosmic-Ray Muon-Based Testing Source Code

Source code for cosmic-ray muon–based testing and validation of optical modules used in large-scale neutrino detectors.

---

## Overview

This repository contains the source code developed for a **cosmic-ray muon–based testing and validation framework** for optical modules used in large-scale neutrino detectors, with direct application to the **IceCube / IceCube-Gen2 LOM-16 optical modules**.

The analysis pipeline focuses on waveform-level inspection and detector-level characterization using naturally occurring atmospheric muons. The framework enables:

- Waveform processing and charge integration
- Coincidence-based muon event selection
- PMT- and module-level performance characterization
- Cross-checks of detector stability and response uniformity
- Figure generation for detector validation and reporting

The code was developed during the IceCube Summer Research Program (2025) and reflects a hardware-aware, data-driven approach suitable for detector testing and large-scale production validation.

---

## Repository Structure

```text
.
├── tools/                  # Analysis scripts (entry points)
│   ├── 00_hdf_data_inspector.py
│   ├── 01_analysis_quick_check.py
│   ├── 04_muon_coincidence.py
│   ├── 05_full_muon_analysis.py
│   ├── 09_ultimate_muon_analysis.py
│   └── ...
│
├── util/                   # Reusable utility modules
│   ├── hdf_reader.py
│   ├── eventHist.py
│   ├── plotting_functions.py
│   └── HDFWriterModuleInspection.py
│
├── README.md
├── .gitignore
└── LICENSE

```
## Quickstart

## Requirements

- Python ≥ 3.10
- NumPy, SciPy, Matplotlib, h5py (standard scientific Python stack)

## Installation

- Clone the repository:

-- git clone https://github.com/poundthammaratx/Cosmic-Ray-Muon-Based-Testing-Source-Code.git
-- cd Cosmic-Ray-Muon-Based-Testing-Source-Code

## create a virtual environment:

```text
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

Install dependencies:

```text
pip install -r requirements.txt

Run a quick analysis check:

python tools/01_analysis_quick_check.py
```

Note: The analysis scripts expect user-provided HDF5 input files.
Only lightweight examples or synthetic data should be used outside the IceCube environment.

## Data Policy

- This repository does not include raw or internal IceCube detector data.

- Raw experimental data, internal calibration files, and collaboration-restricted resources are explicitly excluded.

- The code is provided for transparency, reproducibility of analysis logic, and methodological reference.

- Users are expected to supply their own data in compatible formats (e.g. HDF5 files with equivalent structure).

This repository is not an official IceCube software product and is intended solely for research and educational purposes.

## Citation

If you use this code or parts of it in academic work, please cite the associated technical record:

Thammarat Yawisit, Design and Validation of a Cosmic Muon-Based Testing Method for
IceCube-Gen2 Optical Modules, Zenodo, 2025.
https://zenodo.org/records/18614095
