# Cosmic-Ray Muon-Based Testing Source Code

Source code for cosmic-ray muon–based testing and validation of optical modules
used in large-scale neutrino detectors.

---

## Overview

This repository contains the source code developed for a **cosmic-ray muon–based
testing and validation framework** for optical modules used in large-scale neutrino
detectors, with direct application to the **IceCube / IceCube-Gen2 LOM-16 optical modules**.

The analysis pipeline focuses on waveform-level inspection and detector-level
characterization using naturally occurring atmospheric muons. The framework enables:

- Waveform processing and charge integration  
- Coincidence-based muon event selection  
- PMT- and module-level performance characterization  
- Cross-checks of detector stability and response uniformity  
- Figure generation for detector validation and reporting  

The code was developed during the IceCube Summer Research Program (2025) and reflects
a hardware-aware, data-driven approach suitable for detector testing and large-scale
production validation.

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
