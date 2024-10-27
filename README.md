# SrRuO₃ Thin Film Deposition using Pulsed Laser Deposition (PLD)

This repository contains the source code, notebooks, and data analysis for the **deposition of Strontium Ruthenate (SRO) thin films on Strontium Titanate (STO)** substrates. It integrates structural analysis results obtained from **X-ray Diffraction (XRD)**, **Atomic Force Microscopy (AFM)**, and **plume dynamics imaging(PDI)** to correlate growth conditions with material quality. The project emphasizes the study of plume behavior during deposition, using **in-situ plume dynamic imaging** tools.

---

## Overview

The study focuses on:
- **Depositing SRO films on STO substrates** using the PLD technique.
- **Characterizing the film's structure and quality** through XRD, AFM, and PFM analysis.
- **Analyzing in-situ plume dynamics** to understand the relationship between plume behavior and film quality.

This repository provides:
- **Source code and Jupyter notebooks** for data visualization and analysis.
- **Scripts and examples for XRD and AFM data processing**, including domain structure and film quality assessment. Related package: (https://github.com/m3-learning/AFM-Learn.git, https://github.com/m3-learning/XRD-Learn.git)
- **Tools for analysis of plume dynamics imaging**. Related package: (https://github.com/m3-learning/Plume-Learn.git)

---

## Project Structure

```
/SrRuO3_Plume_Dynamics
│
├── notebooks/               # Jupyter notebooks for analysis and visualization
│   ├── xrd_analysis.ipynb   # XRD analysis of SRO thin films
│   ├── afm_analysis.ipynb   # AFM imaging and domain structure analysis
│   └── plume_analysis.ipynb # Plume dynamics and growth process analysis
│
├── data/                    # Raw and processed data from experiments
│   ├── AFM_PFM/             # AFM image data (.ibw, .png)
│   └── XRD_RSM/             # XRD data files (.xrdml)
│   └── Plumes/              # Plume imaging data (.tiff, .png)
│
├── figures/                 # Generated figures for result demonstration and visualization.
│
├── src/                     # Source code for data processing and analysis
│
├── requirements.txt         # Python dependencies
├── README.md                # Project overview (this file)
└── LICENSE                  # License information
```