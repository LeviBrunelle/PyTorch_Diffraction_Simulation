# PyTorch Lens Simulation

A small library of classes and utilities to carve and simulate
optical lenses in PyTorch using BPM/WPM methods.

For operation instructions, see https://docs.google.com/document/d/1dKjZ28fx72_DnELoRHUIq_JHhT7iG-vK/edit?usp=drive_link&ouid=116603507756496699408&rtpof=true&sd=true

## Contents

- `lens.py` – `Lens` and `CompoundLens` classes, to create lens objects for simulations 
- `utils.py` – All utility functions (carving, BPM/WPM propagators, FWHM calculators, plotting helpers, etc)  
- `simulation.py` – Simulation object to help manage steps and intermediate data
- `main.py` - script that will run the simulation (requires user-editing for config) *** NOT UPDATED FOR A WHILE, DO NOT USE ***
- `gui.py` – a Streamlit gui for easier configuration of simulation parameters (renders images more slowly)
- `requirements.txt` – pinned dependencies for reproducible installs.

## Installation

### 1. Navigate to your desired directory


### 2. Create & activate a virtual environment:
      # MacOS
      python3 -m venv .venv
      source .venv/bin/activate

      # Windows
      python3 -m venv .venv
      venv/Scripts/Activate.ps1

### 3. Clone this repo
      git clone https://github.com/Irradiant-Tech/pyTorch_Lens_Sim
      git checkout main
   

### 4. Enter folder
      cd YOUR_FILE_PATH/pyTorch_Lens_Sim


### 5. Install dependencies
      pip install -r requirements.txt


## Usage

### Launch the GUI
      # Ensure you're in the project directory and have your venv active
      streamlit run gui.py

### Close the GUI
      #Type ctrl+C in the terminal, then close the browser tab


## Future Improvements
- Update the main so users can run simulations manually without GUI
- Add classes and analysis tools for different types of optical components (aspherics, meta-optics)
- Enable x-offsets for lenses (All lenses are currently constructed symmetrically around x=0)
- Enable tilt angle for individual lenses
- Add more advanced focal length calculations
- Rasterize Matplotlib images to be more memory-friendly

