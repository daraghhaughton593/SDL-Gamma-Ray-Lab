## SDL-Gamma-Ray-Lab
- SDL Gamma Ray Lab for Caimin, Dominic, Liam and Daragh

## Overview
The script found in this repo can be used to analyse the response of 3 different detectors to the radioactive stimuli found in the DATA folder. This analysis includes peak identification, energy calibration, energy resolution calculations, effiency studies and off-axis response. The three detectors used are:

1) CdTe Detector - This is a solid state detector that is primarily designed for use in medical environments (mammography, radiology). It is therefore more sensitive to x-ray radiation than the other two detectors.
2) BGO Detector - This is a scintillator primarily designed for use in particle physics and geophysical research.
3) NaI Detector - This is another scintillator that is designed for research in the field of spectroscopy. Wide range of applications in industry, security and nuclear power plant monitoring

## Using the Script
Requisites:
. User must ensure that 'gamma.py' script is located in the same directory as the DATA folder.
. All python libraries found in the list below must be installed.

The script can be called in the command line using the following format:
python gamma.py detector_name spectrum_plots

Here, the arguments detector_name and spectrum_plots are entered by the user depending on:
detector_name - the detector type desired ("CdTe", "BGO", "NaI")
spectrum_plots - if spectrum plots are wanted ("Plot"), otherwise no entry required


## Libraries Required
This script requires the following Python libraries:

- numpy      	Numerical computations and array handling
- matplotlib	Plotting and data visualization
- scipy	      Scientific computing (optimization, signal processing, curve fitting)
- yaml	      Reading and writing YAML configuration files
- os	      Operating system interfaces (file paths, environment variables)
- pathlib	Object-oriented filesystem paths
- re	      Regular expression operations
- lmfit	      Advanced curve fitting and model optimization
- pandas	Data handling and tabular data analysis
- sys	      System-specific parameters and command-line arguments

# Detectors used: 
- i) CdTe (Cadmium Telluride) - Distance: 10.6 cm
- ii) NaTI (Soidum Thallumide) - Distance: 16 cm
- iii) BGO (Bismuth Germanate) - Distance: 15 cm


