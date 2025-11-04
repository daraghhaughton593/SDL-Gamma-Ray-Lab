## SDL-Gamma-Ray-Lab
- SDL Gamma Ray Lab for Caimin, Dominic, Liam and Daragh

# For Calibration routine:

- i) use data from initial test runs
- ii) Pick most important peak, based on largest emission fraction for each source (maybe as a function arg, for modularity?)
- iii) fit gaussian to specified peak:
      - def Gau1(data, a, b, c, h):
          return (a * np.exp(-(data - b)**2 / (2 * c**2))) + h

- iv) extract channel value for peak centre from fit params (b in above code)

- Notes:
- - The calibration routine for each detector is basically the same process.

# Detectors used: 
- i) CdTe (Cadmium Telluride) - Distance: 10.6 cm
- ii) NaTI (Soidum Thallumide) - Distance: 16 cm
- iii) BGO (Bismuth Germanate) - Distance: 15 cm


