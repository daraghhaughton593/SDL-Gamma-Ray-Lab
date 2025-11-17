
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
from scipy.signal import find_peaks
import yaml
import os
from pathlib import Path
import re
import lmfit
import pandas as pd
import sys
from scipy.optimize import curve_fit

def load_yaml(file_path):
    """Safely load a YAML file and return its contents.

    Inputs:
    file_path - location and name of YAML file

    Returns:
    data - all data within the YAML file
    """
    
    with open(file_path, 'r') as stream:
        data = yaml.safe_load(stream)
    return data


def filenumtoangle(filename):
  """
  Extracts the measurement angle from an entered filename

  Input:
  Filename - e.g. BGO_Ba15deg.Spe

  Returns:
  Degree of measurement used for file.
  """
  angles = re.findall(r'\d+', filename)

  return int(angles[0])

def file_storer(detector):
    """
    Finds all relevant files for detector type

    Input:
    Detector Name - BGO, CdTe

    Returns:
    files - list of files related to that detector
    

    """
    # The DATA folder is defined as the location to look for data files.
    base_dir = Path("DATA")/f"{detector} detector"

    # Data files related to the detector entered by user are gathered.
    files = [f for f in base_dir.rglob("*") if f.is_file()]

    return files

def file_reader(file):
    """
    Reads a spectrum data file (.spe or .mca)
    and returns counts normalized by live time (counts/s), channels and the exposure time.
    """
    # The type of file is saved to the variable 'ext'
    ext = os.path.splitext(file)[1].lower()

    # .SPE FILES
    if ext == ".spe":
        reading_data = False
        exposure_time = False
        counts = []
        skip_line = False

        with open(file, "r") as f:
            data = f.readlines()

        for line in data:
            line = line.strip()
            if not line:
                continue  

            # Detect start of exposure time section
            if "MEAS_TIM" in line:
                exposure_time = True
                continue

            # Read exposure time from next line
            elif exposure_time:
                stripped = line.split()
                try:
                    exp_time = float(stripped[0])
                except (ValueError, IndexError):
                    exp_time = 1.0
                # exposure_time boolean reset to False
                exposure_time = False
                continue

            # Detect start of data section
            elif "data" in line.lower():
                # Booleans for data extraction set to True
                reading_data = True
                skip_line = True
                continue  

            # Read actual counts
            elif reading_data:
                # One line skipped before data reading begins.
                if skip_line:
                    skip_line = False
                    continue

                if line.startswith("$"):
                    break

                try:
                    counts.append(float(line))
                except ValueError:
                    continue

        counts = np.asarray(counts)
        channels = np.arange(len(counts))

    # .MCA FILES
    else:
        reading_data = False
        counts = []

        with open(file, "r") as f:
            data = f.readlines()

        for line in data:
            line = line.strip()
            if not line:
                continue  

            if line.startswith("LIVE_TIME"):
                parts = line.split("-")
                if len(parts) > 1:
                    try:
                        exp_time = float(parts[1].strip())
                    except ValueError:
                        exp_time = 1.0
            # Different format for .mca files
            elif line.lower() == "<<data>>":
                reading_data = True
                continue  

            elif reading_data:
                if line.startswith("<<"):
                    break
                try:
                    counts.append(float(line))
                except ValueError:
                    continue

        counts = np.asarray(counts)
        channels = np.arange(len(counts))

    if exp_time <= 0:
        exp_time = 1.0  

    # Extracted data converted to pandas dataframe for use in subsequent functions.
    df = pd.DataFrame({
        "Channel": np.arange(len(counts), dtype=int),
        "Counts": np.array(counts, dtype=float)/exp_time,
        "Exposure_Time" : exp_time
    })

    return df

def Gau1(data, a, b, c):
  """
  returns the value of a Gaussian fit with constants a,b and c.

  Inputs:
  data (series) - data to be fitted.
  a,b,c - constants to be fitted

  Returns: 
  Value of Gaussian fit at each data point.
  """
    
  return (a * np.exp(-(data - b)**2 / (2 * c**2)))

#########

def line(x, a, b):
    
    """
    Fits a straight line to a set of data
    
    Inputs:
    x (series) - data points
    a, b - constants to be fitted.
    
    Returns:
    Value of line at each data point (x)
    """

    return a * x + b

#########

def paramsgenerator(counts, max, channel):
    """
    This function generates and saves initial input parameters for a Gaussian fitting process. It calculates estimates for the amplitude, centre wavelength, full-width at half maximum and
    standard deviation (sigma).

    Inputs:
    counts (series) - set of counts per second figures
    max (float) - previously identified channel of peak counts
    channel (series) - set of channel figures

    Returns:
    fit_params (lmfit.parameters) - A set of input parameters for use in Gaussian fitting 
    fwhm (float) - estimate of full-width at half maximum of the peak in question
    sig (float) - standard deviation derived from the fwhm 
    """
    fit_params = lmfit.Parameters()
    amp = counts.max()
    centre = max
    halfh = amp/2
    inds = np.where(counts >= halfh)[0]
    fwhm = channel.values[inds[-1]] - channel.values[inds[0]]
    sig = fwhm / 2.355
        
    fit_params.add('amp', value=amp, min = 0)
    fit_params.add('cen', value=centre, min = centre - 20, max = centre + 20)
    fit_params.add('wid', value=sig, min = 0.5)
        
    return fit_params, fwhm, sig

#########

def myfunc(params, x, data):
    """
    This function is defined to be fed to the lmfit.minimize function. It involves the unpacking of the initial parameters calculated by paramsgenerator and the generation of a Gaussian fit.
    The function then returns the residual found between this fit and the experimental data.

    Inputs:
    params (lmfit.parameters) - set of parameters as calculated by paramsgenerator
    x (series) - channel data (independent variable)
    data (series) - counts data (dependent variable) to be fitted with Gaussian.

    Returns:
    model - data (series) - set of residuals describing the discrepancy between the Gaussian fit and the experimental data.
    """
    amp = params['amp'].value
    cen = params['cen'].value
    wid = params['wid'].value

    model = Gau1(x, amp, cen, wid)

    return model - data

##########

def peakerrcalcer(peakE, centroiderrval, gain):
  """
  Function calculates and prints the peak energy and uncertainty

  Parameters:
    peakE : float
      Peak energy in keV
    centroiderrval : float
      uncertainty in centroid position in channel
    gain : float
      calibration value

  Returns:
    None
  """
  delE = centroiderrval * gain
  print(f'Peak energy =  {peakE:.3f} +/- {delE:.3f} (keV)')

#########

def RESerrorcalcer(R, sigval, sigerr, gain, peakE, peakEerr):
  """
    Function calculates and prints energy resolution and uncertainty.
    
    Parameters
    R : float
        Resolution (%)
    sigval : float
        Gaussian width sigma (channels)
    sigerr : float
        Uncertainty in sigma (channels)
    gain : float
        Energy calibration gain (keV/channel)
    peakE : float
        Peak energy in keV
    peakEerr : float
        Uncertainty in peak energy in keV
    
    Returns
    None
        Prints resolution with propagated uncertainty.
  """

  FWHM = sigval * 2.355
  delFWHM = sigerr * 2.355

  FWHMe = FWHM * gain
  FWHMeErr = delFWHM * gain

  delR = R * np.sqrt(((FWHMeErr/FWHMe)**2) + ((peakEerr/peakE)**2))

  print(f'Resolution = {R:.3f} +/- {delR:.3f}')

#########

def peakfinder(cps, channels, roi, source, detname, dec=None):
    """
    This function looks at a specified region of interest (roi) and identifies a peak therein. A Gaussian fit is then applied to this peak using lmfit, and the fitted 
    parameters along with their respective uncertainties are extracted. The fitted parameters are then used to produce a plot of the fit overlaid on the spectrum, which 
    can be viewed by the user if the Plot argument is entered.

    Inputs:
    cps (series) - the counts per second data series
    channels (series) - the channel numbers 
    roi (list) - list of relevant regions of interest from YAML file
    source (string) - the radioactive source that produced the spectrum
    detname (string) - the type of detector being used (BGO, NaI, CdTe, CdTe_Sample)
    dec - optional argument to produce plots of spectra
    """

    # The region of interest in defined as a mask.
    roi_mask = (channels >= roi[0]) & (channels <= roi[1])
    roi_channels = channels[roi_mask]
    roi_counts = cps[roi_mask]

    # The maximum counts figure within the masked region is determined and the corresponding channel is saved
    idx_max = roi_counts.values.argmax()
    channelone = roi_channels.iloc[idx_max]
    
    # The estimated parameters are generated 
    fit_params, fwhm1, sig1 = paramsgenerator(roi_counts, channelone, roi_channels)

    # A minimization fitting process is carried out, which returns fitted parameters that minimize the residuals of the fitted model.
    result = lmfit.minimize(myfunc, fit_params,
                            args=(roi_channels.values, roi_counts.values))

    # The fitted parameters are extracted from the minimization results
    centre = result.params['cen'].value
    amp = result.params['amp'].value
    wid = result.params['wid'].value

    # Uncertainties extracted if available
    amp_err = result.params['amp'].stderr
    cen_err = result.params['cen'].stderr
    wid_err = result.params['wid'].stderr

    # Gaussian data generated using fitted parameters.
    gauss = Gau1(roi_channels, amp, centre, wid)

    # Plots generated if requested.
    if dec == 'Plot':
        plt.figure(figsize=(12, 5))
        plt.plot(channels, cps, label='Spectrum')
        plt.scatter(roi_channels, roi_counts, color='green', s=30, label='ROI points')
        plt.plot(roi_channels, gauss, color='orange', lw=2, label='Gaussian fit')
        plt.axvline(centre, color='red', linestyle='--', label=f'Fitted peak: {centre:.1f}')    
        plt.xlabel('Channel')
        plt.ylabel('Counts / s')
        plt.title(f'{source} - {detname}')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Peak centroid in channel: {centre:.3f} ± {cen_err:.3f}")
        print(f"Amplitude: {amp:.3f} ± {amp_err:.3f}")
        print(f"Width (σ): {wid:.3f} ± {wid_err:.3f}")

    # Return parameters and uncertainties
    return {
        'centre': centre, 'centre_err': cen_err,
        'amplitude': amp, 'amplitude_err': amp_err,
        'width': wid, 'width_err': wid_err
    }


#######

def calibcurvecalc(channels, guessenergies, detname):
  """
  This function produces a calibration curve that illustrates the relation between the channel numbers and their corresponding energy values. 
  The relation should resemble a line, which is plotted

  Inputs:
  channels (series) - channel figures corresponding to identified peaks in spectra
  guessenergues (list) - expected energies of peaks identified (loaded in from YAML file)
  detname (string) - Name of detector type

  Returns:
  gain, offset (floats) - slope (gain) and intercept (offset) of fitted calibration curve relating channels and energy
  """
  # gain and offset calculated using a linear fit
  gain, offset = np.polyfit(channels, guessenergies, 1)

  print(f'Gain: {gain:.3f}, Offset: {offset:.3f}')

  # Each peak energy's distance to the fitted line is calculated and printed as a residual
  counter = 1
  for channel, energy in zip(channels, guessenergies):
    calcenergy = channel * gain + offset
    resid = calcenergy  - energy

    print(f'Peak: {counter} | Residual: {resid}')
    counter += 1

  chanrange = np.linspace(0, max(channels) * 1.1, 1000)
  line = gain * chanrange + offset

  plt.figure(figsize=(6, 4))
  plt.scatter(channels, guessenergies, color='red', s=50, zorder=5, label='Calibration points')
  plt.xlabel('Channels')
  plt.ylabel('Energies (keV)')
  plt.title(f'Calibration curve for detector: {detname}')
  plt.plot(chanrange, line, 'b-', linewidth=2, label=f'E = {gain:.3f}×Ch + {offset:.3f}')
  plt.grid(True)
  plt.ylim(ymin=0)
  plt.legend()
  plt.show()

  return gain, offset


#######

def resfinder(counts,channel, T, maxchannel, title, detname, gain, offset):
  """
  The resolution of each peak is calculated by dividing the fwhm by the intensity of the peak in question. 
  """

  energy = channel * gain + offset

  filteredonechannel = (channel <= (maxchannel + T)) & (channel >= (maxchannel - T))
  filteredonecounts = counts[filteredonechannel]
  filteredonechannel = channel[filteredonechannel]

  fit_params, fwhm1, sig1 = paramsgenerator(filteredonecounts, maxchannel, filteredonechannel)

  result = lmfit.minimize(myfunc, fit_params,
                          args=(filteredonechannel, filteredonecounts))

  print("Fit success:", result.success)
  print(f"Initial FWHM guess: {fwhm1:.3f} channels")

  centre1 = result.params['cen'].value
  centre1err = result.params['cen'].stderr
  amp1 = result.params['amp'].value
  wid1 = result.params['wid'].value
  wid1err = result.params['wid'].stderr

  gauss1 = Gau1(filteredonechannel, amp1, centre1, wid1)

  centre1_idx = int(round(centre1))

  energyatpeak1 = centre1 * gain + offset

  fwhmkev = fwhm1 * gain

  print(f'Detector: {detname} | Isotope: {title}')

  print(f'Peak FWHM: {fwhmkev:.3f}')# | Peak Energy: {energyatpeak1:.3f}')

  Res = (fwhmkev / energyatpeak1) * 100

  energyerr = centre1err * gain
  peakerrcalcer(energyatpeak1, centre1err, gain)

  RESerrorcalcer(Res, wid1, wid1err, gain, energyatpeak1, energyerr)

  return Res

#######
def resvenergyplotter(Reslist, energieslist):
    """
    Plots resolution vs energy and fits ΔE^2 = a + bE + cE^2

    Inputs:
    Reslist (list) - relative resolution values (ΔE/E)
    energieslist (list) - corresponding energies in keV

    Returns:
    None (plots the resolutions for desired detector)
    """
    # Convert relative resolution to ΔE in keV
    deltaE_list = np.array(Reslist) * np.array(energieslist)
    deltaE2_list = deltaE_list**2

    # Filter peaks based on minimum separation (optional)
    min_sep = 50
    filtered_energies = []
    filtered_deltaE2 = []
    last_energy = -np.inf
    for E, dE2 in sorted(zip(energieslist, deltaE2_list)):
        if E - last_energy >= min_sep:
            filtered_energies.append(E)
            filtered_deltaE2.append(dE2)
            last_energy = E

    # Scatter plot of experimental ΔE^2
    plt.scatter(filtered_energies, np.sqrt(filtered_deltaE2), color='red', label='Experimental Data')

    try:
        # Fit ΔE^2 = a + bE + cE^2
        popt, pcov = curve_fit(lambda E, a, b, c: a + b*E + c*E**2,
                               filtered_energies, filtered_deltaE2, p0=[1,1,1], bounds=(0, np.inf))
        perr = np.sqrt(np.diag(pcov))

        # Plot fitted curve in ΔE (keV)
        E_fit = np.logspace(np.log10(min(filtered_energies)), np.log10(max(filtered_energies)), 200)
        deltaE_fit = np.sqrt(popt[0] + popt[1]*E_fit + popt[2]*E_fit**2)
        plt.plot(E_fit, deltaE_fit, color='blue')

        # Display fitted parameters in top-right corner
        textstr = (f"a = {popt[0]:.2e} ± {perr[0]:.2e}\n"
                   f"b = {popt[1]:.2e} ± {perr[1]:.2e}\n"
                   f"c = {popt[2]:.2e} ± {perr[2]:.2e}")
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except Exception as e:
        print("Fit failed:", e)

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.xlabel('Energy (keV)')
    plt.ylabel('ΔE (keV)')
    plt.title("Resolution vs Energy")
    plt.show()


#######

def absefffinder(countrates, livetime, activity):
  """
  Function calculates detector absolute efficiency and uncertainties

  Paremeters:
  countrates : array
    array of countrates
    
  activity : float
    Activity of source in bq

  Returns:
    abseff : float
      absolute efficiency of detector
    absefferr : float
      absolute efficiency error of detector

  Notes:
    Uses Poisson statistics for uncertainties
  """
  total_countrates = float(countrates.sum())

  countrateerr = np.sqrt(total_countrates)

  print(f'Count-rate = {countrate:.3f} +/- {countrateerr:.3f} cts/s')
  abseff = (countrate / (activity)) 
  absefferr = (countrateerr / (activity)) 
  print(f'Absolute Efficiency = {abseff * 100:.3f} +/- {absefferr * 100:.3f}%')
  print()

  return abseff, absefferr


def microCi_2_Bq(microCi):
    """
    A function that converts from micro Curies to Becquerels.

    Inputs:
    microCi - Float: activity of source in microCi

    Outputs:
    Bq - Float: activity of the source in Becquerels

    """
    Bq = microCi*37_000
    return Bq

def activity(Element):
    """
    Compute activites of a source at time of measurments.

    Input:
    Element - String: element you wish to compute activity for,
                      (Am, Ba, Co, Cs)

    Output:
    A - Float: The activity of the source at the time of measurement (05/11/25)

    """

    #load the data
    data = load_yaml('Source_Info.yaml')

    #load the activities and half-lives
    A0_list = data['Activities']
    halflives = data['Half-Life']

    #iterate over keys to get the element
    for key in A0_list:
        if key == Element:
            A0 = A0_list[key]
            A0 = float(A0[0])
            hl = halflives[key]
            hl = hl[0]
            break  # stop looping once found

    #warning if string input is not an element in the yaml file
    if A0_list is None:
        print('WARNING: No Detector found')
        return None


    lam = decayconst(hl)
    A0_bq = microCi_2_Bq(A0) #convert from micro Curies to bequerels
    t = 1_449_446_148 #number of seconds passed since A0 was measured
    A = A0_bq * np.exp(-lam * t)
    return A

def decayconst(halflife):
    """
    Calculates the Decay constant of an element once given its half-life in years.

    Inputs:
    halflife - Float: Halflife of element in years.

    Outputs:
    decay_constant - Float: Decay constant,  unit s^-1

    """

    halflifesec = halflife*365.25*24*60*60
    decay_constant = np.log(2)/ halflifesec

    return decay_constant



def effsvsenergies(absefficiencies, insefficiencies, energies):
  """
  Produces logarithmic plot of the absolute and intrinsic efficiencies found for each peak identified for a given detector.
  Fits a curve to the intrinsic efficiencies to visualise relation.

  Inputs: 
  absefficiencies (list) - set of absolute efficiencies returned by absefffinder
  insefficiencies (list) - set of corresponding intrinsic effiencies, once the solid angle of the detector is accounted for
  energies (list) - set of peak energies for fitting and plotting the relation curve

  Returns:
  None 
  """
  logenergy = np.log(energies)
  absefflog = np.log(absefficiencies)
  insefflog = np.log(insefficiencies)

  coeffs = np.polyfit(logenergy, insefflog, 2)

  espace = np.linspace(min(energies), max(energies), 100)
  effectivefit = coeffs[2] + coeffs[1]*np.log(espace) + coeffs[0]*(np.log(espace))**2
  expeffectfit = np.exp(effectivefit)

  plt.scatter(energies, absefficiencies, c = 'blue', label = 'Absolute efficiencies')
  plt.scatter(energies, insefficiencies, c = 'green', label = 'Intrinsic efficiencies')
  plt.plot(espace, expeffectfit, ':', color = 'red', label = 'Polyfit of intrinsic efficiencies')

  plt.xscale('log')
  plt.yscale('log')

  plt.title("Plot of Detector Efficiencies as a function of Energy")
  plt.xlabel('Energy Values (keV)')
  plt.ylabel('Efficiencies')
  plt.legend(loc = 'best')
  plt.grid(True, which='both')
  plt.show()




def Angular_Detector_size(diam, h, dist, theta):
    """
    Compute the approximate solid angle subtended by a cylindrical detector.

    Inputs:
    diam - Float: diameter of detector in m
    h - Float: height of the detetctor in m
    dist-Float: Distance to detector in m
    theta - Float: angle the detector is at in degrees

    Output:
    omega - Float: Angular size of detector in steridians

    """

    theta = theta*np.pi/180
    A = abs((np.pi*(diam/2)**2)*(np.cos(theta)))+abs((diam*h*np.sin(theta)))
    omega = A/(dist**2)
    return omega

def Solid_Angle(detector, theta):
    """
    Function that takes a detector type and orientation angle,
    and returns the detector’s solid angle as a fraction of the total 4π steradians
    (i.e., the fraction of the full spherical emission that the detector sees).

    Inputs:
    detector - String: type of detector used
    theta - Float: angle in degrees that the detector was placed

    Outputs:
    Frac - Float: Fraction of the full spherical emission that the detector sees
    """


    data = load_yaml('Source_Info.yaml')

    Distances = data['Distances']

    distance = None
    Sizes = None

    for key in Distances:
        if key == detector:
            distance = Distances[key]
            Sizes = data['Sizes'][detector]
            break  # stop looping once found

    if distance is None:
        print('WARNING: No Detector found')
        return None
    r,h = Sizes
    r, h = r/1000, h/1000
    Frac = (Angular_Detector_size(r,h,distance,theta))/(4*np.pi)
    return Frac


def angular_response(detector, yaml_info):
    """
    This function appraises the detectors response at various angles of source location. The source used for this was Caesium as it was the most active.
    Peak heights and resolutions (fwhm) are calculated and plotted as a function of offset angle (in degrees)

    Inputs:
    detector (string) - name of detector type
    yaml_info (lists) - set of lists containing all data relevant to the alignment of the detector setup and the areas of interest for the Caesium peak (662 keV)

    Returns:
    None (Plots of peak heights and fwhm with respect to the angle of deviation from on-axis position)
    """
    files = file_storer(detector)
    back_cps = None

    degrees, peak_heights, peak_errs, fwhms, fwhm_errs = [], [], [],[],[]


    for file in files:
        name = file.name.lower()
        if "background" in name:
            back_df = file_reader(file)
            back_cps = back_df.Counts
            continue

        # Angled measurements only taken for Caesium. Ignore other sources.
        if "cs" not in name:
            continue

        df = file_reader(file)
        cps = df.Counts
        ch = df.Channel
        bg_subtract = cps - back_cps

        deg = filenumtoangle(name)

        # Only one ROI for Caesium (662 keV)
        roi = yaml_info["ROIs"][detector]["Cs"][0]  
        if not roi:
            continue

        # ROI mask applied
        roi_mask = (ch >= roi[0]) & (ch <= roi[1])
        roi_counts = bg_subtract[roi_mask]
        roi_channels = ch[roi_mask]

        # Gaussian fitted to data to extract amplitude.
        peak_ch = roi_channels.iloc[roi_counts.values.argmax()]
        fit_params, fwhm_ch, sig = paramsgenerator(roi_counts, peak_ch, roi_channels)
        result = lmfit.minimize(myfunc, fit_params, args=(roi_channels.values, roi_counts.values))

        # Amplitude and FWHM of peak determined from Gaussian fit.
        amp = result.params['amp'].value
        amp_err = result.params['amp'].stderr
        cen = result.params['cen'].value
        wid = result.params['wid'].value
        wid_err = result.params['wid'].stderr

        fwhm = wid * 2.355
        fwhm_err = wid_err*2.355

        degrees.append(deg)

        peak_heights.append(amp)
        peak_errs.append(amp_err)

        fwhms.append(fwhm)
        fwhm_errs.append(fwhm_err)


    # Plot results
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.errorbar(degrees, peak_heights, yerr = peak_errs, fmt = 'o-', label='Peak height')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Peak height (counts)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2,1,2)
    plt.errorbar(degrees, fwhms ,  yerr = fwhm_errs, fmt = 'o-',  color='red', label='FWHM')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('FWHM (channels)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()




def main(detector, dec=None):
    """
    - Processes files for entered detector
    - Takes Counts and Channels
    - Subtracts Background
    - Identifies peaks in spectra
    - Performs Calibration
    - Performs Resolution calculations and presentation
    - Computes efficiencies for each identified peak and presents them
    - Appraises off-axis response of detector
    ChatGPT was consulted to ensure that each function received inputs in the correct format and returned outputs compatible for further processing
    """
    plt.close('all')

    # The files relevant to the detector entered are gathered and stored.
    files = file_storer(detector)

    # The YAML file data is loaded in
    yaml_info = load_yaml("Source_Info.yaml")

    count_rates = []
    channels = []
    degrees = []
    sources = []
    livetimes = []
    back_cps = None

    # A number of lists are occupied with appropriate data from data files.
    for file in files:
        name = file.name.lower()

        if "background" in name:
            back_df = file_reader(file)
            back_cps = back_df.Counts
            continue

        source = (name.split("_")[1][:2].capitalize())
        df = file_reader(file)
        cps = df.Counts
        channel = df.Channel
        livetime = df.Exposure_Time.iloc[0]
        bg_subtract = cps - back_cps



        degrees.append(filenumtoangle(name))
        count_rates.append(bg_subtract)
        channels.append(channel)
        sources.append(source) 
        livetimes.append(float(livetime))

    # on-axis information is stored separately for easier access
    on_axis_channels = [ch for ch, d in zip(channels,degrees) if d ==0]
    on_axis_counts = [cr for cr, d in zip(count_rates, degrees) if d ==0]
    on_axis_sources = [s for s, d in zip(sources, degrees) if d == 0]
    on_axis_livetimes = [t for t, d in zip(livetimes, degrees) if d ==0]

    # regions of interest for each source are loaded in from YAML data.
    source_rois = [yaml_info["ROIs"][detector][src] for src in on_axis_sources]

    centres = []

    unique_source_data = {}

    for cr, ch, s, t, d in zip(count_rates, channels, sources, livetimes, degrees):
        if d != 0:
            continue  # only 0° files
        if s not in unique_source_data:
            unique_source_data[s] = (cr, ch, float(t))

    found_peaks = []

    # Each region of interest from the YAML file is used to search the spectra and return the peaks location in terms of channel number.
    for s, (cr, ch, t) in unique_source_data.items():
        rois = yaml_info["ROIs"][detector][s]
        if not rois:
            continue
        for i, roi in enumerate(rois):
            if not roi:
                continue

            peak = peakfinder(cr, ch, roi, s, detector, dec)

            # Print parameters with uncertainties
            print(f"Source: {s}")
            print(f"  Centre = {peak['centre']:.3f} ± {peak['centre_err']:.3f}")
            print(f"  Amplitude = {peak['amplitude']:.3f} ± {peak['amplitude_err']:.3f}")
            print(f"  Width (σ) = {peak['width']:.3f} ± {peak['width_err']:.3f}\n")

            # The identified peaks are stored in dictionary form.
            peak_channel = peak['centre']
            peak_channel_unc = peak['centre_err']
            if peak_channel is not None:
                found_peaks.append({
                    'src':s,
                    'peak_idx': i,
                    'centre': peak_channel,
                    'centre_err': peak_channel_unc,
                    'cr': cr,
                    'ch':ch,
                    't': t
                })
    # Calibration Process
    centres = [f['centre'] for f in found_peaks]
    exp_energies = [yaml_info["Peaks"][f['src']][f['peak_idx']] for f in found_peaks]
    gain, offset = calibcurvecalc(centres, exp_energies, detector)


    # Compute solid angle of detector for on-axis sources.
    sol_angle = Solid_Angle(detector, 0)

    res_list, energy_list, absol_eff, absol_efferr = [], [], [], []
    
    # Resolutions and efficiencies are calculated for each fitted peak.
    for f in found_peaks:
        src = f['src']
        centre = f['centre']
        cr = f['cr']
        ch = f['ch']
        energy = yaml_info["Peaks"][src][f['peak_idx']]
        t = f['t']

        # Energy resolution
        Res = resfinder(cr, ch, 40, centre, src, detector, gain, offset)
        if Res is not None:
            res_list.append(Res)
            energy_list.append(energy)

        # Absolute efficiency
        activ = activity(src)
        a_eff, a_efferr = absefffinder(cr, t, activ)
        absol_eff.append(a_eff)
        absol_efferr.append(a_efferr)

    # Convert absolute to intrinsic efficiency
    intrin_eff = [a_eff / sol_angle for a_eff in absol_eff]
    intrin_efferr = [a_efferr / sol_angle for a_efferr in absol_efferr]

    for eff, err in zip(intrin_eff, intrin_efferr):
      print(f'Intrinsic Efficiency = {eff:.3} +/- {err:.3}%')

    # Plot resolution vs energy
    resvenergyplotter(res_list, energy_list)

    # Plot efficiency vs energy
    effsvsenergies(absol_eff, intrin_eff, energy_list)

    # Plot Peak heights and FWHMs w.r.t off-axis angle
    angular_response(detector, yaml_info)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        main(args[0])
    elif len(args) == 2:
        main(args[0], args[1])
    else:
        print("Usage: python gamma.py <detector> [plot]")













