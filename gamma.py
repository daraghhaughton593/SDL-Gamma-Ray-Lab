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
    """Safely load a YAML file and return its contents."""
    with open(file_path, 'r') as stream:
        data = yaml.safe_load(stream)
    return data


def filenumtoangle(filename):
  angles = re.findall(r'\d+', filename)

  return int(angles[0])

def file_storer(detector):
    """
    Finds all relevant files for detector type

    """
    # The DATA folder is defined as the location to look for data files.
    base_dir = Path("DATA")/f"{detector} detector"

    # Data files related to the detector entered by user are gathered.
    files = [f for f in base_dir.rglob("*") if f.is_file()]

    return files

def file_reader(file):
    """
    Reads a spectrum data file (.spe or similar text-based format)
    and returns counts normalized by live time (counts/s) and channels.
    """

    ext = os.path.splitext(file)[1].lower()
    exp_time = 1.0  # default live time

    # .SPE FILES --- --
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
                exposure_time = False
                continue

            # Detect start of data section
            elif "data" in line.lower():
                reading_data = True
                skip_line = True
                continue  

            # Read actual counts
            elif reading_data:
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

    df = pd.DataFrame({
        "Channel": np.arange(len(counts), dtype=int),
        "Counts": np.array(counts, dtype=float)/exp_time,
        "Exposure_Time" : exp_time
    })

    return df

def Gau1(data, a, b, c):
  return (a * np.exp(-(data - b)**2 / (2 * c**2)))

#########

def line(x, a, b):
  return a * x + b

#########

def paramsgenerator(counts, max, channel):
  fit_params = lmfit.Parameters()
  amp = counts.max()
  center = max
  halfh = amp/2
  inds = np.where(counts >= halfh)[0]
  fwhm = channel.values[inds[-1]] - channel.values[inds[0]]
  sig = fwhm / 2.355

  fit_params.add('amp', value=amp, min = 0)
  fit_params.add('cen', value=center, min = center - 20, max = center + 20)
  fit_params.add('wid', value=sig, min = 0.5)

  return fit_params, fwhm, sig

#########

def myfunc(params, x, data):
    amp = params['amp'].value
    cen = params['cen'].value
    wid = params['wid'].value

    model = Gau1(x, amp, cen, wid)

    return model - data

#########


def peakfinder(cps, channels, roi, source, detname, dec=None):

    roi_mask = (channels >= roi[0]) & (channels <= roi[1])
    roi_channels = channels[roi_mask]
    roi_counts = cps[roi_mask]

    idx_max = roi_counts.values.argmax()
    channelone = roi_channels.iloc[idx_max]

    fit_params, fwhm1, sig1 = paramsgenerator(roi_counts, channelone, roi_channels)

    result = lmfit.minimize(myfunc, fit_params,
                            args=(roi_channels.values, roi_counts.values))

    centre = result.params['cen'].value
    amp = result.params['amp'].value
    wid = result.params['wid'].value

    # Extract uncertainties if available
    amp_err = result.params['amp'].stderr
    cen_err = result.params['cen'].stderr
    wid_err = result.params['wid'].stderr

    gauss = Gau1(roi_channels, amp, centre, wid)

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
  gain, offset = np.polyfit(channels, guessenergies, 1)

  print(f'Gain: {gain:.3f}, Offset: {offset:.3f}')

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
  amp1 = result.params['amp'].value
  wid1 = result.params['wid'].value

  gauss1 = Gau1(filteredonechannel, amp1, centre1, wid1)

  centre1_idx = int(round(centre1))

  energyatpeak1 = centre1 * gain + offset

  fwhmkev = fwhm1 * gain

  print(f'Detector: {detname} | Isotope: {title}')

  print(f'Peak FWHM: {fwhmkev:.3f} | Peak Energy: {energyatpeak1:.3f}')

  Res = (fwhmkev / energyatpeak1) * 100
  print(f'Detector Resolution: {Res:.3f}%')

  return Res

#######

def res_approx(a,b,c,E):
    """
    This function returns the approximated peak resolution at a given peak energy.

    Inputs:
    a,b,c  - constants
    E - Energy (keV)

    Returns:
    Resolution (keV)
    """

    return np.sqrt(a/E**(2) + b/E +c)

def resvenergyplotter(Reslist, energieslist):
  min_sep = 50
  if len(Reslist) == len(energieslist):
    # Filter peaks based on minimum separation
    filtered_energies = []
    filtered_res = []

    last_energy = -np.inf
    for E, R in sorted(zip(energieslist, Reslist)):
        if E - last_energy >= min_sep:
            filtered_energies.append(E)
            filtered_res.append(R)
            last_energy = E

    plt.scatter(filtered_energies, filtered_res, label="Experimental Data", color='red')

    try:
        # Bounds are included to avoid square root of negative values.
        popt, pcov = curve_fit(
             res_approx, energieslist, Reslist,
            p0=[1, 1, 1], bounds=(0, np.inf)
            )
        perr = np.sqrt(np.diag(pcov))

        # Plot fitted curve
        E_fit = np.logspace(np.log10(min(energieslist)), np.log10(max(energieslist)), 200)
        Res_fit = res_approx(E_fit, *popt)
        plt.plot(E_fit, Res_fit, color='blue', label="Fitted Curve")

                    # Display fitted parameters + uncertainties
        textstr = (f"a = {popt[0]:.2e} ± {perr[0]:.2e}\n"
                    f"b = {popt[1]:.2e} ± {perr[1]:.2e}\n"
                    f"c = {popt[2]:.2e} ± {perr[2]:.2e}")
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    except Exception as e:
        print("Fit failed:", e)

  plt.scatter(energieslist, Reslist)
  plt.xscale('log')
  plt.yscale('log')
  plt.grid(True)

  plt.title("Resolutions for found Photopeaks")
  plt.xlabel('Known Energy Values (keV)')
  plt.ylabel('Resolutions')
  plt.show()


#######

def absefffinder(counts, livetime, activity):

  total_counts = float(counts.sum())
  lt = float(livetime)

  countrate = total_counts/lt

  print(f'Count-rate = {countrate:.3f} cts/s')
  abseff = countrate/ (activity)
  print(f'Absolute Efficiency = {abseff * 100:.3f}%')

  return abseff


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
    diam - Float: diameter of detector in mm
    h - Float: height of the detetctor in mm
    dist-Float: Distance to detector in m
    theta - Float: angle the detector is at in degrees

    Output:
    omega - Float: Angular size of detector in steridians

    """

    theta = theta*np.pi/180
    A = np.pi*(diam/2)**2*np.cos(theta)+diam*h*np.sin(theta)
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
    Frac = (Angular_Detector_size(r,h,distance,15))/(4*np.pi)
    return Frac


def angular_response(detector, yaml_info):
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
    plt.errorbar(degrees, fwhms,  yerr = fwhm_errs, fmt = 'o-',  color='red', label='FWHM')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('FWHM (channels)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()




def main(detector, dec=None):
    """
    Processes files for entered detector
    - Takes Counts and Channels
    - Subtracts Background
    - Performs Calibration
    """
    files = file_storer(detector)

    yaml_info = load_yaml("Source_Info.yaml")

    count_rates = []
    channels = []
    degrees = []
    sources = []
    livetimes = []
    back_cps = None

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

    on_axis_channels = [ch for ch, d in zip(channels,degrees) if d ==0]
    on_axis_counts = [cr for cr, d in zip(count_rates, degrees) if d ==0]
    on_axis_sources = [s for s, d in zip(sources, degrees) if d == 0]
    on_axis_livetimes = [t for t, d in zip(livetimes, degrees) if d ==0]


    source_rois = [yaml_info["ROIs"][detector][src] for src in on_axis_sources]

    centres = []
    seen = set()


    unique_source_data = {}

    for cr, ch, s, t, d in zip(count_rates, channels, sources, livetimes, degrees):
        if d != 0:
            continue  # only 0° files
        if s not in unique_source_data:
            unique_source_data[s] = (cr, ch, float(t))


    found_peaks = []

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

    res_list, energy_list, absol_eff = [], [], []

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
        a_eff = absefffinder(cr, t, activ)
        absol_eff.append(a_eff)

    # Convert absolute to intrinsic efficiency
    intrin_eff = [a_eff / sol_angle for a_eff in absol_eff]

    # Plot resolution vs energy
    resvenergyplotter(res_list, energy_list)

    # Plot efficiency vs energy
    effsvsenergies(absol_eff, intrin_eff, energy_list)


    angular_response(detector, yaml_info)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        main(args[0])
    elif len(args) == 2:
        main(args[0], args[1])
    else:
        print("Usage: python gamma.py <detector> [plot]")



