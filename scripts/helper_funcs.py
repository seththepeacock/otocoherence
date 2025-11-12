import numpy as np
import scipy.io as sio
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, kaiserord, firwin, lfilter, get_window
import os
import pickle
from phaseco import *
import phaseco as pc
import time
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm
from collections import defaultdict


def load_calc_colossogram(
    wf,
    wf_idx,
    wf_fn,
    wf_len_s,
    species,
    fs,
    filter_meth,
    pkl_folder,
    mode,
    tau,
    nfft,
    xi_min_s,
    xi_max_s,
    hop,
    win_meth,
    demean=True,
    scale=True,
    wf_pp=None,
    force_recalc_colossogram=0,
    plot_what_we_got=0,
    only_calc_new_coherences=0,
    const_N_pd=0,
    N_bs=0,
    f0s=None,
    nbacf=False,
):
    # Make sure this is a numpy array
    if f0s is not None:
        f0s = np.array(f0s)

    # Handle case where hop is a prop of tau
    if hop < 1:
        hop = int(round(hop * tau))

    # Build strings
    filter_str = get_filter_str(filter_meth)
    win_meth_str = pc.get_win_meth_str(win_meth)
    N_bs_str = "" if N_bs == 0 else f"N_bs={N_bs}, "
    const_N_pd_str = "" if const_N_pd else "N_pd=max, "
    nbacf_str = "" if not nbacf else f"NBACF, "
    f0s_str = (
        ""
        if f0s is None
        else f"f0s={np.array2string(f0s, formatter={'float' : lambda x: "%.0f" % x})}, "
    )
    nfft_str = "" if nfft is None else f"nfft={nfft}, "
    delta_xi_str = "" if xi_min_s == 0.001 else f"delta_xi={xi_min_s*1000:.1f}ms, "
    demean_str = "DM=True, " if demean else ""

    pkl_fn_id = rf"{species} {wf_idx}, mode={mode}, {win_meth_str}, hop={hop}, tau={tau}, {filter_str}, xi_max={xi_max_s*1000:.0f}ms, {nbacf_str}{delta_xi_str}{nfft_str}{f0s_str}{const_N_pd_str}{N_bs_str}{demean_str}wf_len={wf_len_s}s, wf={wf_fn.split('.')[0]}"
    pkl_fn = f"{pkl_fn_id} (Colossogram).pkl"

    # First, try to load
    pkl_fp = os.path.join(pkl_folder, pkl_fn)
    print(f"Processing '{pkl_fp}'")
    os.makedirs(pkl_folder, exist_ok=True)

    if os.path.exists(pkl_fp) and not force_recalc_colossogram:
        with open(pkl_fp, "rb") as file:
            (cgram_dict) = pickle.load(file)
        cgram_dict["fn_id"] = pkl_fn_id
        with open(pkl_fp, "wb") as file:
            pickle.dump(cgram_dict, file)
        if only_calc_new_coherences:
            cgram_dict["only_calc_new_coherences"] = 1

    else:
        # Now, we know they don't exist as a pickle, so we recalculate
        if (
            plot_what_we_got
        ):  # Unless plot_what we got, in which case we just end the func here
            return {"plot_what_we_got": 1}

        # First, process the wf (unless it's already processed)
        if wf_pp is None:

            # Crop wf
            wf = crop_wf(wf, fs, wf_len_s)

            if scale:  # Scale wf
                wf = scale_wf(wf, species)

            # Subtract mean
            if demean:
                wf = wf - np.mean(wf)

            # Apply filter (filter_meth could be None)
            wf = filter_wf(wf, fs, filter_meth)

            wf_pp = wf
        # If it's already been processed and passed in, just use it
        else:
            print("Calculating colossogram with prefiltered waveform!")

        # Then get colossogram!
        cgram_dict = pc.get_colossogram(
            wf_pp,
            fs,
            xis={"xi_min_s": xi_min_s, "xi_max_s": xi_max_s, "delta_xi_s": xi_min_s},
            hop=hop,
            tau=tau,
            nfft=nfft,
            win_meth=win_meth,
            mode=mode,
            const_N_pd=const_N_pd,
            N_bs=N_bs,
            f0s=f0s,
            nbacf=nbacf,
            return_dict=True,
        )
        # Add some extra keys
        extra_keys = {
            "fn_id": pkl_fn_id,
            "win_meth_str": win_meth_str,
            "filter_str": filter_str,
        }
        cgram_dict.update(extra_keys)

        with open(pkl_fp, "wb") as file:
            pickle.dump(cgram_dict, file)

    # Add the preprocessed waveform
    cgram_dict.update({"wf_pp": wf_pp})

    # Add powerweights
    cgram_dict.update({"mode": mode})

    # We now have colossogram_dict either from a saved pickle (new or old) or from the calculation; return it!
    return cgram_dict


def get_wf(wf_fn=None, species=None, wf_idx=None):
    if wf_fn is None:
        if species is None or wf_idx is None:
            raise ValueError("You must input either fn or species and idx!")
        else:
            wf_fn = get_fn(species, wf_idx)

    # Load wf
    data_folder = "data"
    wf_fp = os.path.join(data_folder, wf_fn)
    if species == "Tokay":
        wf = sio.loadmat(wf_fp)["wf"][0]
    elif species == "V Sim Human":
        wf = sio.loadmat(wf_fp)["oae"][:, 0]
    else:
        wf = sio.loadmat(wf_fp)["wf"][:, 0]

    # Get fs
    if wf_fn in ["Owl2R1.mat", "Owl7L1.mat"]:
        fs = 48000
    elif species == "V Sim Human":
        fs = 40000
    elif species == "Tokay":
        fs = 50000
    else:
        fs = 44100

    # Get peak list
    match wf_fn:

        # Anoles
        case "AC6rearSOAEwfB1.mat":  # 0
            good_peak_freqs = [
                1235,
                2153,
                3704,
                4500,
            ]
            bad_peak_freqs = []
        case "ACsb4rearSOAEwf1.mat":  # 1
            good_peak_freqs = [
                966,
                3023,
                3152,
                3954,
            ]
            bad_peak_freqs = []

        case "ACsb24rearSOAEwfA1.mat":  # 2
            good_peak_freqs = [
                1811,
                2178,
                3112,
                3478,
            ]
            bad_peak_freqs = []

        case "ACsb30learSOAEwfA2.mat":  # 3
            good_peak_freqs = [
                1798,
                2140,
                2417,
                2783,
            ]
            bad_peak_freqs = []
        # Humans
        case "ALrearSOAEwf1.mat":  # 0
            good_peak_freqs = [
                2662,
                2805,
                2942,
                3863,
            ]
            bad_peak_freqs = [
                3219,
            ]
        case "JIrearSOAEwf2.mat":  # 1
            good_peak_freqs = [
                2339,
                4051,
                5838,
                8309,
            ]
            bad_peak_freqs = [
                3400,
                8675,
            ]  # Note 8675 is only bad in C_xi^phi, it's good in C_xi^P

        case "LSrearSOAEwf1.mat":  # 2
            good_peak_freqs = [
                732,
                985,
                1634,
                2226,
            ]
            bad_peak_freqs = []

        case "TH13RearwaveformSOAE.mat":  # 3
            good_peak_freqs = [
                904,
                1521,
                2038,
                2694,
            ]
            bad_peak_freqs = []

        # Owls
        case "Owl2R1.mat":  # 0
            good_peak_freqs = [
                4351,
                7453,
                8452,
                9026,
            ]
            bad_peak_freqs = []

        case "Owl7L1.mat":  # 1
            good_peak_freqs = [
                6838,
                7901,
                8836,
                9258,
            ]
            bad_peak_freqs = []
        case "TAG6rearSOAEwf1.mat":  # 2
            good_peak_freqs = [
                5626,
                8096,
                8489,
                9865,
            ]
            bad_peak_freqs = []

        case "owl_TAG4learSOAEwf1.mat":  # 3
            good_peak_freqs = [
                4945,
                5768,
                7184,
                9633,
            ]
            bad_peak_freqs = []

        # Tokays
        case "tokay_GG1rearSOAEwf.mat":  # 0
            good_peak_freqs = [
                1343,
                1779,
                3650,
                4211,
            ]
            bad_peak_freqs = []
        case "tokay_GG2rearSOAEwf.mat":  # 1
            good_peak_freqs = [
                1364,
                1776,
                3607,
                4395,
            ]
            bad_peak_freqs = []
        case "tokay_GG3rearSOAEwf.mat":  # 2
            good_peak_freqs = [
                1257,
                1837,
                2579,
                3568,
            ]
            bad_peak_freqs = []
        case "tokay_GG4rearSOAEwf.mat":  # 3
            good_peak_freqs = [
                1251,
                2591,
                3217,
                3583,
            ]
            bad_peak_freqs = []

    return wf, wf_fn, fs, np.array(good_peak_freqs), np.array(bad_peak_freqs)


def filter_wf(wf, fs, filter_meth):
    if filter_meth is not None:
        match filter_meth["type"]:
            case "spectral":
                wf = spectral_filter(wf, fs, filter_meth["cf"], type="hp")
            case "kaiser":
                wf = kaiser_filter(
                    wf, fs, filter_meth["cf"], filter_meth["df"], filter_meth["rip"]
                )
            case _:
                raise ValueError(f"{filter_meth['type']} is not a valid HPF type!")
    return wf


def crop_wf(wf, fs, wf_len_s):
    desired_wf_len = round(wf_len_s * fs)
    og_wf_len = len(wf)
    if og_wf_len < desired_wf_len:
        raise ValueError(f"Waveform is less than {wf_len_s}s long!")
    # Start index for the middle chunk
    start = max(0, (og_wf_len - desired_wf_len) // 2)
    wf_cropped = wf[start : start + desired_wf_len]

    return wf_cropped


def scale_wf_long_way(wf):
    # First, undo the mic amplifier gain
    gain = 40  # dB
    wf = wf * 10 ** (-gain / 20)
    gain = 40  # dB
    wf = wf * 10 ** (-gain / 20)
    # Then account for the calibration factor
    cal_factor = 0.84
    cal_factor = 0.84
    wf = wf / cal_factor
    # The waveform is now in units of volts, where 1 micro volt = 0dB SPL = 20 micropascals
    # Let's rescale so that now 1 waveform unit (*volt*) = 0dB SPL = 20 micropascals
    wf = wf * 1e6
    # Now, 20*log10(dft_mags(wf)) would directly be in dB SPL.

    # Finally, (optional), we'll just convert it directly to pascals by multiplying by 20 micropascals:
    wf_pa = wf * 20 * 1e-6
    # Now, using this version, we would have to do 20*np.log10(dft_mags(wf_pa) / (20*1e-6)) to get dB SPL.)

    return wf_pa


def scale_wf(wf, species):
    if species in ["Anole", "Human"]:
        # Proven this is equivalent to above
        factor = (20 * 0.01) / 0.84
        wf = wf * factor
    return wf


def get_fn(species, idx):
    match species:
        case "Anole":
            match idx:
                case 0:
                    return "AC6rearSOAEwfB1.mat"
                case 1:
                    return "ACsb4rearSOAEwf1.mat"
                case 2:
                    return "ACsb24rearSOAEwfA1.mat"
                case 3:
                    return "ACsb30learSOAEwfA2.mat"
        case "Tokay":
            return f"tokay_GG{idx+1}rearSOAEwf.mat"

        case "Human":
            match idx:
                case 0:
                    return "ALrearSOAEwf1.mat"
                case 1:
                    return "JIrearSOAEwf2.mat"
                case 2:
                    return "LSrearSOAEwf1.mat"
                case 3:
                    return "TH13RearwaveformSOAE.mat"
        case "Owl":
            match idx:
                case 0:
                    return "Owl2R1.mat"
                case 1:
                    return "Owl7L1.mat"
                case 2:
                    return "TAG6rearSOAEwf1.mat"
                case 3:
                    return "owl_TAG4learSOAEwf1.mat"
        case "V Sim Human":
            return "longMCsoaeL1_20dBdiff100dB_InpN1InpYN0gain85R1rs43.mat"
        case _:
            raise ValueError("Species must be 'Anole', 'Human', 'Tokay', or 'Owl'!")


def get_precalc_tau_from_bw(bw, fs, win_type, pkl_folder):
    pkl_fp = os.path.join(pkl_folder, "precalc_taus.pkl")
    key = (win_type, fs, bw)

    # Load or initialize dictionary
    if os.path.exists(pkl_fp):
        with open(pkl_fp, "rb") as file:
            d = pickle.load(file)
    else:
        print(f"Precalc tau dictionary not found at '{pkl_folder}', making a new one!")
        d = {}

    # Check if key exists
    if key in d:
        tau = d[key]
    else:
        print(
            f"Found precalc_tau dict but {key} hasn't been calculated! Calculating now..."
        )
        tau, _ = get_tau_from_bw(bw, win_type, fs, nfft=2**25, verbose=True)
        d[key] = tau
        with open(pkl_fp, "wb") as file:
            pickle.dump(d, file)

    return tau


def spectral_filter(wf, fs, cutoff_freq, type="hp"):
    """Filters waveform by zeroing out frequencies above/below cutoff frequency

    Parameters
    ------------
        wf: array
          waveform input array
        fs: int
          sample rate of waveform
        cutoff_freq: float
          cutoff frequency for filtering
        type: str, Optional
          Either 'hp' for high-pass or 'lp' for low-pass
    """
    fft_coefficients = np.fft.rfft(wf)
    frequencies = np.fft.rfftfreq(len(wf), d=1 / fs)

    if type == "hp":
        # Zero out coefficients from 0 Hz to cutoff_frequency Hz
        fft_coefficients[frequencies <= cutoff_freq] = 0
    elif type == "lp":
        # Zero out coefficients from cutoff_frequency Hz to Nyquist frequency
        fft_coefficients[frequencies >= cutoff_freq] = 0

    # Compute the inverse real-valued FFT (irfft)
    filtered_wf = np.fft.irfft(
        fft_coefficients, n=len(wf)
    )  # Ensure output length matches input

    return filtered_wf


def kaiser_filter(wf, fs, cf=300, df=50, rip=100):
    """
    Apply an FIR, linear phase filter designed with a Kaiser window.

    Parameters:
        wf (array): Input waveform.
        fs (float): Sampling rate (Hz).
        cf (float or tuple of floats): Cutoff freq (Hz); Two frequencies = BP, one freq = HP
        df (float): Transition bandwidth (Hz).
        rip (float): Max allowed ripple in dB; that is, abs(A(w) - D(w))) < 10**(-ripple/20)

    Returns:
        array: Filtered waveform.
    """
    print(f"Filtering wf with cf={cf}Hz, df={df}Hz, rip={rip}dB")
    start = time.time()
    # Compute filter parameters
    numtaps, beta = kaiserord(rip, df / (0.5 * fs))

    if numtaps % 2 == 0:
        numtaps += 1  # Make it odd for a HPF

    # Design the high-pass FIR filter
    taps = firwin(
        numtaps,
        cf,
        window=("kaiser", beta),
        fs=fs,
        pass_zero=False,  # Neither HPF or BPF want zero
    )

    # Apply filtering
    filtered_wf = lfilter(
        taps, [1.0], wf
    )  # b, the denominator, is 1 for no poles, only zeros = FIR
    stop = time.time()
    print(f"Filtering took {stop-start:.3f}s")
    return filtered_wf


def get_filter_str(filter_meth):
    if filter_meth is None:
        return "filter=None"
    match filter_meth["type"]:
        case "kaiser":
            if not isinstance(filter_meth["cf"], tuple):
                filter_str = rf"HPF=({filter_meth['cf']}Hz cf, {filter_meth['df']}Hz df, {filter_meth['rip']}dB rip)"
            else:
                filter_str = rf"BPF=({filter_meth['cf']}Hz cf, {filter_meth['df']}Hz df, {filter_meth['rip']}dB rip)"
        case "spectral":
            filter_str = rf"HPF=({filter_meth['cf']}Hz)"
    return filter_str


def get_hpbw(win_type, tau, fs, nfft=None):
    if nfft is None:
        nfft = tau * 8
    win = get_window(win_type, tau)
    win_psd = np.abs(rfft(win, nfft)) ** 2
    target = win_psd[0] / 2

    idx = np.where(win_psd <= target)[0][0]
    hpbw = rfftfreq(nfft, 1 / fs)[idx] * 2
    return hpbw


# print(get_hpbw('flattop', 2**13, 44100))


def get_tau_from_bw(hpbw, win_type, fs, nfft=2**25, verbose=False):
    # Get the tau that leads to a window with hpbw closest to the target

    # Exponential search for an upper bound
    lo = 2
    hi = 8
    if verbose:
        print(f"Initializing exponential search for upper bound;")
        print(f"Lower bound is tau={lo}")
        print(f"Testing {hi}:")
    while get_hpbw(win_type, tau=hi, fs=fs, nfft=nfft) > hpbw:
        lo = hi
        hi *= 2
        if verbose:
            print(f"Too small!")
            print(f"Testing {hi}:")
    if verbose:
        print(f"Found upper bound: {hi}")
        print(f"Initializing binary search")
    # Binary search between lo and hi until they are neighbors
    while hi - lo > 1:
        mid = (lo + hi + 1) // 2
        if verbose:
            print(f"[{lo}, {hi}] --- testing {mid}")
        mid_hpwb = get_hpbw(win_type, tau=mid, fs=fs, nfft=nfft)
        if mid_hpwb == hpbw:
            return mid_hpwb
        elif mid_hpwb > hpbw:
            lo = mid
        else:
            hi = mid
    if verbose:
        print(f"Now we're down to [{lo}, {hi}]")
    lo_hpbw = get_hpbw(win_type, tau=lo, fs=fs, nfft=nfft)
    hi_hpbw = get_hpbw(win_type, tau=hi, fs=fs, nfft=nfft)
    # Check which is closer
    if np.abs(hi_hpbw - hpbw) < np.abs(hpbw - lo_hpbw):
        tau = hi
        hpbw = hi_hpbw
    else:
        tau = lo
        hpbw = lo_hpbw
    if verbose:
        print(f"Final answer: {tau} for HPBW={hpbw:.5g}")
    return tau, hpbw


def get_colors(peak_qual):
    match peak_qual:
        case "good":
            return [
                "#1f77b4",
                "#ff7f0e",
                "#e377c2",
                "#9467bd",
            ]
        case "bad":
            return [
                "#d62728",
                "#8c564b",
                "#7f7f7f",
                "#bcbd22",
            ]


def fit_lorentzian(f, psd):
    """
    Fit a single Lorentzian to a PSD peak.

    Parameters
    ----------
    f : ndarray
        Frequency array.
    psd : ndarray
        PSD array corresponding to f.

    Returns
    -------
    popt : ndarray
        Optimal parameters [x0, gamma, A].
    lorentz_fit : ndarray
        Lorentzian evaluated at f with fitted parameters.
    """

    # --- Lorentzian model ---
    def lorentzian(x, x0, gamma, A):
        return A / (1 + ((x - x0) / gamma) ** 2)

    # Normalize for nicer dynamic range
    norm_factor = np.max(psd)
    psd_norm = psd / norm_factor

    # --- Initial guesses ---
    peak_idx = np.argmax(psd_norm)
    x0_guess = f[peak_idx]
    A_guess = psd_norm[peak_idx]
    # y0_guess = np.min(psd_norm)

    # Rough HWHM estimate: find freq span where PSD > half max
    half_max = A_guess / 2
    indices_half = np.where(psd_norm > half_max)[0]
    if len(indices_half) > 1:
        hwhm_guess = (f[indices_half[-1]] - f[indices_half[0]]) / 2
    else:
        hwhm_guess = (f[-1] - f[0]) / 2  # fallback guess
    # p0 = [x0_guess, y0_guess, hwhm_guess, A_guess]
    p0 = [x0_guess, hwhm_guess, A_guess]

    # --- Bounds ---
    x0_bounds = (f[0], f[-1])
    # y0_bounds = (0, np.inf)
    hwhm_bounds = (0, f[-1] - f[0])
    A_bounds = (A_guess * 0.5, A_guess * 2)  #

    # bounds = (
    #     [x0_bounds[0], y0_bounds[0], hwhm_bounds[0], A_bounds[0]],
    #     [x0_bounds[1], y0_bounds[1], hwhm_bounds[1], A_bounds[1]],
    # )
    bounds = (
        [x0_bounds[0], hwhm_bounds[0], A_bounds[0]],
        [x0_bounds[1], hwhm_bounds[1], A_bounds[1]],
    )

    # --- Fit ---
    try:
        popt, pcov = curve_fit(lorentzian, f, psd_norm, p0=p0, bounds=bounds)
    except RuntimeError:
        print("Lorentzian fit did not converge, returning initial guess.")
        popt = p0
    # x0, y0, gamma, A = popt
    # A, y0 = np.array([A, y0]) * norm_factor
    # lorentz_fit = lorentzian(f, x0, y0, gamma, A)

    # return x0, y0, gamma, A, lorentz_fit
    x0, gamma, A = popt
    A = A * norm_factor
    lorentz_fit = lorentzian(f, x0, gamma, A)

    return x0, gamma, A, lorentz_fit


def get_hop_from_hop_thing(hop_thing, tau, fs):
    match hop_thing[0]:
        case "tau":
            hop = int(round(tau * hop_thing[1]))
        case "s":
            hop = int(round(fs * hop_thing[1]))
        case "int":
            hop = hop_thing[1]
            if not isinstance(hop, int):
                raise ValueError("You passed in hop as an 'int' but it's not an int...")
    return hop


def get_human_peak_freqs(wf_fn):
    match wf_fn:
        case "human_TH14RearwaveformSOAEshort":
            mag_freqs = [0.6, 0.86, 0.93, 1.26, 1.62, 2.26, 2.83, 4.38]
            C_freqs = [
                0.603,
                0.861,
                0.933,
                1.119,
                1.191,
                1.263,
                1.350,
                1.536,
                1.623,
                2.255,
                2.829,
                3.087,
                3.446,
                4.373,
                4.524,
                4.723,
                6.060,
                6.318,
            ]
        case "human_RRrearSOAEwf1short":
            mag_freqs = [2.14, 3.26, 3.73, 4.41, 5.56, 6.47, 7.66, 11.72]
            C_freqs = [
                2.139,
                3.262,
                3.736,
                4.165,
                4.412,
                5.561,
                6.474,
                7.666,
                11.715,
            ]
        case "human_TH13RearwaveformSOAEshort":
            mag_freqs = [0.68, 0.91, 1.44, 1.52, 1.67, 2.04, 2.28, 2.70, 6.04]
            C_freqs = [
                0.373,
                0.647,
                0.847,
                0.904,
                0.977,
                1.092,
                1.276,
                1.337,
                1.435,
                1.523,
                1.595,
                1.674,
                1.811,
                1.925,
                2.039,
                2.154,
                2.284,
                2.701,
                2.844,
                3.633,
                6.044,
                7.293,
            ]
        case "human_KClearSOAEwf2":
            mag_freqs = [1.22, 1.5, 1.88, 3.79]
            C_freqs = [
                0.704,
                0.880,
                1.106,
                1.220,
                1.351,
                1.494,
                1.881,
                3.040,
                3.159,
                3.303,
                3.791,
            ]
        case "human_AP7RearwaveformSOAEshort":
            mag_freqs = [
                0.58,
                1.18,
                1.37,
                1.78,
                2.04,
                2.40,
                2.54,
                2.7,
                2.93,
                3.72,
                3.92,
            ]
            C_freqs = [
                0.373,
                0.586,
                0.690,
                0.790,
                0.861,
                1.178,
                1.265,
                1.364,
                1.794,
                2.039,
                2.168,
                2.291,
                2.397,
                2.542,
                2.703,
                2.928,
                3.116,
                3.718,
                3.918,
            ]
        case "human_coNW_fgF090728R":
            mag_freqs = [1.578, 3.65, 3.86, 4.18, 4.7, 7.05, 7.34, 9.07]
            C_freqs = [
                0.569,
                0.618,
                0.734,
                0.792,
                0.921,
                0.992,
                1.076,
                1.135,
                1.323,
                1.510,
                1.580,
                1.666,
                3.649,
                3.862,
                4.020,
                4.179,
                4.695,
                5.657,
                6.761,
                7.047,
                7.337,
                7.625,
                9.074,
            ]
        case "human_TH21RearwaveformSOAE":
            mag_freqs = [
                1.41,
                1.72,
                1.88,
                2.01,
                2.14,
                2.29,
                2.46,
                2.61,
                2.76,
                3.07,
                3.27,
                4.14,
            ]
            C_freqs = [
                1.408,
                1.507,
                1.624,
                1.724,
                1.881,
                2.011,
                2.140,
                2.297,
                2.466,
                2.610,
                2.759,
                2.917,
                3.075,
                3.274,
                4.135,
            ]
        case "human_AVGrearSOAEwf2":
            mag_freqs = [1.71, 1.85, 2.24, 2.43, 2.46, 2.64, 2.98, 3.54, 4.35, 6.58]
            C_freqs = [
                0.688,
                1.263,
                1.365,
                1.566,
                1.710,
                1.853,
                1.997,
                2.112,
                2.240,
                2.414,
                2.455,
                2.642,
                2.980,
                3.542,
                4.350,
                6.576,
            ]
        case "human_FMlearSOAEwfA01":
            mag_freqs = [1.55, 1.72, 2.02, 2.24, 2.76, 3.04, 3.17, 4.01]
            C_freqs = [
                0.308,
                0.575,
                0.653,
                0.702,
                0.862,
                0.992,
                1.048,
                1.307,
                1.379,
                1.481,
                1.557,
                1.623,
                1.721,
                1.794,
                1.897,
                2.023,
                2.239,
                2.757,
                2.887,
                3.042,
                3.173,
                4.006,
                4.179,
            ]
        case "human_JBrearSOAEwf2short":
            mag_freqs = [1.23, 1.435, 1.7, 1.97, 3.56, 4.06, 7.28]
            C_freqs = [
                0.732,
                0.790,
                0.849,
                0.904,
                1.234,
                1.436,
                1.705,
                1.969,
                3.560,
                4.063,
                4.983,
                7.281,
            ]
        case "human_LSrearSOAEwf1short":
            mag_freqs = [0.73, 0.99, 1.64, 2.22, 3.12]
            C_freqs = [
                0.288,
                0.734,
                0.993,
                1.208,
                1.294,
                1.639,
                2.111,
                2.226,
                3.115,
                5.011,
            ]
        case "human_JIrearSOAEwf2short":
            mag_freqs = [
                1.29,
                1.72,
                1.84,
                2.34,
                2.81,
                3.40,
                4.05,
                5.12,
                5.84,
                7.94,
                8.31,
                8.68,
            ]
            C_freqs = [
                0.171,
                0.349,
                0.516,
                1.294,
                1.581,
                1.724,
                1.840,
                2.343,
                2.815,
                3.074,
                3.405,
                4.048,
                5.123,
                5.843,
                7.940,
                8.314,
                8.685,
            ]

    return np.array(mag_freqs), np.array(C_freqs)


# # Chris' list before I removed some
# def get_human_peak_freqs(wf_fn):
#     match wf_fn:
#         case "human_TH14RearwaveformSOAEshort":
#             mag_freqs= [0.6,0.86,0.93,1.26,1.62,2.26,2.83,4.38]
#             C_freqs= [0.433,0.603,0.706,0.861,0.933,1.119,1.191,1.263,1.350,
#                         1.536,1.623,2.255,2.317,2.699,2.829,3.087,3.446,3.862,4.373,4.524,
#                         4.723,6.060,6.318]
#         case "human_RRrearSOAEwf1short":
#             mag_freqs= [2.14,3.25,3.73,4.17,4.41,5.56,6.47,7.66,11.72]
#             C_freqs= [0.691,1.496,1.680,1.824,
#                 2.139,2.468,2.943,3.262,3.461,3.736,4.165,4.412,5.561,
#                 6.466,7.666,7.839,7.902,8.041,8.372,8.498,
#                 9.850,10.294,10.351,10.537,11.715]
#         case "human_TH13RearwaveformSOAEshort":
#             mag_freqs = [0.68,0.91,1.52,1.67,2.04,2.28,2.70,6.04]
#             C_freqs =[0.373,0.647,0.702,0.847,0.904,0.977,1.092,1.276,1.337,1.435,
#       1.523,1.595,1.674,1.811,1.925,2.039,2.154,2.284,2.585,2.701,
#       2.844,3.633,6.044,7.293]
#         case "human_KClearSOAEwf2":
#             mag_freqs = [1.23,1.5,1.89,3.8]
#             C_freqs = [0.661,0.704,0.880,1.106,1.161,1.220,1.351,1.494,1.881,
#       1.939,3.040,3.159,3.303,3.791]
#         case "human_AP7RearwaveformSOAEshort":
#             mag_freqs = [0.58,1.18,1.27,1.37,2.04,2.54,2.7,3.72,3.92]
#             C_freqs = [0.366,0.586,0.690,0.790,0.861,1.063,1.178,1.265,1.373,1.794,2.039,
#       2.168,2.291,2.397,2.542,2.703,2.928,3.116,3.718,3.918]
#         case "human_coNW_fgF090728R":
#             mag_freqs = [3.65,3.87,4.7,7.06,7.34,9.08]
#             C_freqs = [0.569,0.618,0.663,0.734,0.792,0.921,0.992,1.069,
#       1.135,1.323,1.441,1.510,1.580,1.666,1.773,3.649,
#       3.862,4.020,4.179,4.493,4.695,5.657,7.047,7.337,
#       7.625,8.614,9.074]
#         case "human_TH21RearwaveformSOAE":
#             mag_freqs = [1.41,1.62,1.72,1.88,2.01,2.14,2.29,2.46,2.61,2.76,3.07,3.27,4.14]
#             C_freqs =[1.408,1.494,1.624,1.724,1.881,2.011,2.140,2.297,2.466,2.610,
#       2.759,2.917,3.075,3.274,4.135]
#         case "human_AVGrearSOAEwf2":
#             mag_freqs = [1.71,1.86,2.24,2.43,2.46,2.64,2.98,3.54,4.35,6.58]
#             C_freqs = [0.675,0.778,1.135,1.252,1.365,1.566,1.710,1.853,1.997,
#       2.112,2.240,2.414,2.470,2.642,2.743,2.980,3.347,3.542,
#       4.350,4.452,6.576]
#         case "human_FMlearSOAEwfA01":
#             mag_freqs = [1.56,1.72,2.02,2.24,2.76,3.04,3.17,4.01]
#             C_freqs = [0.308,0.349,0.575,0.653,0.702,0.862,0.905,0.992,1.048,
#       1.307,1.379,1.481,1.557,1.623,1.721,1.794,1.897,2.023,
#       2.239,2.757,2.887,3.042,3.173,4.006,4.179]
#         case "human_JBrearSOAEwf2short":
#             mag_freqs = [1.23,1.7,1.97,3.56,4.07,7.28]
#             C_freqs = [0.604,0.732,0.790,0.849,0.891,1.136,1.234,1.436,
#        1.705,1.969,2.357,3.560,3.869,4.063,
#        4.148,4.194,4.279,4.379,4.596,
#        4.983,5.031,6.979,
#        7.281]
#         case "human_LSrearSOAEwf1short":
#             mag_freqs = [0.73,0.99,1.21,1.29,1.64,2,23,3.12]
#             C_freqs = [0.288,0.650,0.734,0.993,1.208,1.294,1.639,1.766,
#        2.111,2.226,3.115,4.335,5.011]
#         case "human_JIrearSOAEwf2short":
#             mag_freqs = [1.29,1.58,1.72,1.84,2.34,2.81,3.40,4.05,5.12,5.84,7.93,8.31,8.68]
#             C_freqs = [0.171,0.349,0.516,0.704,1.294,1.581,1.724,1.840,2.223,
#        2.343,2.815,3.074,3.405,4.048,5.123,5.843,6.116,6.963,
#        7.940,8.314,8.685]

#     return np.array(mag_freqs), np.array(C_freqs)
