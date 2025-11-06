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
    pw,
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
    if hop < 1:
        hop = int(round(hop * tau))
    pkl_fn_id = rf"{species} {wf_idx}, PW={pw}, {win_meth_str}, hop={hop}, tau={tau}, {filter_str}, xi_max={xi_max_s*1000:.0f}ms, {nbacf_str}{delta_xi_str}{nfft_str}{f0s_str}{const_N_pd_str}{N_bs_str}{demean_str}wf_len={wf_len_s}s, wf={wf_fn.split('.')[0]}"
    pkl_fn = f"{pkl_fn_id} (Colossogram).pkl"

    # Convert to samples
    xi_min = round(xi_min_s * fs)
    xi_max = round(xi_max_s * fs)

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
            wf = filter_wf(wf, fs, filter_meth, species)

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
            pw=pw,
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
    cgram_dict.update({"pw": pw})

    # Check if we need to correct for old squared definition of PW
    if pw and "unsquared_pw" not in cgram_dict.keys():
        print("Here's an older dict that squared the powerweights, so sqrting now!")
        unsquared_colossogram = np.sqrt(cgram_dict["colossogram"])
        cgram_dict["colossogram"] = unsquared_colossogram

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
            ]  # Note 8675 is only bad in PW=False, it's good in PW=True

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


def filter_wf(wf, fs, filter_meth, species, subtract_mean=True):
    if subtract_mean:
        wf = wf - np.mean(wf)
    if filter_meth is not None and species != "V Sim Human":
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
