import numpy as np
from helper_funcs import *
import os
from phaseco import *
import matplotlib.pyplot as plt
import pandas as pd
import phaseco as pc

"PARAMETERS"
# Loop parameters
all_species = [
    "Anole",
    "Owl",
    "Tokay",
    "Human",
]

speciess = all_species
wf_idxs = range(4)


# WF pre-processing parameters
filter_meths = [
    # {
    #     "type": "kaiser",
    #     "cf": 300,
    #     "df": 50,
    #     "rip": 100,
    # },
    None,
]
# cf=cutoff freq, df=transition band width, rip=max allowed ripple (in dB)

wf_len_s = 60  # Will crop waveform to this length (in seconds)
scale = True  # Scale the waveform for dB SPL (shouldn't have an effect outisde of vertical shift on PSD;
# only actually scales if we know the right scaling constant, which is only Anoles and Humans)
demean = True  # subtract mean

# Coherence Parameters
pws = [False]
rho_bw_hops = [
    (1.0, 50, ("s", 0.01)),
    (None, "species", ("s", 0.01)),
]
wa = False
const_N_pd = 0
nfft = 2**14
nbacf = False


# PSD Parameters
tau_psd = nfft
hop_psd = 0.5 # Half of tau
win_type_psd = "hann"


# Options for iterating through subjects
force_recalc_colossogram = 0
plot_what_we_got = 0
only_calc_new_coherences = 0


# Fitting Parameters
noise_freqs = []
N_bs = 0  # Num of bootstraps to do to calculate fit func CI (0 = standard, no bootstrapping)
mse_thresh = np.inf
A_max = np.inf  # 1 or np.inf
A_const = False  # Fixes the amplitude of the decay at 1
start_fit = "frac"
start_fit_frac = 0.9
stop_fit = "frac"  # stops fit when it reaches a fraction of the fit start value
stop_fit_frac = 0.1  # aforementioned fraction
sigma_weighting_power = 0  # < 0 -> less weight on lower coherence part of fit
fit_func = "exp"  # 'exp' or 'gauss'

# Output parameters
output_peak_picks = 1
output_fits = 1
output_bad_fits = 1
output_filtered_peaks_general = 1
output_spreadsheet = (
    ((wf_idxs == range(4)) and (speciess == all_species))
    and output_fits
    and not plot_what_we_got
)
output_colossograms = 1
show_plots = 0
force_all_freqs = 0


# Plotting parameters
plot_noise_floor = 1

# Species specific params

# Maximum xi value
xi_max_ss = {
    "Anole": 0.1,
    "Owl": 0.1,
    "Human": 1.0,
    "V Sim Human": 0.2,
    "Tokay": 0.1,
}

xi_min_ss = {
    "Anole": 0.001,
    "Owl": 0.001,
    "Human": 0.001,
    "Tokay": 0.001,
}

# Maximum frequency to plot (in khz)
max_khzs = {
    "Anole": 6,
    "Tokay": 6,
    "Human": 10,
    "V Sim Human": 10,
    "Owl": 12,
}

species_bws = {"Anole": 150, "Human": 50, "Owl": 300, "Tokay": 150}

# Define a folder
pkl_folder = "pickles"
soae_pkl_folder = os.path.join(pkl_folder, "soae")

"Loops"
for filter_meth in filter_meths:
    for pw in pws:
        for rho, bw_type, hop_thing in rho_bw_hops:
            # Handle windowing method
            if rho is not None:
                win_meth = {"method": "rho", "win_type": "flattop", "rho": rho}
            else:
                win_meth = {"method": "static", "win_type": "flattop"}
            # Check if we can do filtered peaks
            if output_filtered_peaks_general:
                if win_meth["method"] == "static":
                    output_filtered_peaks = 1
                else:
                    print("Can't do filtered peaks for dynamic widnowing!")
                    output_filtered_peaks = 0
                    
            for p in [0]:
                # Initialize list for row dicts for xlsx file
                rows = []
                for species in speciess:
                    for wf_idx in wf_idxs:
                        wf_pp = None
                        xi_min_s = xi_min_ss[species]
                        if bw_type == 'species':
                            xi_min_s = 0.0005
                        # if species != 'Human' and win_meth['method'] == 'static':
                        #     xi_max_s = 0.05

                        if bw_type == "species":
                            bw = species_bws[species]
                        else:
                            bw = bw_type
                        
                        

                        "Get waveform"
                        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(
                            species=species,
                            wf_idx=wf_idx,
                        )
                        all_sel_freqs = np.concat(
                            (good_peak_freqs, bad_peak_freqs, noise_freqs)
                        )

                        # Get precalculated tau for this bandwidth
                        tau = get_precalc_tau_from_bw(
                            bw, fs, win_meth["win_type"], pkl_folder
                        )
                        print(f"{fs}Hz -- tau={tau}")

                        # Check we haven't exceeded our max set by nfft
                        if tau > nfft:
                            raise ValueError(f"Can't have tau = {tau} > {nfft} = nfft!")

                        # Process species-specific params
                        decay_start_limit_xi_s = 0.05 if species == "Human" else 0.015
                        max_khz = max_khzs[species]
                        xi_max_s = xi_max_ss[species]

                        # Get hop
                        hop = get_hop_from_hop_thing(hop_thing, tau, fs)

                        # These ones didn't end by 1.0, so will take em out to 1.5 instead
                        if species == "Human" and wf_idx in [2, 3]:
                            if xi_max_s == 1.0:
                                xi_max_s = 1.5

                        "Calculate/load things"

                        # Deal with # of freqs to calculate
                        if output_colossograms or force_all_freqs:
                            # hop_cgram = hop
                            f0s_cgram = None
                        else:
                            # hop_cgram = 1  # So we can implement via NBACF
                            f0s_cgram = all_sel_freqs
                        hop_cgram = hop

                        # This will either load it if it's there or calculate it (and pickle it) if not
                        cgram_dict = load_calc_colossogram(
                            **{
                                "wf": wf,
                                "wf_idx": wf_idx,
                                "wf_fn": wf_fn,
                                "wf_len_s": wf_len_s,
                                "wf_pp": wf_pp,
                                "species": species,
                                "fs": fs,
                                "filter_meth": filter_meth,
                                "pkl_folder": soae_pkl_folder,
                                "pw": pw,
                                "tau": tau,
                                "nfft": nfft,
                                "xi_min_s": xi_min_s,
                                "xi_max_s": xi_max_s,
                                "hop": hop_cgram,
                                "win_meth": win_meth,
                                "force_recalc_colossogram": force_recalc_colossogram,
                                "plot_what_we_got": plot_what_we_got,
                                "only_calc_new_coherences": only_calc_new_coherences,
                                "demean": demean,
                                "const_N_pd": const_N_pd,
                                "scale": scale,
                                "N_bs": N_bs,
                                "f0s": f0s_cgram,
                                "nbacf":nbacf,
                            }
                        )
                        if "plot_what_we_got" in cgram_dict.keys():
                            print("NO PICKLE, SKIPPING")
                            continue
                        if "only_calc_new_coherences" in cgram_dict.keys():
                            print("ALREADY CALCULATED, SKIPPING")
                            continue

                        # Load everything that wasn't explicitly "saved" in the filename
                        colossogram = cgram_dict["colossogram"]
                        wf_pp = cgram_dict[
                            "wf_pp"
                        ]  # this is now no longer None and we can use it next time (unless it gets reset because we're not doing anything else with this subject)
                        win_meth_str = cgram_dict["win_meth_str"]
                        f = cgram_dict[
                            "f"
                        ]  # This will be just f0s if they were passed in
                        xis_s = cgram_dict["xis_s"]
                        N_pd_min = cgram_dict["N_pd_min"]
                        N_pd_max = cgram_dict["N_pd_max"]
                        hop = cgram_dict["hop"]

                        # Handle transpose from old way
                        if colossogram.shape[0] != xis_s.shape[0]:
                            colossogram = colossogram.T

                        # Get peak bin indices now that we have f
                        # (this is either the full nfft freq axis or just the short list, works either way)
                        good_peak_idxs = np.argmin(
                            np.abs(f[:, None] - good_peak_freqs[None, :]), axis=0
                        )
                        bad_peak_idxs = np.argmin(
                            np.abs(f[:, None] - bad_peak_freqs[None, :]), axis=0
                        )

                        all_sel_freq_idxs = np.argmin(
                            np.abs(f[:, None] - all_sel_freqs[None, :]), axis=0
                        )  # Trivial in the case that f = all_sel_freqs

                        "Make more directories"
                        bw_str = f"BW=Species" if bw_type == "species" else f"BW={bw}Hz"
                        relevant_comp_str = rf"PW={pw}, {bw_str}, {win_meth_str}"
                        results_folder = os.path.join(
                            "results","soae",
                            rf"SOAE Results ({relevant_comp_str})",
                        )
                        os.makedirs(results_folder, exist_ok=True)

                        "Plots"
                        # Build plot-related strings
                        N_bs_str = "" if N_bs == 0 else f"N_bs={N_bs}, "
                        pw_str = f"{pw}" if not wa or not pw else "WA"
                        const_N_pd_str = "" if const_N_pd else "N_pd=max, "
                        f0s_str = (
                            ""
                            if f0s_cgram is None
                            else f"f0s={np.array2string(f0s_cgram, formatter={'float' : lambda x: "%.0f" % x})}, "
                        )
                        nfft_str = "" if nfft is None else f"nfft={nfft}, "
                        delta_xi_str = (
                            ""
                            if xi_min_s == 0.001
                            else f"delta_xi={xi_min_s*1e3:.0f}ms, "
                        )
                        bw_str = (
                            f"HPBW={bw}Hz"
                            if bw is not None
                            else f"tau={1e3 * tau / fs:.0f}ms"
                        )
                        N_pd_str = pc.get_N_pd_str(const_N_pd, N_pd_min, N_pd_max)
                        filter_str = get_filter_str(filter_meth)
                        hop_str = (
                            f"Hop={(hop / fs)*1e3:.1f}ms"
                            if bw_type != "species"
                            else f"Hop=1"
                        )
                        # Build IDs
                        plot_fn_id = rf"{species} {wf_idx}, {bw_str}, hop={hop:.0f}, PW={pw_str}, {win_meth_str}, {filter_str}, xi_max={xi_max_s*1e3:.0f}ms, {delta_xi_str}{nfft_str}{f0s_str}{const_N_pd_str}{N_bs_str}wf_len={wf_len_s}s, wf={wf_fn.split('.')[0]}"
                        method_id = rf"[$\tau$={(tau/fs)*1e3:.2f}ms]   [PW={pw}]   [{win_meth_str}]   [{hop_str}]   [{N_pd_str}]   [nfft={nfft}]"
                        suptitle = rf"[{species} {wf_idx}]   [{wf_fn}]   [HPBW={bw}Hz]   {method_id}   [{filter_str}]"
                        f_khz = f / 1e3
                        if output_colossograms:
                            print("Plotting Colossogram")
                            plt.close("all")
                            plt.figure(figsize=(15, 5))
                            pc.plot_colossogram(
                                xis_s,
                                f,
                                colossogram,
                                pw=pw,
                                cmap="magma",
                            )
                            plt.title(rf"Colossogram")
                            plt.ylim(0, max_khz)
                            for peak_idx in good_peak_idxs:
                                plt.scatter(
                                    xi_min_s * 1e3 + (xi_max_s * 1e3) / 50,
                                    f_khz[peak_idx],
                                    c="w",
                                    marker=">",
                                    label="Peak at " + f"{f[peak_idx]:0f}Hz",
                                    alpha=0.5,
                                )
                            plt.title(f"Colossogram", fontsize=18)
                            plt.suptitle(suptitle, fontsize=10)
                            for folder in [results_folder]:
                                colossograms_folder = os.path.join(
                                    folder, "Colossograms"
                                )
                                os.makedirs(colossograms_folder, exist_ok=True)
                                cgram_fp = os.path.join(
                                    colossograms_folder,
                                    rf"{plot_fn_id} (Colossogram).jpg",
                                )
                                plt.savefig(cgram_fp, dpi=300)
                            if show_plots:
                                plt.show()
                        # See if we need to preprocess waveform
                        if wf_pp is None and (output_peak_picks or output_filtered_peaks):
                            wf = crop_wf(wf, fs, wf_len_s)
                            wf = filter_wf(wf, fs, filter_meth, species)
                            if species in ["Anole", "Human"] and scale:  # Scale wf
                                wf = scale_wf(wf, species)
                            if demean:
                                wf -= np.mean(wf)
                        else:
                            wf = wf_pp
                            
                        if output_peak_picks:
                            print("Plotting Peak Picks")

                            # Calculate arrays for plotting
                            target_xi_s = 0.01
                            xi_idx = np.argmin(np.abs(xis_s - target_xi_s))
                            coherence_slice = colossogram[xi_idx, :]
                            f_psd, psd = pc.get_welch(
                                wf=wf, fs=fs, tau=tau_psd, hop=hop_psd, win=win_type_psd
                            )
                            psd_db = 10 * np.log10(psd)
                            f_psd_khz = f_psd / 1e3
                            good_peak_idxs_psd = np.argmin(
                                np.abs(good_peak_freqs[:, None] - f_psd[None, :]),
                                axis=1,
                            )
                            bad_peak_idxs_psd = np.argmin(
                                np.abs(bad_peak_freqs[:, None] - f_psd[None, :]),
                                axis=1,
                            )

                            # Initialize plot
                            plt.close("all")
                            plt.figure(figsize=(12, 8))
                            plt.suptitle(suptitle)

                            # Get colors
                            good_colors = get_colors("good")
                            bad_colors = get_colors("bad")

                            # Coherence slice plot
                            if f0s_cgram is None:
                                plt.subplot(2, 1, 1)
                                plt.title(
                                    rf"Colossogram Slice at $\xi={xis_s[xi_idx]:.3f}$"
                                )
                                plt.plot(
                                    f_khz,
                                    coherence_slice,
                                    label=r"$C_{\xi}$, $\xi={target_xi}$",
                                    color="k",
                                )
                                for peak_idxs, colors in zip(
                                    [good_peak_idxs, bad_peak_idxs],
                                    [good_colors, bad_colors],
                                ):
                                    for peak_idx, color in zip(peak_idxs, colors):
                                        plt.scatter(
                                            f_khz[peak_idx],
                                            coherence_slice[peak_idx],
                                            c=color,
                                        )
                                plt.xlabel("Frequency (kHz)")
                                plt.ylabel(r"$C_{\xi}$")
                                plt.xlim(0, max_khz)
                                plt.ylim(0, 1)
                                plt.subplot(2, 1, 2)

                            # PSD plot
                            plt.title(rf"Power Spectral Density")
                            plt.plot(f_psd_khz, psd_db, label="PSD", color="k")
                            bw_khz = bw / 1e3
                            for peak_idxs_psd, colors in zip(
                                [good_peak_idxs_psd, bad_peak_idxs_psd],
                                [good_colors, bad_colors],
                            ):
                                for peak_idx_psd, color in zip(peak_idxs_psd, colors):
                                    plt.scatter(
                                        f_psd_khz[peak_idx_psd],
                                        psd_db[peak_idx_psd],
                                        c=color,
                                    )
                                    f0_khz = f_psd_khz[peak_idx_psd]
                                    # Color according to bandwidth
                                    a, b = f0_khz - bw_khz / 2, f0_khz + bw_khz / 2

                                    # Interpolate so you get exact endpoints
                                    ya = np.interp(a, f_psd_khz, psd_db)
                                    yb = np.interp(b, f_psd_khz, psd_db)

                                    # Restrict x,y to the band plus the interpolated endpoints
                                    mask = (f_psd_khz >= a) & (f_psd_khz <= b)
                                    f_band = np.concatenate(([a], f_psd_khz[mask], [b]))
                                    psd_band = np.concatenate(
                                        ([ya], psd_db[mask], [yb])
                                    )
                                    plt.plot(
                                        f_band,
                                        psd_band,
                                        c=color,
                                    )

                            plt.xlabel("Frequency (kHz)")
                            plt.ylabel("PSD [dB]")
                            plt.legend()
                            plt.xlim(0, max_khz)
                            plt.tight_layout()
                            for folder in [results_folder]:
                                pp_folder = os.path.join(results_folder, "Peak Picks")
                                os.makedirs(pp_folder, exist_ok=True)
                                pp_fp = os.path.join(
                                    pp_folder, rf"{plot_fn_id} (Peak Picks).jpg"
                                )
                                plt.savefig(pp_fp, dpi=300)
                            if show_plots:
                                plt.show()

                        "FITTING"
                        if output_fits:
                            print(rf"Fitting {wf_fn}")

                            p0 = [1, 0.5]
                            bounds = ([0, 0], [np.inf, A_max])  # [T, amp]
                            for zoom_to_fit in [True, False]:
                                for peak_freqs, peak_idxs, colors, good_peaks in zip(
                                    [good_peak_freqs, bad_peak_freqs],
                                    [good_peak_idxs, bad_peak_idxs],
                                    [good_colors, bad_colors],
                                    [True, False],
                                ):
                                    # If there are no peaks, continue
                                    if len(peak_idxs) == 0:
                                        if good_peaks:
                                            print("WARNING: No good peaks were picked!")
                                        # Could also just be that there are no bad peaks! either way...
                                        continue
                                    if not good_peaks and not output_bad_fits:
                                        continue

                                    if not good_peaks:
                                        print("Now fitting bad peaks...")

                                    plt.close("all")
                                    decay_fig = plt.figure(figsize=(15, 10))
                                    swp_str = (
                                        ""
                                        if sigma_weighting_power == 0
                                        else f"   [SWP={sigma_weighting_power}]"
                                    )
                                    plt.suptitle(f"{suptitle}{swp_str}")

                                    # If we're gonna add filtered peaks, initialize that figure
                                    if output_filtered_peaks and not zoom_to_fit:
                                        filtered_peak_fig = plt.figure(figsize=(15, 10))
                                        plt.suptitle(
                                            rf"{species} {wf_idx}    [$BW_{{\text{{{species}}}}}$={bw}]   [$\tau_\text{{PSD}}$={tau_psd}, {win_type_psd.capitalize()}]   [H={hop_psd}$\tau$]"
                                        )
                                        # Then switch back
                                        plt.figure(decay_fig)


                                    for f0, peak_idx, color, subplot_idx in zip(
                                        peak_freqs, peak_idxs, colors, [1, 2, 3, 4]
                                    ):

                                        # Fit peak
                                        N_xi, N_xi_dict = pc.get_N_xi(
                                            cgram_dict,
                                            f0,
                                            decay_start_limit_xi_s=decay_start_limit_xi_s,
                                            mse_thresh=mse_thresh,
                                            stop_fit=stop_fit,
                                            stop_fit_frac=stop_fit_frac,
                                            start_fit_frac=start_fit_frac,
                                            sigma_power=sigma_weighting_power,
                                            A_max=A_max,
                                            A_const=A_const,
                                            fit_func=fit_func,
                                        )

                                        # Unpack dictionary
                                        f0_exact_bin = N_xi_dict["f0_exact"]
                                        N_xi = N_xi_dict["N_xi"]
                                        N_xi_std = N_xi_dict["N_xi_std"]
                                        T_xi = N_xi_dict["T_xi"]
                                        T_xi_std = N_xi_dict["T_xi_std"]
                                        A_xi = N_xi_dict["A_xi"]
                                        A_xi_std = N_xi_dict["A_xi_std"]
                                        mse = N_xi_dict["mse"]
                                        decayed_idx = N_xi_dict["decayed_idx"]

                                        # Plot the fit
                                        plt.subplot(2, 2, subplot_idx)

                                        pc.plot_N_xi_fit(
                                            N_xi_dict,
                                            color,
                                            plot_noise_floor=plot_noise_floor,
                                            zoom_to_fit=zoom_to_fit,
                                        )

                                        # Fit peak in PSD domain
                                        if output_filtered_peaks and not zoom_to_fit:
                                            print(f"Fitting filtered peak [{f0_exact_bin:.0f}Hz]")
                                            plt.figure(filtered_peak_fig)
                                            plt.subplot(2, 2, subplot_idx)
                                            # Filter wf for a filtered wf equivalent to STFT bin with H=1
                                            win_type = "flattop"
                                            win = get_window(win_type, tau)
                                            omega_0_norm = f0_exact_bin * 2 * np.pi / fs
                                            n = np.arange(len(win))
                                            kernel = win * np.exp(1j * omega_0_norm * n)
                                            wf_filtered = convolve(wf, kernel, mode="valid", method="fft") / np.sum(
                                                win
                                            )
                                            f_psd, psd_filt = get_welch(
                                                wf_filtered,
                                                fs,
                                                tau=tau_psd,
                                                hop=hop_psd,
                                                win=win_type_psd,
                                                realfft=False,
                                            )
                                            psd = get_welch(
                                                wf, fs, tau=tau_psd, hop=hop_psd, win=win_type_psd, realfft=False
                                            )[1]

                                            xmin, xmax = f0_exact_bin - bw * 2, f0_exact_bin + bw * 2
                                            xmin_idx, xmax_idx = np.argmin(np.abs(f_psd - xmin)), np.argmin(
                                                np.abs(f_psd - xmax)
                                            )
                                            f_psd_crop, psd_crop, psd_filt_crop = (
                                                f_psd[xmin_idx:xmax_idx],
                                                psd[xmin_idx:xmax_idx],
                                                psd_filt[xmin_idx:xmax_idx],
                                            )
                                            plt.plot(f_psd_crop, psd_filt_crop, label="Filtered", color=color)
                                            plt.plot(f_psd_crop, psd_crop, label="Unfiltered", color="k")
                                            plt.ylabel(f"PSD")
                                            plt.axvline(x=f0_exact_bin - bw / 2, color="g")
                                            plt.axvline(x=f0_exact_bin + bw / 2, color="g")
                                            plt.xlabel("Frequency [Hz]")
                                            plt.title(f"{f0_exact_bin:.0f} Hz")
                                            L_f0, L_gamma, L_amp, fitted_lorentz = fit_lorentzian(
                                                f_psd_crop, psd_filt_crop
                                            )
                                            plt.plot(
                                                f_psd_crop,
                                                fitted_lorentz,
                                                label=rf"$\gamma={L_gamma:.3g}$, $A={L_amp:.3g}$",
                                                color=color,
                                                ls="--",
                                            )
                                            L_row = {
                                                "L_gamma": L_gamma,
                                                "L_amp": L_amp,
                                                "L_f0": L_f0,
                                            }
                                        
                                            plt.legend(fontsize=6)

                                            # Switch back to decay fig
                                            plt.figure(decay_fig)

                                        # Add params to a row dict
                                        if (
                                            good_peaks
                                            and output_spreadsheet
                                            and not zoom_to_fit # this way we only do it once
                                        ):
                                            row = {
                                                "Species": species,
                                                "WF Index": wf_idx,
                                                "Filename": wf_fn,
                                                "Frequency": f0_exact_bin,
                                                "N_xi": N_xi,
                                                "N_xi_std": N_xi_std,
                                                "T_xi": T_xi,
                                                "T_xi_std": T_xi_std,
                                                "A_xi": A_xi,
                                                "A_xi_std": A_xi_std,
                                                "MSE": mse,
                                                "Decayed Xi": xis_s[decayed_idx],
                                                "Decayed Num Cycles": xis_s[decayed_idx]
                                                * f0_exact_bin,
                                            }
                                            if output_filtered_peaks and not zoom_to_fit:
                                                row.update(L_row)
                                            if N_bs > 0:
                                                row.update(
                                                    {
                                                        "Average CI Width": N_xi_dict[
                                                            "avg_delta_CI"
                                                        ]
                                                    }
                                                )
                                            rows.append(row)
                                    # Book it!
                                    plt.tight_layout()
                                    os.makedirs(results_folder, exist_ok=True)
                                    zoom_str = "[ZOOMED]" if zoom_to_fit else ""
                                    fits_str = (
                                        f"Fits [Good]" if good_peaks else f"Fits [Bad]"
                                    )
                                    for folder in [results_folder]:
                                        fits_folder = os.path.join(folder, fits_str)
                                        fit_plot_fp = os.path.join(
                                            fits_folder,
                                            rf"{plot_fn_id} ({fits_str}{zoom_str}).jpg",
                                        )
                                        os.makedirs(fits_folder, exist_ok=True)
                                        plt.savefig(fit_plot_fp, dpi=300)
                                    if show_plots:
                                        plt.show()
                                    # Ditto for the filtered peaks
                                    if output_filtered_peaks and not zoom_to_fit:
                                        plt.figure(filtered_peak_fig)
                                        filt_peaks_str = f"Filtered Peaks [{'Good' if good_peaks else 'Bad'}]"
                                        filt_peaks_folder = os.path.join(results_folder, "Filtered Peaks")
                                        filt_peaks_plot_fp = os.path.join(filt_peaks_folder, rf"{plot_fn_id} ({fits_str}).jpg")
                                        os.makedirs(filt_peaks_folder, exist_ok=True)
                                        plt.savefig(filt_peaks_plot_fp, dpi=300)
                                    if show_plots:
                                        plt.show()
                                    plt.close('all')

                if output_spreadsheet and not only_calc_new_coherences:
                    # Save parameter data as xlsx
                    df_fitted_params = pd.DataFrame(rows)
                    N_xi_fitted_parameters_fn = os.path.join(
                        results_folder, rf"SOAE N_xi Fitted Parameters ({relevant_comp_str})"
                    )
                    df_fitted_params.to_excel(
                        rf"{N_xi_fitted_parameters_fn}.xlsx", index=False
                    )

                print("Done!")
