import os
import phaseco as pc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from collections import defaultdict
from scipy.signal import correlate, get_window, convolve, correlation_lags, hilbert
from helper_funcs import *
dirs = get_dirs()
os.chdir(dirs["oto"])

# Parameters

# ---WF---
speciess = ['Human', 'Anole', 'Tokay', 'Owl']
wf_idxs = range(4)
# speciess = ['Anole']
# wf_idxs = [1]
wf_len_s = 60

# ---PSD---
tau_s_psd = 0.5
nfft_psd = None
win_type_psd = 'hann'
hop_s_psd = tau_s_psd / 2

# ---Plots---
flims = {'Anole':[1, 6], 'Human':[1, 10], 'Owl':[1, 12], 'Tokay':[1, 6]}
figsize_psd = (8, 5)
figsize_ind = (5, 5)
ind_extra_bins = 50
pe_fit = [
    pe.Stroke(linewidth=10, foreground="black", alpha=1),
    pe.Normal(),
]
alpha_fit = 0.3
lw_fit = 5

# ---ACF Fit---

# Exp fit
acf_exp_fit_min = 0.1
acf_exp_fit_max = 0.9

# eta cumulative fit
eta = 0.9
sig_thresh_eta = 0.1

# Overall
T_xi_type = "int"
# max_lag_s = 2

# ---Lorentzian Fitting---
crop_bw = 200

# ---Filtering---
kaiser_rip = 100
kaiser_df = 50
exp_order = 10
p_filt = {'type':'kaiser', 'rip':kaiser_rip, 'df':kaiser_df}
# p_filt = {'type':'exp', 'order':exp_order}

# Automatic bandwidth calculation
bw_filt_thresh = 0.1



# ID
match p_filt['type']:
    case 'kaiser':
        if p_filt["df"] < 1:
            df = f"{p_filt["df"]*100:.0f}p bw"
        else:
            df = f"{p_filt["df"]}hz"
        bpf_id = f"kaiser, df={df}, rip={p_filt['rip']}db"
    case 'exp':
        bpf_id = f"exp, order={p_filt['order']}hz"
filt_id = f"bw={bw_filt_thresh*100:.0f}p max, {bpf_id}, crop={crop_bw}hz, tau={tau_s_psd*1000:.0f}ms, hop_psd={hop_s_psd*1000:.0f}ms, nfft={nfft_psd}, win={win_type_psd}"

for max_lag_s in [0.5, 0.025, 0.1]:
    # Initialize spreadsheet rows
    rows = []
    for species in speciess:
        for wf_idx in wf_idxs:
        
            print(f"{species} {wf_idx}")
            # Get wf
            wf, wf_fn, fs = get_wf(species=species, wf_idx=wf_idx)
            fp_pp = os.path.join(dirs["results"], "picked_peaks.json")
            picked_peaks = get_picked_peaks(fp_pp, wf_fn)
            wf = crop_wf(wf, fs, wf_len_s)
            
            # Convert from fs
            tau_psd = int(round(tau_s_psd * fs))
            hop_psd = int(round(hop_s_psd * fs))

            # Get species specific
            flim = flims[species]

            # Get psd for plotting
            f, psd = pc.get_welch(wf, fs, tau_psd, hop=hop_psd, win=win_type_psd, nfft=nfft_psd)
            psd_db = 10*np.log10(psd)
            f_khz = f / 1000

            # Plot PSD
            plt.close('all')
            plt.figure("full", figsize=figsize_psd)
            plt.plot(f_khz, psd_db, label='PSD', color='k', lw=5)
            plt.xlim(flims[species])
            plt.ylabel("PSD [dB]")
            plt.xlabel("Frequency [kHz]")
            plt.suptitle(wf_fn)

            # Get Lorentzian fits
            for f0_max_saved, color in zip(picked_peaks, get_colors('good')):
                # Deal with different f0 definitions
                f0_max = f[np.argmin(np.abs(f-f0_max_saved))]
                if np.abs(f0_max-f0_max_saved) > 1e-9:
                    raise ValueError(f"Your loaded peak pick {f0_max_saved} does not match the current bin center {f0_max}!")
                f0_max_round = int(round(f0_max)) # Useful for user-facing things

                # Crop axes
                crop_idxs = [np.argmin(np.abs(f-(f0_max-crop_bw/2))), np.argmin(np.abs(f-(f0_max+crop_bw/2)))+1]
                f_crop = f[crop_idxs[0]:crop_idxs[1]+1]
                psd_crop = psd[crop_idxs[0]:crop_idxs[1]+1]
                # Fit Lorentzian
                f0_fit, y0_l, gamma_L, a_L, lorentz_fit = fit_lorentzian(f_crop, psd_crop)
                # Conversions
                f_crop_khz = f_crop / 1000
                psd_crop_db = psd_db[crop_idxs[0]:crop_idxs[1]]
                lorentz_fit_db = 10*np.log10(lorentz_fit)

                # Plot crop and fit on full PSD
                # plt.plot(f_crop_khz, psd_crop_db, color='yellow')
                plt.figure("full")
                plt.plot(f_crop_khz, lorentz_fit_db, label=f"{f0_max:.0f} Hz", color=color)
                
                # Get point at which lorentzian hits bw_filt_thresh % of total
                fmin_filt, fmax_filt = get_band(f, f0_fit, gamma_L, bw_filt_thresh)

                # Crop to the fit plus some bins (only used for plotting)
                extra_bin_fact = 1
                bin_width = f[1]-f[0]
                bw_filt = fmax_filt-fmin_filt
                extra_bins = int(round((fmax_filt-fmin_filt)*extra_bin_fact/(bin_width)))
                
                # Do the crop
                crop_plus_slice = slice(crop_idxs[0] - extra_bins, crop_idxs[1] + extra_bins)
                f_khz_crop_plus = f_khz[crop_plus_slice]
                psd_crop_plus = psd[crop_plus_slice]

                # Filter and demean
                wf -= np.mean(wf)
                wf_filt = filter_wf(wf, fs, fmin_filt, fmax_filt, p_filt)
                wf_filt -= np.mean(wf_filt)
                psd_filt = pc.get_welch(wf_filt, fs, tau_psd, hop=hop_psd, win=win_type_psd, nfft=nfft_psd)[1]
                psd_filt_db = 10*np.log10(psd_filt)

                # Plot individual fits
                plt.figure(f"{f0_max_round}", figsize=figsize_ind)
                plt.plot(f_khz_crop_plus, psd_crop_plus, label='PSD', color='k', lw=2, alpha=0.7)
                plt.plot(f_crop_khz, lorentz_fit, label="Lorentzian Fit", color='green', lw=5, alpha=0.3)
                plt.plot(f_khz, psd_filt, label="Filtered", color='purple')
                plt.ylabel("PSD")
                plt.legend()
                plt.xlim(f_khz_crop_plus[[0, -1]])
                plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} {f0_max_round} Hz [{filt_id}] - PSD"))

                # Get analytic signal
                wf_filt_h = hilbert(wf_filt)
                wf_filt_h_phi = wf_filt_h / np.abs(wf_filt_h)
                acf = correlate(wf_filt_h, wf_filt_h, mode='full', method='auto')
                acf_phi = correlate(wf_filt_h_phi, wf_filt_h_phi, mode='full', method='auto')
                
                # Get lags and crop both to positive lags
                N = len(wf_filt_h)
                lags = correlation_lags(N, N, mode='full')

                # Crop acf
                max_lag_idx = np.argmin(np.abs(lags-max_lag_s*fs))
                acf_slice = slice(N-1, max_lag_idx+1) # Final lag is max_lag_s (inclusive)
                lags = lags[acf_slice]
                acf = acf[acf_slice]
                acf_phi = acf_phi[acf_slice]

                # Normalize acf 
                num_terms = N-lags
                var = np.abs(acf[0])/N
                acf = np.abs(acf)/(num_terms*var)
                acf_phi = np.abs(acf_phi) / num_terms

                # Convert lags to ms
                lags_s = lags / fs
                lags_ms = lags_s * 1000

                if T_xi_type == "exp":
                    # Crop to fit exponentials
                    acf_crop_slice = slice(np.argmax(acf < acf_exp_fit_max), np.argmax(acf < acf_exp_fit_min))
                    acf_phi_crop_slice = slice(np.argmax(acf_phi < acf_exp_fit_max), np.argmax(acf_phi < acf_exp_fit_min))
                    acf_crop = acf[acf_crop_slice]
                    acf_phi_crop = acf_phi[acf_phi_crop_slice]
                    
                    # Crop lags and convert to s and ms
                    lags_crop = lags[acf_crop_slice]
                    lags_crop_s = lags_crop / fs
                    lags_crop_ms = lags_crop_s * 1000
                    lags_phi_crop = lags[acf_phi_crop_slice]
                    lags_phi_crop_s = lags_phi_crop / fs
                    lags_phi_crop_ms = lags_phi_crop_s * 1000

                    # Fit exponentials
                    A_xi, T_xi, acf_fit = fit_exp(lags_crop_s, acf_crop)
                    A_xi_phi, T_xi_phi, acf_phi_fit = fit_exp(lags_phi_crop_s, acf_phi_crop)

                    # Plot ACF
                    def plotter():
                        plt.plot(lags_ms, acf, label=r"$P$", color='orange')
                        plt.plot(lags_ms, acf_phi, label=r"$\phi$", color='purple')
                        plt.plot(lags_crop_ms, acf_fit, color='orange', path_effects=None, lw=lw_fit, alpha=alpha_fit)
                        plt.plot(lags_phi_crop_ms, acf_phi_fit, color='purple', path_effects=None, lw=lw_fit, alpha=alpha_fit)
                        plt.xlabel(r"$\xi$ [ms]")
                        plt.ylabel(r"$C_\xi$")
                        plt.legend()
                    xmax_ms = (lags_crop_ms[-1])*2
                    row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'P', 'T_xi':T_xi, 'A_xi':A_xi, 'gamma_L':gamma_L, 'a_L':a_L, 'fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
                    row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'phi', 'T_xi':T_xi_phi, 'A_xi':A_xi_phi, 'gamma_L':gamma_L, 'a_L':a_L, 'wf_fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
                    T_xi_id = f"exp_fit_bounds=({acf_exp_fit_min}, {acf_exp_fit_max}), max_lag={max_lag_s}s"
                elif T_xi_type == "eta":
                    T_xi = get_T_xi_eta(acf, lags_s, eta=eta, sig_thresh=sig_thresh_eta)
                    T_xi_phi = get_T_xi_eta(acf_phi, lags_s, eta=eta, sig_thresh=sig_thresh_eta)
                    def plotter():
                        # Plot ACF
                        plt.plot(lags_ms, acf, label=r"$P$", color='orange')
                        plt.plot(lags_ms, acf_phi, label=r"$\phi$", color='purple')
                        plt.vlines(T_xi*1000, 0, 1, color='orange')
                        plt.vlines(T_xi_phi*1000, 0, 1, color='purple')
                        plt.xlabel(r"$\xi$ [ms]")
                        plt.ylabel(r"$C_\xi$")
                        plt.hlines(sig_thresh_eta, 0, max_lag_s*1000, color='green')
                        plt.legend()
                    # if np.max([T_xi, T_xi_phi]) > 0.5:
                    #     xmax_ms = np.max([T_xi, T_xi_phi])*1.5*1000
                    # else:
                    #     xmax_ms = 0.5*1000
                    xmax_ms = np.max([T_xi, T_xi_phi])*2*1000
                    row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'P', 'T_xi':T_xi, 'gamma_L':gamma_L, 'a_L':a_L, 'wf_fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
                    row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'phi', 'T_xi':T_xi_phi, 'gamma_L':gamma_L, 'a_L':a_L, 'wf_fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
                    T_xi_id = f"eta={eta}, sig_thresh={sig_thresh_eta}, max_lag={max_lag_s}s"
                elif T_xi_type == "int":
                    T_xi = get_T_xi_int(acf, lags_s)
                    T_xi_phi = get_T_xi_int(acf_phi, lags_s)
                    def plotter():
                        # Plot ACF
                        plt.plot(lags_ms, acf, label=r"$P$", color='orange')
                        plt.plot(lags_ms, acf_phi, label=r"$\phi$", color='purple')
                        plt.vlines(T_xi*1000, 0, 1, color='orange')
                        plt.vlines(T_xi_phi*1000, 0, 1, color='purple')
                        plt.xlabel(r"$\xi$ [ms]")
                        plt.ylabel(r"$C_\xi$")
                        plt.legend()
                    xmax_ms = np.max([T_xi, T_xi_phi])*2*1000
                    row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'P', 'T_xi':T_xi, 'gamma_L':gamma_L, 'a_L':a_L, 'wf_fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
                    row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'phi', 'T_xi':T_xi_phi, 'gamma_L':gamma_L, 'a_L':a_L, 'wf_fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
                    T_xi_id = f"int {max_lag_s*1000:.0f}ms"

                else:
                    raise ValueError(f"{T_xi_type} is not a valid T_xi_type")
                plt.figure(f"{f0_max_round} ACF", figsize=(12, 8))
                plt.suptitle(f"{species} {wf_idx} - {f0_max_round} Hz")
                plt.subplot(1, 2, 1)
                plotter()
                plt.subplot(1, 2, 2)
                plotter()
                plt.xlim(0, xmax_ms)
                plt.savefig(os.path.join(dirs[f"T_xi_{T_xi_type}"], f"{species} {wf_idx} {f0_max_round} Hz - ACF [{T_xi_id}, {filt_id}].jpg"))

                
                rows.append(row)
                rows.append(row_phi)

                # Save final full figure
                plt.figure("full")
                plt.legend()
                plt.title(filt_id)
                plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} Full PSD [{filt_id}]"))

    df = pd.DataFrame(rows)
    fp_sheets = os.path.join(dirs["results"], f"soae_T_xi [{T_xi_id}, {filt_id}].xlsx")
    df.to_excel(fp_sheets, index=False)
        


        
        



        