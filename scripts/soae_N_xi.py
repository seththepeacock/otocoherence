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

# Parameters

# ---WF---
speciess = ['Tokay', 'Human', 'Owl', 'Anole']
wf_idxs = range(4)
# speciess = ['Anole']
# wf_idxs = [0]
wf_len_s = 60

# ---PSD---
tau_s_psd = 0.5
win_type_psd = 'hann'
hop_s_psd = 0.25

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
acf_fit_min = 0.1
acf_fit_max = 0.9

# Eta cumulative fit
max_lag_s_eta = 10
eta = 0.5

T_xi_type = "eta"

max_lag_s = 1 if T_xi_type == "eta" else 0.1

# ---Lorentzian Fitting---
crop_bw = 200

# ---Filtering---
kaiser_rip = 80
kaiser_df = 50
exp_order = 10
p_filt = {'type':'kaiser', 'rip':kaiser_rip, 'df':kaiser_df}
# p_filt = {'type':'exp', 'order':exp_order}

# Automatic bandwidth calculation
bw_filt_thresh = 0.1

def get_band(f, f0, gamma, bw_thresh=0.1):
    peak = lorentzian(f, f0, y0=0, gamma=gamma, a=1)
    pmp_idx = np.argmin(np.abs(peak-1)) # Get peak midpoint index
    peak_left = peak[0:pmp_idx]
    peak_right = peak[pmp_idx:]
    fmin_idx = np.argmin(np.abs(peak_left-bw_thresh))
    fmax_idx = np.argmin(np.abs(peak_right-bw_thresh)) + pmp_idx
    fmin, fmax = f[[fmin_idx, fmax_idx]]
    return fmin, fmax


def filter_wf(wf, fs, fmin, fmax, p_filt):
    # bw = fmax-fmin
    match p_filt['type']:
        case 'kaiser':
            if p_filt["df"] < 1:
                df = (fmax-fmin) * p_filt["df"]
            else:
                df = p_filt["df"]
            return kaiser_filter(wf, fs, (fmin, fmax), df, p_filt['rip'])
        case 'exp':
            return exp_filter(wf, fs, fmin, fmax, p_filt['order'])



# Initialize spreadsheet rows
rows = []

# ID
match p_filt['type']:
    case 'kaiser':
        if p_filt["df"] < 1:
            df = f"{p_filt["df"]*100:.0f}p bw"
        else:
            df = f"{p_filt["df"]}hz"
        filt_id = f"kaiser, df={df}, rip={p_filt['rip']}db"
    case 'exp':
        filt_id = f"exp, order={p_filt['order']}hz"
fn_id = f"bw={bw_filt_thresh*100:.0f}p max, {filt_id}, crop={crop_bw}hz"

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
        f, psd = pc.get_welch(wf, fs, tau_psd, hop=hop_psd, win=win_type_psd)
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
            psd_filt = pc.get_welch(wf_filt, fs, tau_psd, hop=hop_psd, win=win_type_psd)[1]
            psd_filt_db = 10*np.log10(psd_filt)

            # Plot individual fits
            plt.figure(f"{f0_max_round}", figsize=figsize_ind)
            plt.plot(f_khz_crop_plus, psd_crop_plus, label='PSD', color='k', lw=2, alpha=0.7)
            plt.plot(f_crop_khz, lorentz_fit, label="Lorentzian Fit", color='green', lw=5, alpha=0.3)
            plt.plot(f_khz, psd_filt, label="Filtered", color='purple')
            plt.ylabel("PSD")
            plt.legend()
            plt.xlim(f_khz_crop_plus[[0, -1]])
            plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} {f0_max_round} Hz [{fn_id}] - PSD"))

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
                acf_crop_slice = slice(np.argmin(np.abs(acf-acf_fit_max)), np.argmin(np.abs(acf-acf_fit_min)))
                acf_phi_crop_slice = slice(np.argmin(np.abs(acf_phi-acf_fit_max)), np.argmin(np.abs(acf_phi-acf_fit_min)))
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
                plt.figure(f"{f0_max_round} ACF")
                plt.plot(lags_ms, acf, label=r"$P$", color='orange')
                plt.plot(lags_ms, acf_phi, label=r"$\phi$", color='purple')
                plt.plot(lags_crop_ms, acf_fit, color='orange', path_effects=None, lw=lw_fit, alpha=alpha_fit)
                plt.plot(lags_phi_crop_ms, acf_phi_fit, color='purple', path_effects=None, lw=lw_fit, alpha=alpha_fit)
                plt.xlabel(r"$\xi$ [ms]")
                plt.ylabel(r"$C_\xi$")
                plt.legend()
                plt.savefig(os.path.join(dirs["acf"], f"{species} {wf_idx} {f0_max_round} Hz - ACF (exp) [{fn_id}]"))
                row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'P', 'T_xi':T_xi, 'A_xi':A_xi, 'gamma_L':gamma_L, 'a_L':a_L, 'fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
                row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'phi', 'T_xi':T_xi_phi, 'A_xi':A_xi_phi, 'gamma_L':gamma_L, 'a_L':a_L, 'wf_fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
            elif T_xi_type == "eta":
                T_xi = get_T_xi_eta(acf, lags_s, eta=eta)
                T_xi_phi = get_T_xi_eta(acf_phi, lags_s, eta=eta)
                # Plot ACF
                plt.figure(f"{f0_max_round} ACF")
                plt.plot(lags_ms, acf, label=r"$P$", color='orange')
                plt.plot(lags_ms, acf_phi, label=r"$\phi$", color='purple')
                plt.vlines(T_xi*1000, 0, 1, color='orange')
                plt.vlines(T_xi_phi*1000, 0, 1, color='purple')
                plt.xlabel(r"$\xi$ [ms]")
                plt.ylabel(r"$C_\xi$")
                plt.legend()
                plt.savefig(os.path.join(dirs["acf"], f"{species} {wf_idx} {f0_max_round} Hz - ACF (eta={eta*100:.0f}p, max_lag={max_lag_s}s) [{fn_id}] "))
                
                row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'P', 'T_xi':T_xi, 'gamma_L':gamma_L, 'a_L':a_L, 'wf_fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
                row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'phi', 'T_xi':T_xi_phi, 'gamma_L':gamma_L, 'a_L':a_L, 'wf_fn':wf_fn, 'f0_max':f0_max, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_type}
            else:
                raise ValueError(f"{T_xi_type} is not a valid T_xi_type")
            
            
            rows.append(row)
            rows.append(row_phi)

            # Save final full figure
            plt.figure("full")
            plt.legend()
            plt.title(fn_id)
            plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} Full PSD [{fn_id}]"))

df = pd.DataFrame(rows)
fp_sheets = os.path.join(dirs["results"], "soae_N_xi.xlsx")
df.to_excel(fp_sheets, index=False)
        


        
        



        